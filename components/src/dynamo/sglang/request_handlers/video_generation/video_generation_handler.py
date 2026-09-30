# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import base64
import dataclasses
import io
import logging
import os
import random
import time
from typing import Any, AsyncGenerator, Mapping, Optional

import torch

from dynamo._core import Context
from dynamo.common.storage import upload_to_fs
from dynamo.sglang.args import Config
from dynamo.sglang.protocol import (
    CreateVideoRequest,
    VideoData,
    VideoGenerationResponse,
    VideoNvExt,
)
from dynamo.sglang.publisher import DynamoSglangPublisher
from dynamo.sglang.request_handlers.handler_base import BaseGenerativeHandler

logger = logging.getLogger(__name__)


def build_sampling_params_kwargs(
    sampling_params_class: Optional[type],
    transport_args: dict[str, Any],
    passthrough: Optional[Mapping[str, Any]],
    seconds: Optional[int],
) -> dict[str, Any]:
    """Merge transport args into the model's sampling-params contract.

    Models declare their own keywords via dataclass fields. H3-style contracts
    mark timing (num_frames/fps) and CFG fields ``init=False`` — resolved
    server-side — so passing them raises ``TypeError: __init__() got an
    unexpected keyword argument``. Filter the shared transport kwargs to the
    fields the model actually accepts, let the client's media_passthrough
    extensions (task/target/conditions/...) override transport defaults, and
    bridge the OpenAI ``seconds`` into the native duration field on contracts
    that reject frame counts (MiniMax-H3 resolves timing from
    ``target.duration_seconds``).

    ``sampling_params_class=None`` (registry miss, older SGLang) keeps the
    legacy pass-through behavior minus explicitly-None entries — no model
    consumes a None kwarg, and dropping them restores the model's own defaults.
    """
    args = {key: value for key, value in transport_args.items() if value is not None}
    if sampling_params_class is None:
        return args

    accepted = {
        field.name for field in dataclasses.fields(sampling_params_class) if field.init
    }
    args.update(
        {
            key: value
            for key, value in dict(passthrough or {}).items()
            if key in accepted
        }
    )
    args = {key: value for key, value in args.items() if key in accepted}

    accepts_frames = "num_frames" in accepted
    native_duration = "target" in accepted
    if (
        not accepts_frames
        and native_duration
        and "target" not in args
        and seconds is not None
    ):
        # MiniMax-H3's canonical request resolves timing and geometry from the
        # target dict — duration_seconds plus the shape pair. Fill every part
        # the transport actually knows: duration from `seconds`, aspect ratio
        # and short edge from the transport `size`. Anything beyond stays the
        # client's to send via media_passthrough (which lands in `args` above
        # and wins over this default).
        target: dict[str, Any] = {"duration_seconds": int(seconds)}
        width, height = args.get("width"), args.get("height")
        if isinstance(width, int) and isinstance(height, int):
            target["aspect_ratio"] = f"{width}:{height}"
            target["short_edge"] = min(width, height)
        args["target"] = target
    return args


class VideoGenerationWorkerHandler(BaseGenerativeHandler):
    """Handler for video generation (T2V/I2V).

    Inherits from BaseGenerativeHandler for common infrastructure like
    tracing, metrics publishing, and cancellation support.
    """

    def __init__(
        self,
        generator: Any,  # DiffGenerator, not sgl.Engine
        config: Config,
        publisher: Optional[DynamoSglangPublisher] = None,
        fs: Any = None,  # fsspec.AbstractFileSystem for primary storage
    ):
        """Initialize video generation worker handler.

        Args:
            generator: The SGLang DiffGenerator instance.
            config: SGLang and Dynamo configuration.
            publisher: Optional metrics publisher (not used for video currently).
            fs: Optional fsspec filesystem for primary video storage.
        """
        # Call parent constructor for common setup
        super().__init__(config, publisher)

        # Video generation-specific initialization
        self.generator = generator  # DiffGenerator, not Engine
        self._generate_lock = asyncio.Lock()  # Serialize generator access
        self.fs = fs
        self.fs_url = config.dynamo_args.media_output_fs_url
        self.base_url = config.dynamo_args.media_output_http_url

        logger.info(
            f"Video generation worker handler initialized with fs_url={self.fs_url}"
        )

    def cleanup(self) -> None:
        """Cleanup generator resources"""
        if self.generator is not None:
            del self.generator
        torch.cuda.empty_cache()
        logger.info("Video generation generator cleanup complete")
        # Call parent cleanup for base class cleanup
        super().cleanup()

    def _resolve_sampling_params_class(self) -> Optional[type]:
        """The serving model's SamplingParams subclass, when resolvable.

        Uses the same resolution ladder SGLang's own
        ``SamplingParams.from_user_sampling_params_args`` performs:
        ``server_args.pipeline_class_name`` first, then the model index at
        ``model_path`` (under ``model_subfolder`` when the pipeline set one —
        MiniMax-H3's ``_load_config`` assigns it during load). A None return
        keeps ``build_sampling_params_kwargs`` on the legacy pass-through.
        """
        cached = getattr(self, "_sampling_params_class", None)
        if cached is not None or getattr(
            self, "_sampling_params_class_looked_up", False
        ):
            return cached
        self._sampling_params_class_looked_up = True
        try:
            from sglang.multimodal_gen.configs.sample.sampling_params import (
                SamplingParams,
            )
            from sglang.multimodal_gen.registry import get_pipeline_config_classes
        except ImportError:
            return None
        server_args = self.generator.server_args
        class_name = getattr(server_args, "pipeline_class_name", None)
        if class_name:
            try:
                config_classes = get_pipeline_config_classes(class_name)
            except Exception:
                config_classes = None
            if config_classes:
                self._sampling_params_class = config_classes[1]
                return self._sampling_params_class

        # pipeline_class_name is absent on Dynamo's minimal diffusion stub,
        # so resolve the same way SGLang does: read the model's own index and
        # ask the registry. Prefer the partition folder the pipeline loader
        # recorded (model_subfolder) over the bare repository root — the root
        # layout of split checkpoints can carry a different class name.
        try:
            model_path = getattr(server_args, "model_path", None)
            subfolder = getattr(server_args, "model_subfolder", None)
            if model_path and subfolder:
                resolved: Any = SamplingParams.from_pretrained(
                    os.path.join(model_path, subfolder),
                    backend=getattr(server_args, "backend", None),
                    model_id=getattr(server_args, "model_id", None),
                )
                self._sampling_params_class = type(resolved)
                return self._sampling_params_class
            if model_path:
                resolved: Any = SamplingParams.from_pretrained(
                    model_path,
                    backend=getattr(server_args, "backend", None),
                    model_id=getattr(server_args, "model_id", None),
                )
                self._sampling_params_class = type(resolved)
                return self._sampling_params_class
        except Exception:
            logger.debug(
                "SamplingParams.from_pretrained resolution failed", exc_info=True
            )
        return None

    async def generate(
        self, request: dict[str, Any], context: Context
    ) -> AsyncGenerator[dict[str, Any], None]:
        """
        Generate video from text/image prompt.

        Unlike LLM streaming, video returns complete video at end.

        Args:
            request: Request dict with prompt and generation parameters.
            context: Context object for cancellation handling.

        Yields:
            Response dict with generated video (OpenAI-compatible format).
        """
        start_time = time.time()

        # Get trace header for distributed tracing (for logging/observability)
        trace_header = context.trace_headers() if self.enable_trace else None
        if trace_header:
            logger.debug(f"Video generation request with trace: {trace_header}")

        try:
            req = CreateVideoRequest(**request)
            nvext = req.nvext or VideoNvExt()

            logger.info(
                f"Video generation request: model={req.model}, "
                f"size={req.size}, steps={nvext.num_inference_steps}"
            )

            # Parse size
            if req.size is None:
                raise ValueError("Size is required")
            width, height = self._parse_size(req.size)

            # Transport timing: fps only feeds the frame-count math and the
            # output encode rate here; models that resolve canonical timing
            # server-side drop num_frames in build_sampling_params_kwargs.
            fps = nvext.fps or 24
            if nvext.num_frames is not None:
                num_frames = nvext.num_frames
            elif req.seconds is not None:
                num_frames = fps * req.seconds
            else:
                raise ValueError("Seconds or nvext.num_frames is required")

            # Generate video
            context_id = context.id()
            if context_id is None:
                raise ValueError("Context ID is required")
            if nvext.num_inference_steps is None:
                raise ValueError("Num inference steps is required")
            output_format = req.output_format or "mp4"
            if output_format != "mp4":
                raise ValueError(
                    f"Unsupported output_format: {output_format!r}; only 'mp4' is supported"
                )
            video_bytes = await self._generate_video(
                prompt=req.prompt,
                width=width,
                height=height,
                num_frames=num_frames,
                fps=fps,
                seconds=req.seconds,
                num_inference_steps=nvext.num_inference_steps,
                guidance_scale=nvext.guidance_scale,
                seed=nvext.seed,
                request_id=context_id,
                negative_prompt=nvext.negative_prompt,
                input_reference=req.input_reference,
                passthrough=(req.extra_args or {}).get("media_passthrough"),
            )

            video_data = []
            response_format = req.response_format or "b64_json"
            if response_format == "url":
                url = await self._upload_to_fs(video_bytes, context_id)
                video_data.append(VideoData(output_format=output_format, url=url))
            elif response_format == "b64_json":
                b64 = self._encode_base64(video_bytes)
                video_data.append(VideoData(output_format=output_format, b64_json=b64))
            else:
                raise ValueError(
                    f"Unsupported response_format: {response_format!r}; expected 'url' or 'b64_json'"
                )

            inference_time = time.time() - start_time

            response = VideoGenerationResponse(
                id=f"video-{context.id()}",
                model=req.model,
                created=int(time.time()),
                data=video_data,
                inference_time_s=inference_time,
            )

            yield response.model_dump()

        except Exception as e:
            logger.error(f"Error in video generation: {e}", exc_info=True)
            # Return error response
            error_response = VideoGenerationResponse(
                id=f"video-{context.id()}",
                model=request.get("model", "unknown"),
                created=int(time.time()),
                status="failed",
                progress=0,
                data=[],
                error=str(e),
            )
            yield error_response.model_dump()

    async def _generate_video(
        self,
        prompt: str,
        width: int,
        height: int,
        num_frames: int,
        fps: int,
        num_inference_steps: int,
        guidance_scale: float,
        seed: Optional[int],
        request_id: str,
        negative_prompt: Optional[str] = None,
        input_reference: Optional[str] = None,
        passthrough: Optional[Mapping[str, Any]] = None,
        seconds: Optional[int] = None,
    ) -> bytes:
        """Generate video using SGLang DiffGenerator.

        Args:
            prompt: Text prompt for video generation.
            width: Video width in pixels.
            height: Video height in pixels.
            num_frames: Number of frames to generate.
            fps: Frames per second for output video.
            num_inference_steps: Number of denoising steps.
            guidance_scale: CFG scale for generation.
            seed: Random seed for reproducibility.
            request_id: Request ID for logging.
            negative_prompt: Optional negative prompt.
            input_reference: Optional image path for I2V.

        Returns:
            Video bytes (mp4 format).
        """
        # Build args for DiffGenerator
        args = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "height": height,
            "width": width,
            "num_frames": num_frames,
            "fps": fps,
            "num_inference_steps": num_inference_steps,
            "save_output": False,  # We handle saving ourselves
            "guidance_scale": guidance_scale,
            "seed": seed if seed is not None else random.randint(0, 1000000),
        }

        # Add image_path for I2V if provided
        if input_reference:
            args["image_path"] = input_reference

        logger.info(
            f"Generating video with {num_frames} frames at {width}x{height}, "
            f"{num_inference_steps} steps, request_id={request_id}"
        )

        # Serialize access -- DiffGenerator has mutable state (CUDA graph
        # caches, shared config objects) and is not thread-safe.
        async with self._generate_lock:
            # Run in thread pool to avoid blocking event loop
            result = await asyncio.to_thread(
                self.generator.generate,
                sampling_params_kwargs=build_sampling_params_kwargs(
                    self._resolve_sampling_params_class(),
                    args,
                    passthrough,
                    seconds,
                ),
            )

        # DiffGenerator.generate() returns GenerationResult | list[GenerationResult] | None
        if result is None:
            raise RuntimeError("DiffGenerator returned None")
        if isinstance(result, list):
            result = result[0]
        frames = result.frames
        if not frames:
            raise RuntimeError("DiffGenerator returned no frames")

        # Convert frames to video bytes
        video_bytes = await self._frames_to_video(frames, fps)
        return video_bytes

    async def _frames_to_video(
        self, frames: list, fps: int, codec: str = "libvpx-vp9"
    ) -> bytes:
        """Convert list of frames to video bytes.

        Args:
            frames: List of frames (PIL Images or numpy arrays).
            fps: Frames per second.
            codec: Video codec to use.

        Returns:
            Video bytes in mp4 format.
        """
        try:
            import numpy as np
            from PIL import Image

            # Convert frames to numpy arrays if needed
            np_frames = []
            for frame in frames:
                if isinstance(frame, Image.Image):
                    np_frames.append(np.array(frame))
                elif isinstance(frame, np.ndarray):
                    np_frames.append(frame)
                else:
                    raise ValueError(f"Unsupported frame type: {type(frame)}")

            import imageio

            def encode_with_codec(codec_name: str) -> bytes:
                output_buffer = io.BytesIO()
                with imageio.get_writer(
                    output_buffer,
                    format="mp4",  # type: ignore
                    fps=fps,
                    codec=codec_name,
                    output_params=["-pix_fmt", "yuv420p"],
                ) as writer:
                    for frame in np_frames:
                        writer.append_data(frame)  # type: ignore

                output_buffer.seek(0)
                return output_buffer.read()

            # VP9 (libvpx-vp9) is a royalty-free CPU encoder in the in-tree LGPL
            # ffmpeg; no HW/GPU fallback and no libx264 (GPL, H.264) fallback.
            return encode_with_codec(codec)

        except ImportError as e:
            raise RuntimeError(
                f"Missing dependency for video encoding: {e}. "
                "Install with: pip install imageio imageio-ffmpeg"
            )

    def _parse_size(self, size_str: str) -> tuple[int, int]:
        """Parse 'WxH' -> (width, height)"""
        try:
            w, h = size_str.split("x")
            return int(w), int(h)
        except (ValueError, TypeError) as e:
            raise ValueError(
                f"Invalid size format '{size_str}', expected 'WxH' (e.g. '832x480')"
            ) from e

    async def _upload_to_fs(self, video_bytes: bytes, request_id: str) -> str:
        """Upload video to filesystem and return URL.

        Args:
            video_bytes: Video data as bytes.
            request_id: Request context ID.

        Returns:
            URL for the uploaded video.
        """
        storage_path = f"{request_id}.mp4"
        return await upload_to_fs(self.fs, storage_path, video_bytes, self.base_url)

    def _encode_base64(self, video_bytes: bytes) -> str:
        """Encode video as base64 string"""
        return base64.b64encode(video_bytes).decode("utf-8")
