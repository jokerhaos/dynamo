# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise real startup imports and interfaces without loading model weights."""

import asyncio
import inspect
import os
import subprocess
import sys
from types import MethodType, SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import sglang as sgl
import yaml
from sglang.srt.entrypoints import engine as engine_module
from sglang.srt.managers.io_struct import GenerateReqInput
from sglang.srt.server_args import ServerArgs

pytestmark = [
    pytest.mark.unit,
    pytest.mark.sglang,
    pytest.mark.core,
    pytest.mark.gpu_0,
    pytest.mark.profiled_vram_gib(0),
    pytest.mark.pre_merge,
]


@pytest.mark.parametrize(
    "argv, expected",
    [
        (["--help"], "Dynamo SGLang worker configuration"),
        (["--version"], "Dynamo Backend SGLang"),
        (["--enable-metrics", "--help"], "Dynamo SGLang worker configuration"),
    ],
)
def test_worker_module_entrypoint(argv, expected, tmp_path):
    # A fresh process is necessary: cached modules and mocked imports in other
    # unit tests must not hide a broken installed worker entry point.
    env = dict(os.environ)
    env.update(
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        PROMETHEUS_MULTIPROC_DIR=str(tmp_path),
    )
    result = subprocess.run(
        [sys.executable, "-m", "dynamo.sglang", *argv],
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=90,
        check=False,
    )
    assert result.returncode == 0, result.stdout
    assert expected in result.stdout


class _ReachedEngine(Exception):
    """Stop exactly where model/GPU initialization would begin."""


@pytest.mark.parametrize("from_config", [False, True])
@pytest.mark.parametrize(
    "mode, speculative", [("prefill", False), ("decode", False), ("decode", True)]
)
@pytest.mark.asyncio
async def test_worker_dispatch_reaches_engine_with_native_args(
    mode, speculative, from_config, tmp_path, monkeypatch
):
    # Marker-only collection uses SGLang stubs. Import the full worker only
    # when this test executes, so collection does not need runtime-only APIs.
    from dynamo.sglang import init_llm
    from dynamo.sglang import main as worker_main

    monkeypatch.chdir(tmp_path)
    (tmp_path / "dummy").mkdir()
    # This is an existing local path, so argument parsing cannot download a
    # model. ServerArgs and the CLI/config-file parser remain real.
    options = {
        "model-path": "dummy",
        "served-model-name": "startup-contract",
        "device": "cuda",
        "disaggregation-mode": mode,
        "tensor-parallel-size": 4,
        "data-parallel-size": 4,
        "enable-dp-attention": True,
        "enable-dp-lm-head": True,
        "quantization": "fp8",
        "context-length": 1048576,
        "mem-fraction-static": 0.85,
        "kv-cache-dtype": "fp8_e4m3",
        "attention-backend": "dsa",
        "dsa-prefill-backend": "trtllm",
        "dsa-decode-backend": "trtllm",
        "enforce-disable-flashinfer-allreduce-fusion": True,
        "disable-custom-all-reduce": True,
        "reasoning-parser": "glm45",
        "skip-tokenizer-init": True,
        "enable-metrics": True,
    }
    if mode == "prefill":
        options.update(
            {
                "ep-size": 4,
                "load-balance-method": "round_robin",
                "moe-runner-backend": "flashinfer_trtllm_routed",
                "chunked-prefill-size": 32768,
                "max-prefill-tokens": 65536,
                "max-running-requests": 64,
                "enable-hierarchical-cache": True,
                "hicache-size": 32,
                "hicache-write-policy": "write_through",
                "hicache-mem-layout": "page_first",
                "hicache-storage-backend": "mooncake",
                "hicache-storage-prefetch-policy": "timeout",
            }
        )
    else:
        options.update(
            {
                "moe-a2a-backend": "none",
                "max-running-requests": 256,
                "cuda-graph-backend-decode": "full",
                "cuda-graph-max-bs-decode": 64,
                "num-continuous-decode-steps": 3,
            }
        )
    if speculative:
        options.update(
            {
                "speculative-algorithm": "EAGLE",
                "speculative-num-steps": 1,
                "speculative-eagle-topk": 1,
                "speculative-num-draft-tokens": 2,
                "speculative-attention-mode": "decode",
                "speculative-moe-runner-backend": "flashinfer_trtllm_routed",
                "speculative-moe-a2a-backend": "none",
            }
        )
    if from_config:
        config_path = tmp_path / "worker.yaml"
        config_path.write_text(yaml.safe_dump(options))
        argv = ["--config", str(config_path)]
    else:
        argv = []
        for name, value in options.items():
            argv.append(f"--{name}")
            if value is not True:
                argv.append(str(value))
    # SGLang owns --config; Dynamo-only options remain on the worker CLI.
    argv.extend(
        [
            "--namespace",
            "startup-contract",
            "--dyn-tool-call-parser",
            "glm47",
            "--dyn-reasoning-parser",
            "glm45",
        ]
    )

    captured = []
    engine_signature = inspect.signature(sgl.Engine)

    def stop_before_loading_model(*, server_args):
        engine_signature.bind(server_args=server_args)
        captured.append(server_args)
        raise _ReachedEngine

    endpoint = SimpleNamespace(connection_id=lambda: 1)
    runtime = SimpleNamespace(endpoint=lambda name: endpoint)
    # Only external runtime/signal setup and the GPU constructor are replaced.
    # worker(), parse_args(), and the selected init function all execute.
    monkeypatch.setattr(
        worker_main,
        "create_runtime",
        lambda **kwargs: (runtime, asyncio.get_running_loop()),
    )
    monkeypatch.setattr(
        worker_main, "prepare_snapshot_engine", AsyncMock(return_value=None)
    )
    monkeypatch.setattr(
        worker_main, "install_graceful_shutdown", lambda *args: AsyncMock()
    )
    monkeypatch.setattr(init_llm.sgl, "Engine", stop_before_loading_model)

    with pytest.raises(_ReachedEngine):
        await worker_main.worker(argv)

    assert len(captured) == 1
    args = captured[0]
    assert isinstance(args, ServerArgs)
    assert args.disaggregation_mode == mode
    assert args.tp_size == 4
    assert args.dp_size == 4
    assert args.incremental_streaming_output is True
    assert args.quantization == "fp8"
    assert args.attention_backend == "dsa"
    assert args.speculative_algorithm == ("EAGLE" if speculative else None)


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.asyncio
async def test_native_generate_accepts_disaggregated_request(stream, monkeypatch):
    captured = []

    async def generate_request(request, http_request):
        captured.append(request)
        assert http_request is None
        yield {"output_ids": [12]}

    adapter = SimpleNamespace(
        tokenizer_manager=SimpleNamespace(generate_request=generate_request)
    )
    adapter._resolve_routed_dp_rank = MethodType(
        sgl.Engine._resolve_routed_dp_rank, adapter
    )
    monkeypatch.setattr(
        engine_module, "get_parallel", lambda: SimpleNamespace(dp_size=4)
    )
    # Execute the real Engine.async_generate and GenerateReqInput constructor;
    # stop at TokenizerManager's scheduler transport instead of doing inference.
    result = await sgl.Engine.async_generate(
        adapter,
        input_ids=[10, 11],
        sampling_params={"temperature": 0.7, "top_p": 0.95, "max_new_tokens": 2},
        stream=stream,
        bootstrap_host="127.0.0.1",
        bootstrap_port=None,
        bootstrap_room=1,
        routed_dp_rank=2,
        rid="startup-contract",
        external_trace_header={
            "traceparent": "00-" + "1" * 32 + "-" + "2" * 16 + "-01"
        },
    )
    if stream:
        result = await anext(result)
    assert result == {"output_ids": [12]}
    request = captured[0]
    assert isinstance(request, GenerateReqInput)
    assert request.input_ids == [10, 11]
    assert request.bootstrap_room == 1
    assert request.routed_dp_rank == 2
    assert request.rid == "startup-contract"
