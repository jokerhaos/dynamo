{#
# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#}
# === BEGIN templates/sglang_runtime.Dockerfile ===
##################################
########## Runtime Image #########
##################################

{# Internal full-media variant (astro-web3 fork). When
   context.sglang.enable_full_media_stack == "true" this template keeps the
   SGLang base's Ubuntu apt media stack (full-featured /usr/bin/ffmpeg +
   /usr/bin/ffprobe) instead of purging to the VP9-only in-tree build:
   MiniMax-H3's output encode (sglang multimodal_gen entrypoints) hard-codes
   libx264 + AAC, its reference-material decode runs through the ffmpeg CLI,
   and ffprobe must be on PATH. Every other render — including this file's
   own xpu device branch and the vllm/trtllm frameworks — keeps the strict
   upstream posture unchanged (this flag is only set in the sglang section
   of container/context.yaml). #}
{% set full_media = framework == "sglang" and device == "cuda" and context.sglang.get("enable_full_media_stack", "false") == "true" %}

{% if device == "xpu" %}
FROM framework AS runtime
{% else %}
FROM ${RUNTIME_IMAGE}:${RUNTIME_IMAGE_TAG} AS pre_runtime
{% endif %}

ARG MODELEXPRESS_VERSION
{% if device == "cuda" %}
ARG CUDA_MAJOR
{% endif %}

{% if device == "cuda" and cuda_version == "13.0" %}
# Reuse the already-built SGLang source revision containing AIV-883/885,
# while replacing its older Mooncake wheel with the native tracing release.
ARG MOONCAKE_CUDA13_VERSION=0.3.13.post2
ARG MOONCAKE_FIND_LINKS=https://minio.0xtoken.cn/python-packages/simple/mooncake-transfer-engine-cuda13/index.html
RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install --break-system-packages --no-deps --no-index \
        --find-links "${MOONCAKE_FIND_LINKS}" \
        "mooncake-transfer-engine-cuda13==${MOONCAKE_CUDA13_VERSION}"
{% endif %}

WORKDIR /workspace

# Install NATS and ETCD
COPY --from=dynamo_base /usr/bin/nats-server /usr/bin/nats-server
COPY --from=dynamo_base /usr/local/bin/etcd/ /usr/local/bin/etcd/

ENV PATH=/usr/local/bin/etcd:$PATH

{% if device == "cuda" %}
# Install the TurboJPEG runtime used by frontend JPEG decoding and bring
# base-image OS packages up to the current patch releases. --only-upgrade skips
# anything not already installed while keeping both operations in one layer.
# libjemalloc2 lets Dynamo processes opt into jemalloc via
# LD_PRELOAD or DYN_FRONTEND_JEMALLOC; it is not preloaded by default.
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        libturbojpeg \
        libjemalloc2 && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends --only-upgrade \
        dirmngr \
        gnupg \
        gnupg-utils \
        gnupg2 \
        gpg \
        gpg-agent \
        gpgconf \
        gpgsm \
        gpgv \
        keyboxd \
        libssl3t64 \
        openssl && \
    rm -rf /var/lib/apt/lists/* && \
    ldconfig && \
    ldconfig -p | grep -q 'libturbojpeg.so.0'
{% else %}
# Install the TurboJPEG runtime used by frontend JPEG decoding.
# libjemalloc2 lets Dynamo processes opt into jemalloc via
# LD_PRELOAD or DYN_FRONTEND_JEMALLOC; it is not preloaded by default.
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        libturbojpeg \
        libjemalloc2 && \
    rm -rf /var/lib/apt/lists/* && \
    ldconfig && \
    ldconfig -p | grep -q 'libturbojpeg.so.0'
{% endif %}

# Create dynamo user with group 0 for OpenShift compatibility
RUN userdel -r ubuntu > /dev/null 2>&1 || true \
    && useradd -m -s /bin/bash -g 0 dynamo \
    && [ `id -u dynamo` -eq 1000 ] \
    && mkdir -p /home/dynamo/.cache /opt/dynamo \
    # Non-recursive chown - only the directories themselves, not contents
    && chown dynamo:0 /home/dynamo /home/dynamo/.cache /opt/dynamo /workspace \
    # No chmod needed: umask 002 handles new files, COPY --chmod handles copied content
    # Set umask globally for all subsequent RUN commands (must be done as root before USER dynamo)
    # NOTE: Setting ENV UMASK=002 does NOT work - umask is a shell builtin, not an environment variable
    && mkdir -p /etc/profile.d && echo 'umask 002' > /etc/profile.d/00-umask.sh

RUN SITE_PACKAGES="$(python3 -c 'import site; print(site.getsitepackages()[0])')" && \
    CUBINS_DIR="$SITE_PACKAGES/flashinfer_cubin/cubins" && \
    if [ -d "$CUBINS_DIR" ]; then \
        find "$CUBINS_DIR" -type d -exec chmod g+rwx {} + ; \
    fi

{% if device == "xpu" %}
{# XPU runtime: NIXL + UCX are needed for P2P transport on Intel GPUs.
   CUDA sglang runtime does NOT include NIXL/UCX (matching upstream main);
   those are only added in the dev stage for build-time linking. #}
ENV NIXL_PREFIX=/opt/intel/intel_nixl
ENV NIXL_LIB_DIR=$NIXL_PREFIX/lib/x86_64-linux-gnu
ENV NIXL_PLUGIN_DIR=$NIXL_LIB_DIR/plugins

# Copy UCX and NIXL from wheel_builder
COPY --from=wheel_builder /usr/local/ucx /usr/local/ucx
COPY --chown=dynamo:0 --from=wheel_builder $NIXL_PREFIX $NIXL_PREFIX
COPY --chown=dynamo:0 --from=wheel_builder /opt/intel/intel_nixl/lib/x86_64-linux-gnu/. ${NIXL_LIB_DIR}/

COPY --chown=dynamo:0 --from=wheel_builder /opt/dynamo/dist/nixl/ /opt/dynamo/wheelhouse/nixl/
COPY --chown=dynamo:0 --from=wheel_builder /workspace/nixl/build/src/bindings/python/nixl-meta/nixl-*.whl /opt/dynamo/wheelhouse/nixl/

ENV PATH=/usr/local/ucx/bin:$PATH

ENV LD_LIBRARY_PATH=\
$NIXL_LIB_DIR:\
$NIXL_PLUGIN_DIR:\
/usr/local/ucx/lib:\
/usr/local/ucx/lib/ucx:\
${LD_LIBRARY_PATH:-}
{% endif %}

{% if target not in ("dev", "local-dev") %}
# Runtime target installs only the prebuilt Dynamo wheels. SGLang and its NIXL
# packages come from the upstream lmsysorg/sglang runtime image; --no-deps keeps
# pip from replacing that stack. Dev/local-dev build from source later in the
# shared dev stage after the workspace is bind-mounted.
COPY --chmod=775 --chown=dynamo:0 --from=wheel_builder /opt/dynamo/dist/*.whl /opt/dynamo/wheelhouse/

{% if device == "xpu" %}
RUN pip install --no-deps \
        /opt/dynamo/wheelhouse/ai_dynamo_runtime*.whl \
        /opt/dynamo/wheelhouse/ai_dynamo*any.whl \
        /opt/dynamo/wheelhouse/aisimulate*.whl \
        /opt/dynamo/wheelhouse/nixl/nixl*.whl \
        "distro==1.9.0"
{% else %}
RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    export PIP_CACHE_DIR=/root/.cache/pip && \
    pip install --break-system-packages --no-deps \
        /opt/dynamo/wheelhouse/ai_dynamo_runtime*.whl \
        /opt/dynamo/wheelhouse/ai_dynamo*any.whl \
        /opt/dynamo/wheelhouse/aisimulate*.whl

# Install accelerate for diffusion/video worker pipelines (diffusers requires it
# for enable_model_cpu_offload but the upstream SGLang runtime image omits it)
RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    export PIP_CACHE_DIR=/root/.cache/pip && \
    pip install --break-system-packages --no-deps "accelerate==1.13.0"

# Install distro: openai>=1.x's _base_client imports it unconditionally, and
# SGLang server_args eagerly imports sglang.srt.entrypoints.openai.protocol
# which pulls in openai.types.responses → triggers openai pkg init → import distro.
# The upstream lmsysorg/sglang runtime installs openai with --no-deps so distro is
# missing; without this any dynamo.sglang worker fails to import at startup.
RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    export PIP_CACHE_DIR=/root/.cache/pip && \
    pip install --break-system-packages --no-deps "distro==1.9.0"

{% if device == "cuda" and cuda_version == "13.0" %}
# Install the s3 backend for dynamo.common.storage.get_fs (--media-output-fs-url
# s3://; for us the OSS S3-compatibility entrance). s3fs pins fsspec ==2026.6.0,
# exactly the version this image ships. aiobotocore==3.9.1 pins botocore==1.43.75,
# which replaces the newer botocore inherited from the SGLang base -- the pairing
# field-validated 2026-09-30 against OSS on b300-1 (virtual-host addressing and
# checksum config still come from the FSSPEC_S3_* env vars, documented in the
# deployment DGD). Installed WITH deps so the aioitertools closure ships; nothing
# on the serving paths imports boto3 directly. Replaces the pip half of the
# media-deps initContainer in the minimaxh3 DGD.
RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    export PIP_CACHE_DIR=/root/.cache/pip && \
    pip install --break-system-packages "s3fs==2026.6.0" "aiobotocore==3.9.1"
{% endif %}

# Install gpu_memory_service wheel if enabled (all targets)
ARG ENABLE_GPU_MEMORY_SERVICE
RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    if [ "${ENABLE_GPU_MEMORY_SERVICE}" = "true" ]; then \
        export PIP_CACHE_DIR=/root/.cache/pip && \
        GMS_WHEEL=$(ls /opt/dynamo/wheelhouse/gpu_memory_service*.whl 2>/dev/null | head -1); \
        if [ -n "$GMS_WHEEL" ]; then pip install --no-cache-dir --break-system-packages "$GMS_WHEEL"; fi; \
    fi

{% if context.sglang.enable_modelexpress == "true" %}
# Install only the ModelExpress client package. --no-deps preserves the upstream
# SGLang runtime dependency stack. google-crc32c is imported eagerly by the MX
# sglang loader (>=0.5.0) and is not in the SGLang base image, so install it
# alongside; the import check below fails the build on any future --no-deps gap.
RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    set -eux; \
    export PIP_CACHE_DIR=/root/.cache/pip; \
    pip install --break-system-packages --no-deps \
        "modelexpress==${MODELEXPRESS_VERSION}"; \
    pip install --break-system-packages "google-crc32c>=1.5.0"; \
    python3 -c "import modelexpress.engines.sglang"
{% endif %}
{% endif %}
{% endif %}

# Install nvtx pinned in container/deps/requirements.common.txt so DYN_NVTX=1
# profiling works in all targets (runtime, dev, local-dev) — see
# components/src/dynamo/common/utils/nvtx_utils.py. --no-deps preserves the
# upstream lmsysorg/sglang Python stack.
RUN --mount=type=bind,source=./container/deps/requirements.common.txt,target=/tmp/requirements.common.txt \
    --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    export PIP_CACHE_DIR=/root/.cache/pip && \
    pip install --break-system-packages --no-deps $(grep -E '^nvtx==' /tmp/requirements.common.txt)

# Install SGLang-specific runtime dependencies without changing the upstream
# dependency solution. imageio-ffmpeg is installed from source (no bundled
# binary) for the VP9 video-encode path; see requirements.sglang.txt.
{% if device == "cuda" %}
RUN --mount=type=bind,source=./container/deps/requirements.sglang.txt,target=/tmp/requirements.sglang.txt \
    --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    export PIP_CACHE_DIR=/root/.cache/pip && \
    [ "$CUDA_MAJOR" = "13" ] || { echo "ERROR: requirements.sglang.txt hardcodes the mooncake-transfer-engine-cuda13 distribution; got CUDA_MAJOR=$CUDA_MAJOR" >&2; exit 1; } && \
    pip install --break-system-packages --force-reinstall --no-deps \
        --requirement /tmp/requirements.sglang.txt

# Assert what the install left. The pin is repeated from the requirements file
# on purpose; a test asserts the two agree.
RUN --mount=type=bind,source=./container/compliance,target=/tmp/compliance/compliance \
    PYTHONPATH=/tmp/compliance python3 -m compliance.check_pynvvideocodec --pinned 2.2.3
{% else %}
# mooncake and PyNvVideoCodec are CUDA-only. The mooncake floor names the CUDA 13
# distribution, and PyNvVideoCodec decodes on NVDEC through libnvcuvid, so both
# are inert on the XPU image and neither is present in its base. The patterns are
# anchored to the line start so they cannot match inside another requirement, and
# the checks fail the build if either package arrives by another route -- a filter
# that silently stopped matching would otherwise look like success. Both checks
# are positive tests with no `!` and no stderr redirect, so a broken interpreter
# fails the build instead of passing it vacuously.
#
# Whole-RUN branches, rather than a conditional inside one RUN: a `{% raw %}{% if %}{% endraw %}` in the
# middle of a `\`-continued command emits a blank line that ends the command
# early. Matches the equivalent branch in vllm_runtime.Dockerfile.
RUN --mount=type=bind,source=./container/deps/requirements.sglang.txt,target=/tmp/requirements.sglang.txt \
    --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    export PIP_CACHE_DIR=/root/.cache/pip && \
    grep -v -e '^PyNvVideoCodec' -e '^mooncake-transfer-engine-cuda13' \
        /tmp/requirements.sglang.txt > /tmp/requirements.sglang.nonvidia.txt && \
    pip install --break-system-packages --force-reinstall --no-deps \
        --requirement /tmp/requirements.sglang.nonvidia.txt && \
    rm -f /tmp/requirements.sglang.nonvidia.txt && \
    python3 -c "import importlib.util,sys; sys.exit(1 if importlib.util.find_spec('PyNvVideoCodec') else 0)" && \
    python3 -c "import importlib.metadata as m, re, sys; names={re.sub(r'[-_.]+', '-', n).lower() for d in m.distributions() if (n := (d.metadata or {}).get('Name'))}; sys.exit(1 if 'mooncake-transfer-engine-cuda13' in names else 0)"
{% endif %}

# Remove the codec-bearing video-DECODE components from the upstream SGLang image
# (PyAV, decord, OpenCV, torchcodec + any base ffmpeg/libav*), then copy the
# VP9-only in-tree ffmpeg from wheel_builder below for the video-generation
# encode path. H.264/H.265/AAC encoders must never appear (guarded at the end).
#
# Inkling image preprocessing uses Pillow. Its audio feature extractor imports
# soundfile and uses torchaudio only for resampling, so those paths remain
# available for formats supported by libsndfile (for example WAV and FLAC).
# AAC-backed M4A stays removed; video encode is VP9 only.
# Transitional: this exists only until the base image stops shipping these.
# The permanent statement of what may ship is
# tests/dependencies/test_no_software_video_codecs.py -- when this purge
# becomes redundant, delete it and keep those assertions.
#
# CUDA only, and it has to be: the replacement ffmpeg is copied in under
# `device == "cuda"` just below. Running the purge on XPU strips the base image's
# media stack and puts nothing back, leaving that image with no ffmpeg at all and
# an empty IMAGEIO_FFMPEG_EXE -- worse than either leaving it alone or replacing
# it properly. The XPU image is not published, so there is nothing to remove from
# it in the first place; the codec gate skips it for the same reason. Matches the
# equivalent purge in vllm_runtime.Dockerfile.
{% if device == "cuda" %}
RUN set -eux; \
{%- if not full_media %}
    # SGLang 0.5.21's runtime image installs Ubuntu's full GPL/LGPL ffmpeg
    # dependency closure. Remove the packages (and their dpkg metadata) before
    # copying Dynamo's separately built VP9-only ffmpeg below. File deletion
    # alone is insufficient because the compliance generator inventories dpkg.
    # Strict-only: the fork's full-media variant keeps the apt media stack
    # (and the fork-built base may ship neither package at all, which would
    # fail the purge).
    apt-get purge -y --auto-remove \
        ffmpeg \
        libwayland-server0; \
    rm -rf /var/lib/apt/lists/*; \
{%- endif %}
    python3 -m pip uninstall --yes \
        av \
        decord \
        decord2 \
        opencv-python \
        opencv-python-headless \
        torchcodec; \
    SITE_PACKAGES="$(python3 -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"; \
    rm -rf \
        "${SITE_PACKAGES}"/av \
        "${SITE_PACKAGES}"/av-*.dist-info \
        "${SITE_PACKAGES}"/av.libs \
        "${SITE_PACKAGES}"/cv2 \
        "${SITE_PACKAGES}"/decord \
        "${SITE_PACKAGES}"/decord-*.dist-info \
        "${SITE_PACKAGES}"/decord.libs \
        "${SITE_PACKAGES}"/decord2 \
        "${SITE_PACKAGES}"/decord2-*.dist-info \
        "${SITE_PACKAGES}"/decord2.libs \
        "${SITE_PACKAGES}"/opencv_python*.dist-info \
        "${SITE_PACKAGES}"/opencv_python*.libs \
        "${SITE_PACKAGES}"/torchcodec \
        "${SITE_PACKAGES}"/torchcodec-*.dist-info \
        /usr/local/bin/ffmpeg \
        /usr/local/bin/ffprobe \
        /usr/local/include/libav* \
        /usr/local/include/libsw* \
        /usr/local/lib/libav* \
        /usr/local/lib/libpostproc* \
        /usr/local/lib/libsw* \
        /usr/local/lib/pkgconfig/libav*.pc \
        /usr/local/lib/pkgconfig/libpostproc*.pc \
        /usr/local/lib/pkgconfig/libsw*.pc \
        /usr/local/src/ffmpeg \
        /root/.cache/pip; \
{%- if not full_media %}
    find /usr /opt /workspace /sgl-workspace -xdev \
        \( -type f -o -type l \) \
        \( -name 'libx264*.so*' -o -name 'libx265*.so*' \
        -o -name 'libopenh264*.so*' -o -name 'libfdk-aac*.so*' \
        -o -name 'libfaac*.so*' -o -name 'libvo-aacenc*.so*' \
        -o -name 'libaacplus*.so*' \) -delete; \
{%- endif %}
    ldconfig

{% if device == "cuda" and not full_media %}
# The SGLang runtime base ships apt-installed ffmpeg so its torchcodec
# (SGLang's own video-input decode backend) can dlopen the system libav* at
# import. Ubuntu's ffmpeg links GPL libx264/libx265 (trips the codec guard
# below) AND its dependency closure drags in ~34 copyleft media peripherals
# (libpulse, libsdl2, libjack, libzvbi, ...) that the license gate in the
# licenses stage denies. The published Dynamo image carries no software
# H.264/H.265/AAC codec implementation and no third-party libav*
# (codec_policy.yaml + test_no_software_video_codecs.py): encode is the
# in-tree VP9 ffmpeg copied above, and video-input decode runs on NVDEC via
# PyNvVideoCodec, which bundles only libavutil + libavformat and needs no
# system ffmpeg. Purge the base's apt media stack; the guards below assert
# the result. Full package-name prefixes so libavahi-* never matches, while
# digit suffixes can drift across base bumps (libavcodec60 vs 58). No
# autoremove — a bare sweep of auto-installed packages would take the
# toolchain and CUDA math libs with it; see the equivalent purge in
# vllm_runtime.Dockerfile. libva/libvdpau stay behind: license-clean,
# owned by the graphics stack, not the media stack. CUDA only, same gating
# as the purge above: the XPU image is not published and keeps its stack.
RUN set -eux; \
    purge="$(dpkg-query -W -f='${Package}\n' 2>/dev/null \
        | grep -E '^(ffmpeg|libaom|libass[0-9]|libavc1394|libavcodec|libavdevice|libavfilter|libavformat|libavutil|libbluray|libbs2b|libcaca|libcdio|libchromaprint|libcodec2|libdav1d|libdc1394|libfdk-aac|libflite|libgme|libgsm|libiec61883|libjack|libmp3lame|libopencore|libopus|libpocketsphinx|libpostproc|libpulse|libraw1394|librubberband|libsdl[0-9]|libshine|libsndfile|libsphinxbase|libsrt[0-9]|libsoxr|libsvtav1|libswresample|libswscale|libtheora|libtwolame|libudfread|libvidstab|libvorbis|libvpx|libx264|libx265|libxvidcore|libzvbi)' \
        || true)"; \
    if [ -n "${purge}" ]; then \
        echo "purging apt media packages from SGLang base:" ${purge}; \
        DEBIAN_FRONTEND=noninteractive apt-get purge -y ${purge}; \
    fi; \
    ldconfig
{% elif full_media %}
# Full-media variant: the apt media purge above is deliberately SKIPPED.
# This image ships the SGLang base's Ubuntu ffmpeg (libx264 + AAC encoders,
# full decoder set, /usr/bin/ffprobe) as its media backend — that is the
# variant's whole point. The per-package license waivers live in
# licenses.toml and the codec scan runs against
# policy/codec_policy_full_media.yaml; the strict guards below (in-tree
# VP9 copy, codec-encoders check, find-scan) are also skipped. The wheel-level
# purge above still runs: PyAV/decord/opencv/torchcodec stay out, and any
# /usr/local ffmpeg the base carries is removed so /usr/bin/ffmpeg is the
# single CLI on PATH.
RUN set -eux; \
    ffmpeg_bin="$(command -v ffmpeg)"; \
    case "${ffmpeg_bin}" in /usr/bin/ffmpeg) ;; *) echo "full-media variant expects apt ffmpeg at /usr/bin/ffmpeg, found: ${ffmpeg_bin:-none}" >&2; exit 1;; esac; \
    test -x /usr/bin/ffprobe
{% endif %}
{% endif %}

# Drop the Nsight efa_metrics plugin the CUDA floor carries: a Go NIC sampler
# nothing in the serving path loads. On this image it arrives under Nsight
# Compute rather than Nsight Systems, and both roots move with every base bump,
# so the paths are globbed and the removal is asserted rather than pinned. This
# stage has no overlay rebase -- `runtime` is FROM pre_runtime -- so one
# deletion here ships.
RUN rm -rf \
        /usr/local/cuda-*/NsightSystems-cli-*/target-linux-*/plugins/efa_metrics \
        /opt/nvidia/nsight-systems-cli/*/target-linux-*/plugins/efa_metrics \
        /opt/nvidia/nsight-compute/*/host/target-linux-*/plugins/efa_metrics && \
    [ -z "$(find /usr/local /opt -xdev -type d -name efa_metrics 2>/dev/null)" ]

{% if device == "cuda" and not full_media %}
# Copy the in-tree VP9 ffmpeg from wheel_builder: versioned shared libs
# (libav*.so*, libsw*.so*) + libvpx + the in-tree CLI binary that imageio targets
# via IMAGEIO_FFMPEG_EXE. The upstream ffmpeg was purged above, so this
# VP9-only build is the only ffmpeg present; the video-generation
# handler (CUDA DiffGenerator) encodes libvpx-vp9 with it. Same pattern as
# vllm_runtime.Dockerfile.
RUN --mount=type=bind,from=wheel_builder,source=/usr/local/,target=/tmp/usr/local/ \
    mkdir -p /usr/local/lib/pkgconfig && \
    cp -rnL /tmp/usr/local/include/libav* /tmp/usr/local/include/libsw* /usr/local/include/ && \
    cp -nL /tmp/usr/local/lib/libav*.so* /tmp/usr/local/lib/libsw*.so* /usr/local/lib/ && \
    cp -nL /tmp/usr/local/lib/lib*vpx*.so* /usr/local/lib/ 2>/dev/null || true && \
    cp -nL /tmp/usr/local/lib/pkgconfig/libav*.pc /tmp/usr/local/lib/pkgconfig/libsw*.pc /usr/local/lib/pkgconfig/ && \
    cp -nL /tmp/usr/local/bin/ffmpeg /usr/local/bin/ffmpeg && \
    cp -r /tmp/usr/local/src/ffmpeg /usr/local/src/ && \
    ldconfig
ENV IMAGEIO_FFMPEG_EXE=/usr/local/bin/ffmpeg

# Positive codec guard: the shipped ffmpeg MUST expose the VP9 encoder and MUST
# NOT expose any H.264/H.265/AAC/NVENC encoder. A missing/broken copy (no VP9)
# or a codec regression fails the build here rather than at runtime — closing the
# gap where SGLang shipping no encoder passed every PR gate.
RUN set -eu; \
    ff="${IMAGEIO_FFMPEG_EXE:-ffmpeg}"; \
    "$ff" -hide_banner -encoders 2>/dev/null | grep -qiE 'libvpx[-_]vp9' \
      || { echo "ERROR: shipped ffmpeg ($ff) has no VP9 encoder" >&2; exit 1; }; \
    if "$ff" -hide_banner -encoders 2>/dev/null \
         | grep -iE 'h\.?264|h\.?265|hevc|(^| )aac|nvenc|cuvid|nvdec'; then \
        echo "ERROR: shipped ffmpeg ($ff) exposes an H.264/H.265/AAC/NVENC encoder" >&2; \
        exit 1; \
    fi

{% if context.sglang.enable_media_ffmpeg == "true" %}
# Frontend video decoding is part of the shipped SGLang CUDA contract. Fail the
# image build if the runtime wheel was accidentally compiled without it.
{% if target not in ("dev", "local-dev") %}
RUN python3 -c 'from dynamo.llm import MediaDecoder; assert hasattr(MediaDecoder(), "enable_video")'
{% endif %}
{% endif %}
{% elif full_media %}
# Full-media variant: no in-tree VP9 copy. imageio and SGLang's
# _resolve_ffmpeg_exe() (which prefers IMAGEIO_FFMPEG_EXE over PATH) must land
# on the apt ffmpeg — pinning the env explicitly also fails loudly if the base
# ever stops shipping it at /usr/bin/ffmpeg.
ENV IMAGEIO_FFMPEG_EXE=/usr/bin/ffmpeg
# Marks the shipped image for tests/dependencies/test_no_software_video_codecs.py,
# which asserts the strict no-software-codec posture and is skipped for this
# variant (see its module docstring).
ENV DYNAMO_FULL_MEDIA_STACK=1

# Positive capability guard: the strict variant above asserts the ABSENCE of
# H.264/AAC; this variant asserts their PRESENCE. MiniMax-H3's output encoder
# hard-codes libx264 + aac and its pipeline __init__ requires ffprobe on PATH,
# so a base bump that ships a crippled ffmpeg fails HERE, not at the first
# /v1/videos request. libvpx_vp9 is asserted too — dynamo's own t2v encode
# path encodes VP9 with this binary.
RUN set -eu; \
    ff="${IMAGEIO_FFMPEG_EXE}"; \
    test -x "${ff}" || { echo "ERROR: ${ff} missing" >&2; exit 1; }; \
    test -x /usr/bin/ffprobe || { echo "ERROR: /usr/bin/ffprobe missing" >&2; exit 1; }; \
    enc="$("${ff}" -hide_banner -encoders 2>/dev/null)"; \
    echo "${enc}" | grep -qw libx264 || { echo "ERROR: apt ffmpeg has no libx264 encoder" >&2; exit 1; }; \
    echo "${enc}" | grep -qw aac || { echo "ERROR: apt ffmpeg has no aac encoder" >&2; exit 1; }; \
    echo "${enc}" | grep -qiE 'libvpx[-_]vp9' || { echo "ERROR: apt ffmpeg has no libvpx_vp9 encoder" >&2; exit 1; }
{% else %}
ENV IMAGEIO_FFMPEG_EXE=
{% endif %}

{% if device != "xpu" and target not in ("dev", "local-dev") %}
# Add generic UCX aliases beside NIXL's private libraries so native consumers
# use the same implementation without breaking UCX's $ORIGIN-based core and
# module lookup. The wheel's auditwheel dependency directory is deliberately
# placed first for every process; it contains only hash-mangled dependencies
# plus the two generic UCX aliases. No existing wheel file or ELF metadata is
# modified. The same script publishes NIXL's C API directory at the second path
# below and registers it with the runtime linker, so the bare dlopen of
# libnixl_capi.so in nixl-sys resolves. Both paths are passed explicitly because
# the ENV on the next line has to name the same two directories.
RUN --mount=type=bind,source=./container/deps/sglang/install_nixl_ucx_compat.sh,target=/tmp/install_nixl_ucx_compat.sh,readonly \
    --mount=type=bind,source=./container/deps/sglang/discover_nixl_ucx_layout.py,target=/tmp/discover_nixl_ucx_layout.py,readonly \
    bash /tmp/install_nixl_ucx_compat.sh /opt/dynamo/nixl-ucx-compat /opt/dynamo/nixl-capi
ENV LD_LIBRARY_PATH=/opt/dynamo/nixl-ucx-compat:/opt/dynamo/nixl-capi${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
{% endif %}

# Copy tests, deploy and components for CI with correct ownership
COPY --chmod=775 --chown=dynamo:0 tests /workspace/tests
COPY --chmod=775 --chown=dynamo:0 examples /workspace/examples
COPY --chmod=775 --chown=dynamo:0 deploy /workspace/deploy
COPY --chmod=775 --chown=dynamo:0 dev /workspace/dev
COPY --chmod=775 --chown=dynamo:0 components/src/dynamo/common /workspace/components/src/dynamo/common
COPY --chmod=775 --chown=dynamo:0 components/src/dynamo/frontend /workspace/components/src/dynamo/frontend
COPY --chmod=775 --chown=dynamo:0 components/src/dynamo/sglang /workspace/components/src/dynamo/sglang
COPY --chmod=775 --chown=dynamo:0 components/src/dynamo/mocker /workspace/components/src/dynamo/mocker
COPY --chmod=775 --chown=dynamo:0 recipes/ /workspace/recipes/
# The vllm and trtllm images already copy lib/; sglang did not, so its tests had
# no lib/llm/tests/data/media fixtures and any test needing a local media clip
# could only skip. ~32 MB, negligible here and it keeps the three consistent.
COPY --chown=dynamo:0 lib /workspace/lib
COPY --chmod=664 --chown=dynamo:0 LICENSE /workspace/

# Enable forceful shutdown of inflight requests
ENV SGLANG_FORCE_SHUTDOWN=1

# Setup launch banner in common directory accessible to all users
RUN --mount=type=bind,source=./container/launch_message/runtime.txt,target=/opt/dynamo/launch_message.txt \
    sed '/^#\s/d' /opt/dynamo/launch_message.txt > /opt/dynamo/.launch_screen

RUN chmod 755 /opt/dynamo/.launch_screen && \
    echo 'cat /opt/dynamo/.launch_screen' >> /etc/bash.bashrc && \
{%- if device == "xpu" %}
    echo '. /opt/miniforge3/bin/activate sglang' >> /etc/bash.bashrc && \
    echo 'source /opt/intel/oneapi/setvars.sh --force' >> /etc/bash.bashrc && \
    mkdir -p /sgl-workspace && \
    ln -sf /workspace /sgl-workspace/dynamo
{%- else %}
    ln -s /workspace /sgl-workspace/dynamo && \
    NSYS_BIN=$(find /opt/nvidia/nsight-compute -maxdepth 6 -type f -name nsys -executable 2>/dev/null | head -n1) && \
    if [ -n "$NSYS_BIN" ]; then ln -sf "$NSYS_BIN" /usr/local/bin/nsys; \
    else echo "WARNING: no bundled nsys found under /opt/nvidia/nsight-compute"; fi
{% endif %}

{%- if device != "xpu" %}
# Precompile Python bytecode into the image while still root. CI runs tests as
# the non-root `dynamo` user, which cannot write .pyc back to site-packages, and
# the test harness forks a fresh process per test. Without baked .pyc, every test
# process recompiles torch/transformers/sglang from source on first import (~+3.5s
# each), which previously added ~8-10 min to the sglang CI job. This was implicitly
# provided by the now-removed vendored-patch step that ran `import sglang` at build.
RUN SITE_PACKAGES="$(python3 -c 'import site; print(site.getsitepackages()[0])')" && \
    python3 -m compileall -q -j0 "$SITE_PACKAGES" && \
    (python3 -m compileall -q -j0 /sgl-workspace/sglang/python || true)
{%- endif %}

# Belt-and-suspenders guard at the end of the populated runtime stage: fail the
# build if an H.264/H.265/AAC codec *implementation* library appears. The VP9
# ffmpeg (ffmpeg + libav*/libsw* + libvpx) copied above is intentionally
# shipped for the video-encode path, so it is NOT flagged here — the in-tree
# build is VP9-only by construction (wheel_builder's post-build codec-surface
# guard) and the media-codec scan below re-permits it only under /usr/local while
# still catching any stray third-party libav*/ffmpeg. This adds the extra
# non-FFmpeg AAC/H.264 implementation names the scan's deny_globs don't list.
#
# CUDA only, for the same reason as the purge above: this asserts that the purge
# worked, and on XPU there is no purge to assert. Left ungated it would fail that
# build on the base image's own libraries -- which is precisely what happened to
# the first version of this change, where gating the purge alone turned a working
# XPU build into a failing one.
{% if device == "cuda" and not full_media %}
RUN set -eux; \
    remaining="$(find /usr /opt /workspace /sgl-workspace -xdev \
        \( -type f -o -type l \) \
        \( -name 'libx264*.so*' -o -name 'libx265*.so*' \
        -o -name 'libopenh264*.so*' -o -name 'libfdk-aac*.so*' \
        -o -name 'libfaac*.so*' -o -name 'libvo-aacenc*.so*' \
        -o -name 'libaacplus*.so*' \) -print)"; \
    if [ -n "${remaining}" ]; then \
        echo "ERROR: H.264/H.265/AAC codec libraries remain in the SGLang image:" >&2; \
        echo "${remaining}" >&2; \
        exit 1; \
    fi; \
    python3 -c 'import soundfile, torchaudio; from PIL import Image'
{% elif full_media %}
# Full-media variant: the strict find-scan above is skipped (libx264 ships by
# design via apt); the media-codec scan in the licenses stage runs against
# policy/codec_policy_full_media.yaml, which re-permits exactly the apt media
# paths while still failing on codec libraries from anywhere else (e.g. a
# wheel-bundled libav*).
RUN python3 -c 'import soundfile, torchaudio; from PIL import Image'
{% endif %}

# Keep the published runtime honest: Dynamo's SGLang entry points import these
# modules before serving traffic, so fail the image build instead of crashing
# every Frontend and worker after deployment.
{% if device == "cuda" %}
{% if target not in ("dev", "local-dev") %}
RUN python3 -c "import dynamo.frontend.sglang_processor; import dynamo.sglang.args; import sglang"
{% if device == "cuda" and cuda_version == "13.0" %}
# Verify the inherited wheel still loads after runtime dependency cleanup.
RUN --mount=type=bind,source=container/deps/check_mooncake_native_tracing.py,target=/tmp/check_mooncake_native_tracing.py \
    python3 /tmp/check_mooncake_native_tracing.py
{% endif %}
{% endif %}
{% endif %}

USER dynamo
ARG DYNAMO_COMMIT_SHA
ENV DYNAMO_COMMIT_SHA=${DYNAMO_COMMIT_SHA}

{% if device == "xpu" %}
CMD ["bash", "-c", "source /etc/bash.bashrc && exec bash"]
{% else %}
ENTRYPOINT ["/opt/nvidia/nvidia_entrypoint.sh"]
CMD []
{% endif %}


{% if device != "xpu" %}
{# Compliance is skipped for dev/local-dev: those images are not shipped (release
   ships runtime/frontend/operator/planner/snapshot-agent), compliance-extract
   already skips them, and their pre_runtime carries no dynamo venv to scan. #}
{% if target not in ("dev", "local-dev") %}
{% include "templates/compliance.Dockerfile" %}
{% endif %}


#######################################
########## Final runtime image ########
#######################################

FROM pre_runtime AS runtime
# NVDEC (PyNvVideoCodec, added in requirements.sglang.txt) needs the driver
# "video" capability at runtime so libnvcuvid is exposed; without it
# PyNvVideoCodec cannot import. Ensure the K8s pod/runtimeClass does not drop it.
ENV NVIDIA_DRIVER_CAPABILITIES=video,compute,utility
{% if target not in ("dev", "local-dev") %}
COPY --from=licenses /legal /legal
{% endif %}
ENTRYPOINT ["/opt/nvidia/nvidia_entrypoint.sh"]
CMD []
{% endif %}
