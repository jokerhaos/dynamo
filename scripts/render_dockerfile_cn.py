#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Rewrite a rendered Dynamo Dockerfile for the domestic (China) build environment.

The upstream Dockerfiles (container/templates/*.Dockerfile) hardcode github.com,
pypi.org, static.rust-lang.org, download.pytorch.org and nvcr.io / quay.io /
ghcr.io — none reachable from the self-hosted runner's buildkit without routing
them through domestic mirrors. Keeping all of that rewriting here lets the
workflow stay a thin layer of standard GitHub actions.

Reads the rendered Dockerfile (default container/rendered.Dockerfile), applies
the mirror rewrites in place, and prints a report of remaining foreign-looking
hosts.

Env vars consumed (all required unless noted):
  GH_PROXY, RS_PROXY, APT_MIRROR, DNF_MIRROR, PYTORCH_MIRROR,
  BASE_IMAGE_REF, WHEEL_BUILDER_IMAGE_REF, UV_IMAGE_REF,
  BUILD_PROXY (optional, "" = direct egress),
  RENDER_SCCACHE_MINIO (optional, "1" = retarget the in-image sccache
    compile cache from AWS web-identity auth to the internal MinIO:
    replaces the aws-web-identity-token/aws-role-arn secret mounts with
    static S3 access keys and plumbs SCCACHE_ENDPOINT through. Kept
    conditional so an unconfigured repo renders a byte-identical
    Dockerfile and its layer cache survives).
"""

import os
import re
import shlex
import sys

# hosts we redirect, or that are known-internal / expected-foreign, so the
# post-rewrite report does not flag them as noise.
_OK_HOSTS = (
    "rsproxy.cn",
    "mirrors.tuna",
    "nvcr.io",
    "quay.io",
    "ghcr.io",
    "ai-dynamo/nixl/blob",
    "gh-proxy.com",
)


def _insert_after(text, pattern, lines):
    out, done = [], False
    for ln in text.splitlines():
        out.append(ln)
        if not done and re.match(pattern, ln):
            out.extend(lines)
            done = True
    return "\n".join(out), done


def _wrap_clone(m):
    flags = m.group(1).strip()
    url = m.group(2)
    explicit = (m.group(3) or "").strip()
    if explicit:
        d = explicit
        tgt = f"git clone {flags} {url} {explicit}"
    else:
        d = url.rstrip("/").split("/")[-1]
        d = d[:-4] if d.endswith(".git") else d
        tgt = f"git clone {flags} {url}"
    return (
        f"{{ _n=0; until {tgt}; do _n=$((_n+1)); [ $_n -ge 8 ] && exit 1; "
        f"rm -rf {d} 2>/dev/null; echo git_clone_retry_$_n; "
        f"sleep $((_n*4)); done; }}"
    )


def _wrap_fetch(m):
    ref = m.group(1)
    return (
        f"{{ _n=0; until git fetch --depth 1 origin {ref}; "
        f"do _n=$((_n+1)); [ $_n -ge 8 ] && exit 1; "
        f"echo git_fetch_retry_$_n; sleep $((_n*4)); done; }}"
    )


def _sccache_minio(s):
    """Retarget the sccache RUN steps from AWS web-identity to static MinIO keys.

    The upstream templates authenticate sccache's S3 backend with AWS IRSA
    (web-identity token file + role ARN secret) and only know SCCACHE_BUCKET /
    SCCACHE_REGION. The internal MinIO is plain HTTP on a cluster service and
    uses static access keys, so swap the two aws-* secret mounts for an
    AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY pair, drop the token-file export,
    and declare SCCACHE_ENDPOINT alongside the other two SCCACHE args.
    """
    # The mounts: replace the AWS web-identity pair with static S3 keys. The
    # pair leads the RUN in the ffmpeg/UCX/libfabric steps; in the maturin
    # step it follows the pypi-netrc mount — matching just the two mount
    # lines (no RUN anchor) covers both.
    s = s.replace(
        "--mount=type=secret,id=aws-web-identity-token,target=/run/secrets/aws-token \\\n"
        "    --mount=type=secret,id=aws-role-arn,env=AWS_ROLE_ARN \\\n",
        "--mount=type=secret,id=s3-access-key,env=AWS_ACCESS_KEY_ID \\\n"
        "    --mount=type=secret,id=s3-secret-key,env=AWS_SECRET_ACCESS_KEY \\\n",
    )
    # The token-file export: drop it wherever it sits in the preamble (it
    # directly follows the mounts in most steps, after the NETRC export in
    # the maturin one). With static keys the AWS SDK reads the env mounts.
    s = s.replace(
        "    export AWS_WEB_IDENTITY_TOKEN_FILE=/run/secrets/aws-token && \\\n",
        "",
    )
    s = s.replace(
        'ARG SCCACHE_REGION=""\n',
        'ARG SCCACHE_REGION=""\nARG SCCACHE_ENDPOINT=""\n',
    )
    s = s.replace(
        "ARG SCCACHE_REGION\n",
        "ARG SCCACHE_REGION\nARG SCCACHE_ENDPOINT\n",
    )
    s = s.replace(
        "ENV SCCACHE_BUCKET=${USE_SCCACHE:+${SCCACHE_BUCKET}} \\\n"
        "    SCCACHE_REGION=${USE_SCCACHE:+${SCCACHE_REGION}}",
        "ENV SCCACHE_BUCKET=${USE_SCCACHE:+${SCCACHE_BUCKET}} \\\n"
        "    SCCACHE_REGION=${USE_SCCACHE:+${SCCACHE_REGION}} \\\n"
        "    SCCACHE_ENDPOINT=${USE_SCCACHE:+${SCCACHE_ENDPOINT}}",
    )
    return s


def rewrite(s):
    GH = os.environ["GH_PROXY"]
    RS = os.environ["RS_PROXY"]
    APT = os.environ["APT_MIRROR"]
    DNF = os.environ["DNF_MIRROR"]

    # 1) GitHub releases + git clone/remote -> gh-proxy prefix.
    s = s.replace("https://github.com/", f"{GH}/https://github.com/")

    # NIXL fetches Taskflow via Meson's wrap downloader, not git. The wrap
    # only exists after checkout, so Dockerfile URL rewriting cannot reach it.
    # Change only its source URL; keep the upstream source_hash intact.
    wrap_rewrite = (
        "from pathlib import Path; "
        "p = Path('subprojects/taskflow.wrap'); "
        "p.write_text(p.read_text().replace("
        "'source_url = https://github.com/', "
        f"{'source_url = ' + GH.rstrip('/') + '/https://github.com/'!r})) "
        "if p.is_file() else None"
    )
    nixl_wheel_name = "    ./contrib/tomlutil.py --wheel-name $PKG_NAME pyproject.toml"
    s = s.replace(
        nixl_wheel_name,
        f"    python3 -c {shlex.quote(wrap_rewrite)} && \\\n" + nixl_wheel_name,
    )

    # 2) libvpx: googlesource has no China mirror -> GitHub mirror, proxied.
    #    github.com/webm/libvpx is a 404 (served to anon as 401 -> "could not
    #    read Username"); webmproject/libvpx is the real mirror, verified
    #    anon-cloneable at v1.14.1.
    s = s.replace(
        "https://chromium.googlesource.com/webm/libvpx.git",
        f"{GH}/https://github.com/webmproject/libvpx.git",
    )

    # 3) hwloc: open-mpi.org is blocked -> GitHub open-mpi/hwloc release asset.
    #    NB: keep the ${HWLOC_*} build-time tokens; this is plain concatenation
    #    (not an f-string) so {HWLOC_VERSION} is not treated as python output.
    s = s.replace(
        "https://download.open-mpi.org/release/hwloc/v${HWLOC_SERIES}"
        "/hwloc-${HWLOC_VERSION}.tar.gz",
        GH + "/https://github.com/open-mpi/hwloc/releases/download/hwloc-"
        "${HWLOC_VERSION}/hwloc-${HWLOC_VERSION}.tar.gz",
    )

    # 1b) gh-proxy anon github fetches hit github's per-IP rate limit from the
    #     shared IP (transient 403/429), so bump the github wgets to retry on
    #     those instead of failing. Release-asset wgets keep their token; only
    #     github releases are touched here.
    s = s.replace(
        "--tries=3 --waitretry=5",
        "--tries=8 --waitretry=8 --retry-on-http-error=403,429",
    )
    s = s.replace(
        'wget -q "https://gh-proxy.com/',
        'wget -q --tries=8 --waitretry=8 --retry-on-http-error=403,429 "https://gh-proxy.com/',
    )

    # 4) rustup binary: static.rust-lang.org/rustup -> rsproxy.cn/rustup.
    s = s.replace("https://static.rust-lang.org/rustup", f"{RS}/rustup")

    # 4b) PyTorch wheel index (dynamo framework installs torch via
    #     --extra-index-url https://download.pytorch.org/whl/cuXXX).
    s = s.replace("https://download.pytorch.org/whl/", os.environ["PYTORCH_MIRROR"])

    # 4c) Base images: pre-mirrored into orc.0xtoken.cn (runner can't reach
    #     nvcr/quay/ghcr). Override the ARG defaults and the uv COPY --from,
    #     and strip the `# syntax=docker/dockerfile:...` line so buildkit uses
    #     its bundled frontend instead of pulling docker.io/docker/dockerfile.
    bir = os.environ.get("BASE_IMAGE_REF", "")
    wbir = os.environ.get("WHEEL_BUILDER_IMAGE_REF", "")
    uir = os.environ.get("UV_IMAGE_REF", "")
    if bir:
        s = s.replace(
            "ARG BASE_IMAGE=nvcr.io/nvidia/cuda-dl-base", f"ARG BASE_IMAGE={bir}"
        )
    if wbir:
        s = s.replace(
            "ARG WHEEL_BUILDER_IMAGE=quay.io/pypa/manylinux_2_28_x86_64",
            f"ARG WHEEL_BUILDER_IMAGE={wbir}",
        )
    if uir:
        s = s.replace("COPY --from=ghcr.io/astral-sh/uv:0.12.0", f"COPY --from={uir}")
    s = re.sub(r"(?m)^#\s*syntax=docker/.*$\n?", "", s, count=1)

    # Optional HTTP(S) proxy the runner can reach. rsproxy's CDN edges are
    # unreachable from the runner's direct egress, so a proxy that can reach
    # rsproxy (e.g. a sibling box with working foreign egress) fixes rustup
    # + cargo. NO_PROXY keeps the domestic mirrors + orc direct. Empty = direct.
    bp = os.environ.get("BUILD_PROXY", "").strip()
    if bp:
        nop = ",".join(
            [
                "localhost",
                "127.0.0.1",
                "orc.0xtoken.cn",
                "zot.orc-registry.svc.cluster.local",
                "mirrors.tuna.tsinghua.edu.cn",
                "pypi.tuna.tsinghua.edu.cn",
                "mirrors.aliyun.com",
            ]
        )
        block = "\n".join(
            [f"ENV HTTPS_PROXY={bp}", f"ENV HTTP_PROXY={bp}", f"ENV NO_PROXY={nop}"]
        )
        s = re.sub(r"(?m)^(FROM .*)$", lambda m: m.group(1) + "\n" + block, s)

    cargo_cfg = (
        "RUN mkdir -p /usr/local/cargo && "
        "printf '%s\\n' "
        "'[source.crates-io]' 'replace-with = \"rsproxy-sparse\"' "
        "'[source.rsproxy-sparse]' 'registry = \"sparse+" + RS + "/index/\"' "
        "'[net]' 'git-fetch-with-cli = true' "
        # utoipa-swagger-ui's build.rs curls github.com directly for the Swagger
        # UI zip (SWAGGER_UI_DOWNLOAD_URL, default = github.com, no gh-proxy),
        # which buildkit can't reach from China. cargo's [env] section exposes
        # the env to build scripts, so route that one download through gh-proxy.
        # NOTE: v5.17.14 matches utoipa-swagger-ui 9.0.2's expected zip; revisit
        # if that Cargo.lock entry is updated.
        "'[env]' 'SWAGGER_UI_DOWNLOAD_URL = \""
        + GH
        + "/https://github.com/swagger-api/swagger-ui/archive/refs/tags/v5.17.14.zip\"' "
        "> /usr/local/cargo/config.toml"
    )

    # 5) dynamo_base: steer rustup to rsproxy; rewrite apt sources to a
    #    domestic ubuntu mirror before any apt-get.
    s, _ = _insert_after(
        s,
        r"^FROM .* AS dynamo_base$",
        [
            f"ENV RUSTUP_DIST_SERVER={RS}",
            f"ENV RUSTUP_UPDATE_ROOT={RS}/rustup",
            cargo_cfg,
            f'RUN sed -i -e "s|//[^ /]*\\.ubuntu\\.com|//{APT}|g" '
            f"/etc/apt/sources.list /etc/apt/sources.list.d/*.list "
            f"/etc/apt/sources.list.d/*.sources 2>/dev/null || true",
        ],
    )

    # 6) wheel_builder_base (manylinux/almalinux): overwrite the stock repo
    #    file with explicit almalinux-8 mirror baseurls (TUNA has no almalinux
    #    mirror, and the stock mirrorlist CGI 404s on domestic mirrors), and
    #    set skip_if_unavailable so foreign repos added later by
    #    almalinux-release-synergy (not mirrored anywhere domestic) are
    #    skipped, not fatal.
    secs = [
        ("baseos", "BaseOS"),
        ("appstream", "AppStream"),
        ("extras", "extras"),
        ("powertools", "PowerTools"),
    ]
    parts = []
    for sid, path in secs:
        parts += [
            f"[{sid}]",
            f"name=AlmaLinux $releasever - {path}",
            f"baseurl=https://{DNF}/almalinux/$releasever/{path}/$basearch/os/",
            "enabled=1",
            "gpgcheck=0",
        ]
    args = " ".join(f"'{p}'" for p in parts)
    s, _ = _insert_after(
        s,
        r"^FROM .* AS wheel_builder_base$",
        [
            # UCX (and any other repo) init git submodules from .gitmodules URLs
            # that point straight at https://github.com/...; submodule clone from
            # a China buildkit hits github.com directly and dies with "Empty reply
            # from server" (transient), failing the build when the header the
            # submodule carries is later #included. A global insteadOf rewrite
            # routes EVERY github clone through gh-proxy, submodule included. The
            # explicit gh-proxy git clones in the Dockerfile are already prefixed
            # so the insteadOf is a no-op on them.
            f'RUN git config --global url."{GH}/https://github.com/".insteadOf "https://github.com/"',
            f"RUN rm -f /etc/yum.repos.d/almalinux*.repo "
            f"&& printf '%s\\n' {args} > /etc/yum.repos.d/almalinux.repo "
            f"&& (grep -q skip_if_unavailable /etc/dnf/dnf.conf "
            f"|| printf '\\nskip_if_unavailable=True\\n' >> /etc/dnf/dnf.conf) || true",
        ],
    )

    # Retry transient gh-proxy 403s on the github git clones/fetches. The
    #    clones are anonymous (gh-proxy 403s a job token in buildkit), and anon
    #    hits github's per-IP rate limit — but the 403s are transient, so an
    #    `until … do backoff; done` wrapper on each git clone/fetch rides it out.
    s = re.sub(
        r"git clone (.+?)(https://gh-proxy\.com/https://github\.com/[\w./-]+\.git)(\s+/[\w./-]+)?",
        _wrap_clone,
        s,
    )
    s = re.sub(
        r'git fetch --depth 1 origin ("\$\{[A-Z_]+\}"|"[^"]*"|\$\{[A-Z_]+\})',
        _wrap_fetch,
        s,
    )

    # 8) sccache-on-MinIO (opt-in, set by the workflow only when the full
    #    MinIO config exists): keeps unconfigured builds byte-identical so
    #    their layer cache keys do not shift.
    if os.environ.get("RENDER_SCCACHE_MINIO", "") == "1":
        s = _sccache_minio(s)

    return s


def report(s, gh):
    foreign = set()
    for m in re.findall(r"https?://[a-zA-Z0-9._:-]+", s):
        if m.startswith("https://x-access-token:") or m.startswith(gh):
            continue
        for ok in _OK_HOSTS:
            if ok in m:
                break
        else:
            foreign.add(m)
    print("REMAINING foreign-looking hosts:")
    for h in sorted(foreign):
        print("   ", h)


def main():
    df = sys.argv[1] if len(sys.argv) > 1 else "container/rendered.Dockerfile"
    s = open(df).read()
    s = rewrite(s)
    open(df, "w").write(s)
    print("rewrite complete")
    report(s, os.environ["GH_PROXY"])


if __name__ == "__main__":
    main()
