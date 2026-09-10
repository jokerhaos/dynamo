# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib
import shlex
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from scripts.render_dockerfile_cn import rewrite

pytestmark = [pytest.mark.pre_merge, pytest.mark.gpu_0, pytest.mark.unit]


@pytest.fixture
def mirror_env(monkeypatch):
    for name, value in {
        "GH_PROXY": "https://mirror.example/",
        "RS_PROXY": "https://rust.example",
        "APT_MIRROR": "apt.example",
        "DNF_MIRROR": "dnf.example",
        "PYTORCH_MIRROR": "https://torch.example/",
        "BUILD_PROXY": "",
    }.items():
        monkeypatch.setenv(name, value)


@pytest.fixture
def wrap_command(mirror_env):
    template = (
        Path(__file__).resolve().parents[1]
        / "container/templates/wheel_builder.Dockerfile"
    ).read_text()
    rendered = rewrite(template)
    commands = [
        line.strip().removesuffix(" && \\")
        for line in rendered.splitlines()
        if "python3 -c" in line and "subprojects/taskflow.wrap" in line
    ]
    assert len(commands) == 1
    assert rendered.index(commands[0]) < rendered.index("./contrib/tomlutil.py")
    return shlex.split(commands[0])


@pytest.mark.timeout(10)
def test_nixl_wrap_uses_proxy_without_changing_hash(wrap_command, tmp_path):
    wrap = tmp_path / "subprojects/taskflow.wrap"
    wrap.parent.mkdir()
    source = (
        "[wrap-file]\n"
        "source_url = https://github.com/taskflow/taskflow/archive/refs/tags/v3.10.0.tar.gz\n"
        "source_hash = fe86765da417f6ceaa2d232ffac70c9afaeb3dc0816337d39a7c93e39c2dee0b\n"
        "patch_directory = taskflow\n"
    )
    wrap.write_text(source)
    subprocess.run(wrap_command, cwd=tmp_path, check=True, timeout=5)
    expected = source.replace(
        "source_url = https://github.com/",
        "source_url = https://mirror.example/https://github.com/",
    )
    assert wrap.read_text() == expected
    # Re-executing the build-time step must not add the proxy twice.
    subprocess.run(wrap_command, cwd=tmp_path, check=True, timeout=5)
    assert wrap.read_text() == expected


@pytest.mark.timeout(10)
def test_nixl_versions_without_taskflow_wrap_are_unchanged(wrap_command, tmp_path):
    subprocess.run(wrap_command, cwd=tmp_path, check=True, timeout=5)
    assert list(tmp_path.iterdir()) == []


def test_other_build_steps_do_not_get_wrap_rewrite(mirror_env):
    rendered = rewrite("RUN git checkout -q FETCH_HEAD && ninja\n")
    assert "taskflow.wrap" not in rendered


@pytest.fixture
def dockerfile_renderer():
    # The lightweight marker-collection environment does not install Jinja2.
    pytest.importorskip(
        "jinja2",
        reason="Full Dockerfile rendering requires Jinja2",
        exc_type=ImportError,
    )
    return importlib.import_module("container.render")


@pytest.mark.parametrize(
    "framework,target,device,cuda_version,expected_count",
    [
        ("sglang", "runtime", "cuda", "13.0", 0),
        ("sglang", "dev", "cuda", "13.0", 1),
        ("sglang", "runtime", "xpu", "", 1),
        ("vllm", "runtime", "cuda", "13.0", 1),
        ("trtllm", "runtime", "cuda", "13.1", 1),
        ("dynamo", "frontend", "cuda", "13.0", 0),
    ],
)
def test_full_dockerfile_render(
    mirror_env,
    dockerfile_renderer,
    framework,
    target,
    device,
    cuda_version,
    expected_count,
):
    container_dir = Path(__file__).resolve().parents[1] / "container"
    context = yaml.safe_load((container_dir / "context.yaml").read_text())
    args = SimpleNamespace(
        framework=framework,
        target=target,
        device=device,
        cuda_version=cuda_version,
        platform="amd64",
        make_efa=False,
    )
    template = dockerfile_renderer._make_jinja_env(container_dir).get_template(
        "Dockerfile.template"
    )
    original = template.render(
        context=context, **dockerfile_renderer._render_context(args, context)
    )
    rendered = rewrite(original)
    assert rendered.count("subprojects/taskflow.wrap") == expected_count
    assert rendered.count("./contrib/tomlutil.py") == expected_count
