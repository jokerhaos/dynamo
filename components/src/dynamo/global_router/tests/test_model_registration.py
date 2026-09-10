#  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0

"""Tests for GlobalRouter model registration."""

import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from dynamo.global_router import __main__ as global_router_main
from dynamo.llm import WorkerType

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.parallel,
    pytest.mark.unit,
]


class FakeEndpoint:
    """Endpoint stub that completes serving immediately."""

    async def serve_endpoint(self, handler, **kwargs):
        """Return without starting a network service."""
        return None


class FakeRuntime:
    """Runtime stub that creates in-process endpoints."""

    def endpoint(self, endpoint_name):
        """Return a fake endpoint for any endpoint name."""
        return FakeEndpoint()


@pytest.fixture(
    params=[
        (None, None, None, None),
        ("glm47", "glm45", None, None),
        (None, None, "glm47", "glm45"),
        ("hermes", "qwen3", "glm47", "glm45"),
        ("glm47", None, None, None),
        (None, "glm45", None, None),
    ],
    ids=["defaults", "cli", "env", "cli-over-env", "tool-only", "reasoning-only"],
)
def registration_config(request, monkeypatch):
    """Parse real CLI/env configuration for each registration path."""
    cli_tool, cli_reasoning, env_tool, env_reasoning = request.param
    for name, value in (
        ("DYN_TOOL_CALL_PARSER", env_tool),
        ("DYN_REASONING_PARSER", env_reasoning),
    ):
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)

    argv = [
        "global-router",
        "--config",
        "test-config.json",
        "--model-name",
        "test-model",
        "--namespace",
        "test-namespace",
        "--component-name",
        "global-router",
    ]
    if cli_tool is not None:
        argv.extend(["--dyn-tool-call-parser", cli_tool])
    if cli_reasoning is not None:
        argv.extend(["--dyn-reasoning-parser", cli_reasoning])
    monkeypatch.setattr(sys, "argv", argv)
    return (
        global_router_main.parse_args(),
        cli_tool if cli_tool is not None else env_tool,
        cli_reasoning if cli_reasoning is not None else env_reasoning,
    )


@pytest.mark.asyncio
async def test_disagg_registration_skips_model_weights(
    monkeypatch, registration_config
):
    """Disaggregated registration forwards parsers without model weights."""
    register_model = AsyncMock()
    monkeypatch.setattr(global_router_main, "register_model", register_model)
    config, tool_call_parser, reasoning_parser = registration_config
    handler = SimpleNamespace(
        handle_prefill=AsyncMock(),
        handle_decode=AsyncMock(),
    )

    await global_router_main._serve_disagg(FakeRuntime(), config, handler)

    assert register_model.await_count == 2
    prefill_call, decode_call = register_model.await_args_list
    assert prefill_call.kwargs["worker_type"] == WorkerType.Prefill
    assert prefill_call.kwargs["ignore_weights"] is True
    assert prefill_call.kwargs["model_path"] == "test-model"
    assert prefill_call.kwargs["model_name"] == "test-model"
    assert decode_call.kwargs["worker_type"] == WorkerType.Decode
    assert decode_call.kwargs["ignore_weights"] is True
    assert decode_call.kwargs["model_path"] == "test-model"
    assert decode_call.kwargs["model_name"] == "test-model"
    for call in (prefill_call, decode_call):
        runtime_config = call.kwargs["runtime_config"]
        assert runtime_config.tool_call_parser == tool_call_parser
        assert runtime_config.reasoning_parser == reasoning_parser


@pytest.mark.asyncio
async def test_agg_registration_skips_model_weights(monkeypatch, registration_config):
    """Aggregated registration forwards parsers without model weights."""
    register_model = AsyncMock()
    monkeypatch.setattr(global_router_main, "register_model", register_model)
    config, tool_call_parser, reasoning_parser = registration_config
    handler = SimpleNamespace(handle_generate=AsyncMock())

    await global_router_main._serve_agg(FakeRuntime(), config, handler)

    register_model.assert_awaited_once()
    assert register_model.await_args.kwargs["worker_type"] == WorkerType.Aggregated
    assert register_model.await_args.kwargs["ignore_weights"] is True
    assert register_model.await_args.kwargs["model_path"] == "test-model"
    assert register_model.await_args.kwargs["model_name"] == "test-model"
    runtime_config = register_model.await_args.kwargs["runtime_config"]
    assert runtime_config.tool_call_parser == tool_call_parser
    assert runtime_config.reasoning_parser == reasoning_parser
