# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for shared Dynamo runtime arguments."""

import argparse
import logging
import os

import pytest

import dynamo.common.configuration.groups.runtime_args as runtime_args
from dynamo.common.configuration.groups.runtime_args import (
    DynamoRuntimeArgGroup,
    DynamoRuntimeConfig,
)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]


def _parse_runtime_args(argv: list[str]) -> tuple[DynamoRuntimeConfig, str]:
    parser = argparse.ArgumentParser()
    DynamoRuntimeArgGroup().add_arguments(parser)
    args = parser.parse_args(argv)
    config = DynamoRuntimeConfig.from_cli_args(args)
    config.validate()
    return config, parser.format_help()


def test_fpm_trace_defaults_disabled(monkeypatch):
    monkeypatch.delenv("DYN_FPM_TRACE", raising=False)

    config, _ = _parse_runtime_args([])

    assert config.fpm_trace is False
    assert "DYN_FPM_TRACE" not in os.environ


def test_kv_state_endpoint_supports_cli_and_env(monkeypatch):
    monkeypatch.setenv("DYN_KV_STATE_ENDPOINT", "dynamo/kv/events")
    env_config, help_text = _parse_runtime_args([])
    cli_config, _ = _parse_runtime_args(["--kv-state-endpoint", "other/cache/updates"])

    assert env_config.kv_state_endpoint == "dynamo/kv/events"
    assert cli_config.kv_state_endpoint == "other/cache/updates"
    assert "--kv-state-endpoint" in help_text
    assert "DYN_KV_STATE_ENDPOINT" in help_text


def test_response_plane_defaults_to_tcp_and_accepts_quic(monkeypatch):
    monkeypatch.delenv("DYN_RESPONSE_PLANE", raising=False)

    default_config, help_text = _parse_runtime_args([])
    quic_config, _ = _parse_runtime_args(["--response-plane", "quic"])
    monkeypatch.setenv("DYN_RESPONSE_PLANE", "quic")
    env_config, _ = _parse_runtime_args([])

    assert default_config.response_plane == "tcp"
    assert quic_config.response_plane == "quic"
    assert env_config.response_plane == "quic"
    assert os.environ["DYN_RESPONSE_PLANE"] == "quic"
    assert "--response-plane" in help_text


def test_response_plane_rejects_invalid_value():
    with pytest.raises(SystemExit):
        _parse_runtime_args(["--response-plane", "invalid"])


def test_fpm_trace_env_enables_and_is_canonicalized(monkeypatch):
    monkeypatch.setenv("DYN_FPM_TRACE", "on")

    config, _ = _parse_runtime_args([])

    assert config.fpm_trace is True
    assert os.environ["DYN_FPM_TRACE"] == "1"


def test_fpm_trace_env_is_trimmed(monkeypatch):
    monkeypatch.setenv("DYN_FPM_TRACE", " true ")

    config, _ = _parse_runtime_args([])

    assert config.fpm_trace is True
    assert os.environ["DYN_FPM_TRACE"] == "1"


def test_invalid_fpm_trace_warns_once_and_is_disabled(monkeypatch, caplog):
    monkeypatch.setenv("DYN_FPM_TRACE", "sometimes")
    monkeypatch.setattr(runtime_args, "_fpm_trace_invalid_warning_emitted", False)

    with caplog.at_level(logging.WARNING, logger=runtime_args.__name__):
        config, _ = _parse_runtime_args([])
        monkeypatch.setenv("DYN_FPM_TRACE", "still-invalid")
        _parse_runtime_args([])

    assert config.fpm_trace is False
    assert os.environ["DYN_FPM_TRACE"] == "0"
    assert caplog.text.count("Invalid DYN_FPM_TRACE value") == 1


def test_explicit_fpm_port_preserves_precedence_over_invalid_trace(monkeypatch, caplog):
    monkeypatch.setenv("DYN_FORWARDPASS_METRIC_PORT", "23456")
    monkeypatch.setenv("DYN_FPM_TRACE", "sometimes")
    monkeypatch.setattr(runtime_args, "_fpm_trace_invalid_warning_emitted", False)

    with caplog.at_level(logging.WARNING, logger=runtime_args.__name__):
        config, _ = _parse_runtime_args([])

    assert config.fpm_trace is False
    assert os.environ["DYN_FPM_TRACE"] == "0"
    assert "Invalid DYN_FPM_TRACE value" not in caplog.text


def test_fpm_trace_cli_enables_and_is_exported(monkeypatch):
    monkeypatch.delenv("DYN_FPM_TRACE", raising=False)

    config, _ = _parse_runtime_args(["--fpm-trace"])

    assert config.fpm_trace is True
    assert os.environ["DYN_FPM_TRACE"] == "1"


def test_no_fpm_trace_cli_overrides_enabled_env(monkeypatch):
    monkeypatch.setenv("DYN_FPM_TRACE", "true")

    config, _ = _parse_runtime_args(["--no-fpm-trace"])

    assert config.fpm_trace is False
    assert os.environ["DYN_FPM_TRACE"] == "0"


def test_fpm_trace_help_lists_flag_and_env(monkeypatch):
    monkeypatch.delenv("DYN_FPM_TRACE", raising=False)

    _, help_text = _parse_runtime_args([])

    assert "--fpm-trace" in help_text
    assert "--no-fpm-trace" in help_text
    assert "DYN_FPM_TRACE" in help_text


@pytest.mark.parametrize("mode", ["enabled", "disabled"])
def test_default_thinking_mode_cli(mode, monkeypatch):
    monkeypatch.delenv("DYN_DEFAULT_THINKING_MODE", raising=False)

    config, help_text = _parse_runtime_args(["--dyn-default-thinking-mode", mode])

    assert config.dyn_default_thinking_mode == mode
    assert "DYN_DEFAULT_THINKING_MODE" in help_text


def test_media_output_presign_expires_defaults_to_three_days(monkeypatch):
    monkeypatch.delenv("DYN_MEDIA_OUTPUT_PRESIGN_EXPIRES_S", raising=False)

    config, help_text = _parse_runtime_args([])

    assert config.media_output_presign_expires_s == 3 * 24 * 60 * 60
    assert "--media-output-presign-expires-s" in help_text
    assert "DYN_MEDIA_OUTPUT_PRESIGN_EXPIRES_S" in help_text


def test_media_output_presign_expires_supports_cli_and_env(monkeypatch):
    monkeypatch.setenv("DYN_MEDIA_OUTPUT_PRESIGN_EXPIRES_S", "3600")

    env_config, _ = _parse_runtime_args([])
    cli_config, _ = _parse_runtime_args(["--media-output-presign-expires-s", "60"])

    assert env_config.media_output_presign_expires_s == 3600
    assert cli_config.media_output_presign_expires_s == 60


def test_media_output_presign_expires_rejects_nonpositive(monkeypatch):
    monkeypatch.delenv("DYN_MEDIA_OUTPUT_PRESIGN_EXPIRES_S", raising=False)

    for raw in ("0", "-60"):
        with pytest.raises(ValueError):
            _parse_runtime_args(["--media-output-presign-expires-s", raw])


def test_media_output_presign_expires_rejects_over_seven_days(monkeypatch):
    monkeypatch.delenv("DYN_MEDIA_OUTPUT_PRESIGN_EXPIRES_S", raising=False)

    with pytest.raises(ValueError, match="7 days"):
        _parse_runtime_args(["--media-output-presign-expires-s", str(8 * 24 * 60 * 60)])


def test_default_thinking_mode_env(monkeypatch):
    monkeypatch.setenv("DYN_DEFAULT_THINKING_MODE", "disabled")

    config, _ = _parse_runtime_args([])

    assert config.dyn_default_thinking_mode == "disabled"


def test_default_thinking_mode_rejects_invalid_value(monkeypatch):
    monkeypatch.delenv("DYN_DEFAULT_THINKING_MODE", raising=False)

    with pytest.raises(SystemExit):
        _parse_runtime_args(["--dyn-default-thinking-mode", "adaptive"])
