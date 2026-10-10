# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the s3 presigned-URL helper in dynamo.common.storage."""

from urllib.parse import parse_qs, urlparse

import pytest

from dynamo.common.storage import PRESIGNED_URL_EXPIRES_S, presign_s3_get_url

pytestmark = [
    pytest.mark.unit,
    pytest.mark.core,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]

FAKE_ENV = {
    "FSSPEC_S3_KEY": "AKIAFAKEFAKEFAKE",
    "FSSPEC_S3_SECRET": "fakesecretfakesecret",
    "FSSPEC_S3_ENDPOINT_URL": "https://oss-cn-shenzhen.aliyuncs.com",
}


def test_presign_default_expiry_is_three_days():
    assert PRESIGNED_URL_EXPIRES_S == 3 * 24 * 60 * 60


def test_presign_oss_url_shape(monkeypatch):
    pytest.importorskip("botocore")
    for name, value in FAKE_ENV.items():
        monkeypatch.setenv(name, value)
    monkeypatch.delenv("FSSPEC_S3_REGION_NAME", raising=False)

    url = presign_s3_get_url("s3://bairui-shenzhen-1/llm-dynamo/minimaxh3", "req-1.mp4")

    assert url is not None
    parsed = urlparse(url)
    query = {k: v[0] for k, v in parse_qs(parsed.query).items()}
    # Virtual-host addressing on the regional endpoint, object key intact.
    assert parsed.scheme == "https"
    assert parsed.netloc == "bairui-shenzhen-1.oss-cn-shenzhen.aliyuncs.com"
    assert parsed.path == "/llm-dynamo/minimaxh3/req-1.mp4"
    # SigV4 query-presign shape with the OSS-required signing region.
    assert query["X-Amz-Algorithm"] == "AWS4-HMAC-SHA256"
    assert query["X-Amz-Expires"] == str(PRESIGNED_URL_EXPIRES_S)
    assert "oss-cn-shenzhen/s3/aws4_request" in query["X-Amz-Credential"]
    assert len(query["X-Amz-Signature"]) == 64


def test_presign_bucket_without_prefix(monkeypatch):
    pytest.importorskip("botocore")
    for name, value in FAKE_ENV.items():
        monkeypatch.setenv(name, value)

    url = presign_s3_get_url("s3://my-bucket", "req-2.mp4")

    parsed = urlparse(url)
    assert parsed.path == "/req-2.mp4"


def test_presign_region_env_override(monkeypatch):
    pytest.importorskip("botocore")
    for name, value in FAKE_ENV.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("FSSPEC_S3_REGION_NAME", "oss-cn-beijing")

    url = presign_s3_get_url("s3://my-bucket", "x.mp4")

    query = {k: v[0] for k, v in parse_qs(urlparse(url).query).items()}
    assert "oss-cn-beijing/s3/aws4_request" in query["X-Amz-Credential"]


def test_presign_falls_back_to_none(monkeypatch):
    # Non-s3 URL: not applicable.
    assert presign_s3_get_url("file:///tmp/media", "a.mp4") is None
    assert presign_s3_get_url("gs://bucket/media", "a.mp4") is None
    # s3 URL without the FSSPEC_S3_* env contract: no credentials to sign with.
    for name in FAKE_ENV:
        monkeypatch.delenv(name, raising=False)
    assert presign_s3_get_url("s3://bucket/media", "a.mp4") is None
    # Endpoint present but credentials missing.
    monkeypatch.setenv("FSSPEC_S3_ENDPOINT_URL", FAKE_ENV["FSSPEC_S3_ENDPOINT_URL"])
    assert presign_s3_get_url("s3://bucket/media", "a.mp4") is None
