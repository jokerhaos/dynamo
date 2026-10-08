# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Dynamo Common Storage Module

Filesystem Spec (fsspec) is a project to provide a unified pythonic interface to local, remote and embedded file systems and bytes storage.

https://filesystem-spec.readthedocs.io/en/latest/index.html#who-uses-fsspec

Configuration for the storage:

Local Filesystem:
    1. fs_url MUST contain a root path - path must be accessible and writable

S3:
    1. If you want to use S3 please install additional dependencies: fsspec[s3]
    2. fs_url MUST contain a bucket name
    3. Configure credentials     https://s3fs.readthedocs.io/en/latest/?badge=latest#credentials

        a) AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, and AWS_SESSION_TOKEN environment variables

        b) configuration files such as ~/.aws/credentials

        c) for nodes on EC2, the IAM metadata provider

        d) for S3 compatible storage, you can use the following environment variables:
            # export FSSPEC_S3_ENDPOINT_URL=https://...
            # export FSSPEC_S3_KEY='miniokey...'
            # export FSSPEC_S3_SECRET='asecretkey...'



"""

import asyncio
import os
from typing import Optional
from urllib.parse import urlparse

import fsspec
from fsspec.implementations.dirfs import DirFileSystem

# Presigned GET URLs are the delivery contract for private-object media
# stores: S3-style SigV4 presigning tops out at exactly 7 days, which is the
# default here.
PRESIGNED_URL_EXPIRES_S = 7 * 24 * 60 * 60


def presign_s3_get_url(
    fs_url: str,
    storage_path: str,
    expires_s: int = PRESIGNED_URL_EXPIRES_S,
) -> Optional[str]:
    """Presign a GET for an s3:// media object, or None when not applicable.

    Reads the same ``FSSPEC_S3_*`` environment contract ``get_fs()`` exposes
    (``FSSPEC_S3_KEY`` / ``FSSPEC_S3_SECRET`` credentials,
    ``FSSPEC_S3_ENDPOINT_URL`` custom endpoint, optional
    ``FSSPEC_S3_REGION_NAME``). The signing region falls back to the first
    label of the endpoint host: OSS regional endpoints
    (``oss-cn-shenzhen.aliyuncs.com``) must sign with ``oss-cn-shenzhen`` —
    OSS rejects any other region in the SigV4 credential scope.

    Signing is a local HMAC computation (botocore, present wherever s3fs is)
    — no network, no service call. Returns None for non-s3 URLs or when the
    credentials are absent; callers fall back to the plain media URL.
    """
    if not fs_url.startswith("s3://"):
        return None
    key = os.environ.get("FSSPEC_S3_KEY")
    secret = os.environ.get("FSSPEC_S3_SECRET")
    endpoint = os.environ.get("FSSPEC_S3_ENDPOINT_URL")
    if not key or not secret or not endpoint:
        return None

    # fs_url is "s3://bucket/prefix" (get_fs's root_path is everything after
    # the separator); the object key is that prefix joined with storage_path.
    root = fs_url[len("s3://") :].strip("/")
    bucket, _, prefix = root.partition("/")
    object_key = f"{prefix}/{storage_path}" if prefix else storage_path

    region = os.environ.get("FSSPEC_S3_REGION_NAME")
    if not region:
        host = urlparse(endpoint).hostname or ""
        region = "us-east-1" if host.split(".")[0] == "s3" else host.split(".")[0]

    # Local import: botocore rides with s3fs, but this module must stay
    # importable where it is absent (e.g. unit-test environments).
    import botocore.session
    from botocore.config import Config

    session = botocore.session.get_session()
    client = session.create_client(
        "s3",
        aws_access_key_id=key,
        aws_secret_access_key=secret,
        endpoint_url=endpoint,
        region_name=region,
        config=Config(signature_version="s3v4", s3={"addressing_style": "virtual"}),
    )
    return client.generate_presigned_url(
        "get_object",
        Params={"Bucket": bucket, "Key": object_key},
        ExpiresIn=expires_s,
    )


def get_fs(fs_url: str) -> DirFileSystem:
    """
    Initialize fsspec filesystem for the given URL.

    Args:
        fs_url: The URL of the filesystem to initialize. e.g. s3://bucket, gs://bucket, file:///local/path

    Returns:
        The initialized DirFileSystem wrapper for the filesystem.

        fs.fs.protocol to get the protocol of the filesystem
        fs.path to get the bucket or root path

        path to the object in the filesystem - f"{fs.fs.protocol}://{fs.path}/{path}"
    """
    # Extract protocol from URL (s3://, gs://, az://, file://)
    fs_url_parts = fs_url.split("://")
    protocol = fs_url_parts[0] if "://" in fs_url else "file"

    # ... or bucket name
    root_path = fs_url_parts[1] if len(fs_url_parts) > 1 else "/"

    fs_opts = {}
    if protocol in "file":
        # create directory for local filesystem
        fs_opts = {"auto_mkdir": True}

    return DirFileSystem(fs=fsspec.filesystem(protocol, **fs_opts), path=root_path)


def get_media_url(
    fs: DirFileSystem, storage_path: str, base_url: Optional[str] = None
) -> str:
    """Build a public URL for a file stored in the media filesystem.

    Args:
        fs: The DirFileSystem returned by ``get_fs()``.
        storage_path: Relative path within the filesystem (e.g. "videos/req-id.mp4").
        base_url: Optional CDN / proxy base URL.  When set, the returned URL is
            ``{base_url}/{storage_path}``.  When *None*, the URL is constructed
            from the filesystem's protocol and root path.

    Returns:
        Public URL string for the uploaded file.

    Examples:
        >>> fs = get_fs("memory://media")
        >>> get_media_url(fs, "videos/request.mp4", "https://cdn.example.com/media")
        'https://cdn.example.com/media/videos/request.mp4'
        >>> get_media_url(fs, "videos/request.mp4")
        'memory:///media/videos/request.mp4'
    """
    if base_url:
        return f"{base_url.rstrip('/')}/{storage_path}"

    protocol = fs.fs.protocol
    if isinstance(protocol, (list, tuple)):
        protocol = protocol[0]
    return f"{protocol}://{fs.path}/{storage_path}"


async def upload_to_fs(
    fs: DirFileSystem,
    storage_path: str,
    data: bytes,
    base_url: Optional[str] = None,
) -> str:
    """Upload bytes to the media filesystem and return the public URL.

    This is the canonical helper for all backends (vLLM, SGLang, TRT-LLM)
    to store generated images/videos and produce a response URL.

    Args:
        fs: The DirFileSystem returned by ``get_fs()``.
        storage_path: Relative path within the filesystem (e.g. "images/req-id/file.png").
        data: Raw bytes to upload.
        base_url: Optional CDN / proxy base URL for URL rewriting.

    Returns:
        Public URL string for the uploaded file.
    """
    await asyncio.to_thread(fs.pipe, storage_path, data)
    return get_media_url(fs, storage_path, base_url)
