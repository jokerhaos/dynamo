# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise the cached-checkout commands in the internal CI workflows."""

import os
import re
import shutil
import subprocess
import tempfile
import threading
import unittest
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


class CachedCheckoutTest(unittest.TestCase):
    def test_checkout_survives_changing_pack_files_and_cache_removal(self):
        for workflow in ("pre-merge.yml", "internal-build-push.yml"):
            config = yaml.safe_load((ROOT / ".github/workflows" / workflow).read_text())
            for job, definition in config["jobs"].items():
                script = "\n".join(step.get("run", "") for step in definition["steps"])
                match = re.search(
                    r'(?m)^ +(?:git -C "\$MIRROR" config uploadpack.allowFilter true\n +)?'
                    r"git clone --no-local --no-checkout .*?"
                    r'^ +git -C "\$GITHUB_WORKSPACE" checkout --detach "\$GITHUB_SHA"$',
                    script,
                    re.DOTALL,
                )
                if not match:
                    if job != "pre-merge-status-check":
                        self.fail(f"Missing independent checkout in {workflow}/{job}")
                    continue
                with self.subTest(workflow=workflow, job=job):
                    self.check_checkout(match.group())

    def check_checkout(self, script):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            upstream, mirror, workspace = (
                root / name for name in ("upstream", "mirror", "workspace")
            )
            env = {
                **os.environ,
                "GIT_CONFIG_GLOBAL": os.devnull,
                "GIT_CONFIG_NOSYSTEM": "1",
            }

            def git(*args):
                return subprocess.run(
                    ["git", *map(str, args)],
                    env=env,
                    check=True,
                    capture_output=True,
                    text=True,
                ).stdout.strip()

            git("init", upstream)
            for directory in (".github", "components"):
                (upstream / directory).mkdir()
                (upstream / directory / "fixture.txt").write_text(directory)
            git("-C", upstream, "add", ".")
            git(
                "-C",
                upstream,
                "-c",
                "user.name=CI",
                "-c",
                "user.email=ci@example.invalid",
                "-c",
                "commit.gpgsign=false",
                "commit",
                "-m",
                "fixture",
            )
            sha = git("-C", upstream, "rev-parse", "HEAD")
            git("clone", "--depth=1", upstream.as_uri(), mirror)
            git("-C", mirror, "checkout", "--detach", sha)

            # Model Git maintenance creating/removing a temporary pack while a
            # checkout is obtained. It must never become part of the clone.
            temporary_pack = mirror / ".git/objects/pack/tmp_pack_fixture"
            stop = threading.Event()

            def change_pack():
                while not stop.is_set():
                    temporary_pack.write_bytes(b"incomplete pack")
                    temporary_pack.unlink(missing_ok=True)

            writer = threading.Thread(target=change_pack)
            writer.start()
            try:
                subprocess.run(
                    ["bash", "-euo", "pipefail", "-c", script],
                    env={
                        **env,
                        "MIRROR": str(mirror),
                        "GITHUB_WORKSPACE": str(workspace),
                        "GITHUB_SHA": sha,
                        "SPARSE": "components",
                        "url": upstream.as_uri(),
                    },
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
            finally:
                stop.set()
                writer.join(timeout=5)
            shutil.rmtree(mirror)
            self.assertEqual(git("-C", workspace, "rev-parse", "HEAD"), sha)
            self.assertEqual(
                (workspace / "components/fixture.txt").read_text(), "components"
            )
            self.assertFalse(
                (workspace / ".git/objects/pack/tmp_pack_fixture").exists()
            )
            self.assertFalse((workspace / ".git/objects/info/alternates").exists())
            git("-C", workspace, "fsck", "--full")


if __name__ == "__main__":
    unittest.main()
