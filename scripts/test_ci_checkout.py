# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise the cached-checkout commands in the internal CI workflows."""

import os
import re
import shutil
import signal
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
                depth = next(
                    step.get("env", {}).get("DEPTH")
                    for step in definition["steps"]
                    if "git clone --no-local" in step.get("run", "")
                )
                for stale_ref in (False, True):
                    with self.subTest(workflow=workflow, job=job, stale_ref=stale_ref):
                        self.check_checkout(match.group(), stale_ref, depth)

    def check_checkout(self, script, stale_ref, depth=None):
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
            env.pop("DEPTH", None)
            if depth is not None:
                env["DEPTH"] = str(depth)

            def git(*args):
                return subprocess.run(
                    ["git", *map(str, args)],
                    env=env,
                    check=True,
                    capture_output=True,
                    text=True,
                ).stdout.strip()

            def commit(message):
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
                    message,
                )
                return git("-C", upstream, "rev-parse", "HEAD")

            git("init", "--initial-branch=main", upstream)
            git("-C", upstream, "config", "uploadpack.allowFilter", "true")
            for directory in (".github", "components"):
                (upstream / directory).mkdir()
                (upstream / directory / "fixture.txt").write_text(directory)
            old_sha = sha = commit("fixture")
            git("clone", "--depth=1", "--filter=blob:none", upstream.as_uri(), mirror)
            expected = "components"
            if stale_ref:
                # Fetching a SHA leaves the mirror's original main branch behind.
                # An unrestricted clone can repeatedly fetch that shallow tip.
                (upstream / "components/fixture.txt").write_text("middle")
                commit("middle")
                expected = "components updated"
                (upstream / "components/fixture.txt").write_text(expected)
                sha = commit("updated")
                git("-C", mirror, "fetch", "--depth=1", "origin", sha)
                self.assertEqual(git("-C", mirror, "rev-parse", "main"), old_sha)
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
                with subprocess.Popen(
                    ["bash", "-euo", "pipefail", "-c", script],
                    env={
                        **env,
                        "MIRROR": str(mirror),
                        "GITHUB_WORKSPACE": str(workspace),
                        "GITHUB_SHA": sha,
                        "SPARSE": "components",
                        "url": upstream.as_uri(),
                    },
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    start_new_session=True,
                ) as process:
                    try:
                        _, stderr = process.communicate(timeout=10)
                    except subprocess.TimeoutExpired:
                        # Git can spawn more fetch processes while resolving a
                        # shallow promisor tip. Stop the whole group on failure.
                        try:
                            os.killpg(process.pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                        process.communicate()
                        raise
                    self.assertEqual(process.returncode, 0, stderr)
            finally:
                stop.set()
                writer.join(timeout=5)
            shutil.rmtree(mirror)
            self.assertEqual(git("-C", workspace, "rev-parse", "HEAD"), sha)
            self.assertEqual(
                (workspace / "components/fixture.txt").read_text(), expected
            )
            if stale_ref:
                refs = git("-C", workspace, "for-each-ref", "--format=%(objectname)")
                self.assertNotIn(old_sha, refs)
            self.assertFalse(
                (workspace / ".git/objects/pack/tmp_pack_fixture").exists()
            )
            self.assertFalse((workspace / ".git/objects/info/alternates").exists())
            git("-C", workspace, "fsck", "--full")


if __name__ == "__main__":
    unittest.main()
