#  SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0

import os
import sys
import tempfile

if "PYTHONHASHSEED" not in os.environ:
    os.environ["PYTHONHASHSEED"] = "0"

# SGLang's import chain imports prometheus_client at module top level
# (disaggregation/*/conn.py), and prometheus_client picks single- vs
# multi-process mode on first import. Here that import fires while
# `dynamo.sglang.main` loads — long before SGLang itself sets
# PROMETHEUS_MULTIPROC_DIR at sgl.Engine() startup
# (set_prometheus_multiproc_dir()). Without the env var set here, the main
# process (TokenizerManager) locks single-process mode and its metrics
# (TTFT/ITL/e2e latency, prompt|cached|generation tokens, ...) land in the
# in-memory default registry, where dynamo's MultiProcessCollector-based
# exposition cannot see them. Pre-setting the env var forces multiprocess
# mode from the first import; set_prometheus_multiproc_dir() honors a
# pre-set value ("User set" branch), so main process, scheduler children
# and the collector all share one directory.
if "PROMETHEUS_MULTIPROC_DIR" not in os.environ and "--enable-metrics" in sys.argv:
    os.environ["PROMETHEUS_MULTIPROC_DIR"] = tempfile.mkdtemp(
        prefix="dynamo-sglang-prom-multiproc-"
    )

if __name__ == "__main__":
    from dynamo.common.snapshot.restore_context import maybe_run_restore_standby_mode

    # Check before importing dynamo.sglang.main: restore standby mode must
    # capture env and hold without importing SGLang or constructing backend state.
    maybe_run_restore_standby_mode()

    from dynamo.sglang.main import main

    main()
