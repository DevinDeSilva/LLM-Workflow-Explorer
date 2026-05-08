#!/usr/bin/env python
"""Run the analysis2 evaluator for chatbs-base.

The implementation lives in evaluations/biomni-base/biomni_analysis2.py because
the metric and extraction logic is shared across these evaluation folders. This
wrapper only changes the default evaluation folder to chatbs-base.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve()
COMMON_ANALYSIS_PATH = SCRIPT_PATH.parents[1] / "biomni-base" / "biomni_analysis2.py"


def load_common_analysis():
    spec = importlib.util.spec_from_file_location(
        "analysis2_common",
        COMMON_ANALYSIS_PATH,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import analysis2 common module: {COMMON_ANALYSIS_PATH}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    analysis = load_common_analysis()
    args = sys.argv[1:]
    if "--evaluation" not in args:
        args = ["--evaluation", "chatbs-base", *args]

    original_argv = sys.argv
    try:
        sys.argv = [original_argv[0], *args]
        analysis.main()
    finally:
        sys.argv = original_argv


if __name__ == "__main__":
    main()
