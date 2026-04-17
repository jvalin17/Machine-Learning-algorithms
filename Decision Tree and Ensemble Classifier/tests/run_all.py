"""Run all test suites.

Usage:  python3 -m tests.run_all   (from project root)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tests.test_math_core import run_all as run_math
from tests.test_data import run_all as run_data
from tests.test_information import run_all as run_info
from tests.test_tree_ops import run_all as run_tree
from tests.test_metrics import run_all as run_metrics
from tests.test_classifier import run_all as run_classifier


if __name__ == "__main__":
    suites = [run_math, run_data, run_info, run_tree, run_metrics, run_classifier]
    failed = False

    for suite in suites:
        try:
            suite()
        except AssertionError as exc:
            print("  FAIL  %s" % exc)
            failed = True
        except Exception as exc:
            print("  ERROR %s: %s" % (type(exc).__name__, exc))
            failed = True

    print("---")
    if failed:
        print("Some tests FAILED.")
        sys.exit(1)
    else:
        print("All tests passed.")
