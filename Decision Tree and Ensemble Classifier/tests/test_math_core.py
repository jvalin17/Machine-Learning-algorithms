"""Tests for src.engine.math_core"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.engine.math_core import log2


def _approx(a, b, tolerance=1e-9):
    return abs(a - b) < tolerance


def test_log2_powers_of_two():
    assert _approx(log2(1.0), 0.0)
    assert _approx(log2(2.0), 1.0)
    assert _approx(log2(4.0), 2.0)
    assert _approx(log2(8.0), 3.0)
    assert _approx(log2(1024.0), 10.0)
    print("  PASS  log2 powers of two")


def test_log2_fractions():
    assert _approx(log2(0.5), -1.0)
    assert _approx(log2(0.25), -2.0)
    print("  PASS  log2 fractions")


def test_log2_non_powers():
    assert _approx(log2(3.0), 1.58496250072, tolerance=1e-6)
    assert _approx(log2(10.0), 3.32192809489, tolerance=1e-6)
    print("  PASS  log2 non-powers")


def test_log2_edge_cases():
    assert log2(0.0) == 0.0
    assert log2(-5.0) == 0.0
    print("  PASS  log2 edge cases")


def run_all():
    print("test_math_core")
    test_log2_powers_of_two()
    test_log2_fractions()
    test_log2_non_powers()
    test_log2_edge_cases()
    print()


if __name__ == "__main__":
    run_all()
