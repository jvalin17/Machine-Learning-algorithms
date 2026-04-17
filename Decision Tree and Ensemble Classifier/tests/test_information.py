"""Tests for src.engine.information"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.engine.information import entropy, information_gain, best_feature


def _approx(a, b, tolerance=1e-6):
    return abs(a - b) < tolerance


def test_entropy_pure():
    rows = [["e", "x"], ["e", "y"], ["e", "z"]]
    assert entropy(rows, 0) == 0.0
    print("  PASS  entropy pure node")


def test_entropy_balanced():
    rows = [["e", "x"], ["p", "y"]]
    assert _approx(entropy(rows, 0), 1.0)
    print("  PASS  entropy balanced (50/50)")


def test_entropy_empty():
    assert entropy([], 0) == 0.0
    print("  PASS  entropy empty")


def test_entropy_three_classes():
    rows = [["a"], ["b"], ["c"]]
    assert _approx(entropy(rows, 0), 1.58496, tolerance=1e-3)
    print("  PASS  entropy three classes")


def test_information_gain_perfect_split():
    rows = [
        ["e", "x"],
        ["e", "x"],
        ["p", "y"],
        ["p", "y"],
    ]
    gain = information_gain(rows, 1, 0)
    parent_ent = entropy(rows, 0)
    assert _approx(gain, parent_ent)
    print("  PASS  information_gain perfect split")


def test_information_gain_no_split():
    rows = [
        ["e", "x"],
        ["p", "x"],
    ]
    gain = information_gain(rows, 1, 0)
    assert _approx(gain, 0.0)
    print("  PASS  information_gain no split")


def test_best_feature():
    rows = [
        ["e", "a", "z"],
        ["e", "a", "z"],
        ["p", "b", "z"],
        ["p", "b", "z"],
    ]
    col, gain = best_feature(rows, [1, 2], 0)
    assert col == 1
    assert gain > 0
    print("  PASS  best_feature")


def run_all():
    print("test_information")
    test_entropy_pure()
    test_entropy_balanced()
    test_entropy_empty()
    test_entropy_three_classes()
    test_information_gain_perfect_split()
    test_information_gain_no_split()
    test_best_feature()
    print()


if __name__ == "__main__":
    run_all()
