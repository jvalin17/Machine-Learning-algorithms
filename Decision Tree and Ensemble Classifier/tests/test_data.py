"""Tests for src.utils.data"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.utils.data import count_values, majority_class, split_by_feature


def test_count_values():
    result = count_values(["a", "b", "a", "c", "a", "b"])
    assert result == {"a": 3, "b": 2, "c": 1}
    assert count_values([]) == {}
    print("  PASS  count_values")


def test_majority_class():
    rows = [["e", "x"], ["e", "y"], ["p", "x"], ["e", "z"]]
    assert majority_class(rows, 0) == "e"

    rows = [["p"], ["p"], ["e"]]
    assert majority_class(rows, 0) == "p"
    print("  PASS  majority_class")


def test_split_by_feature():
    rows = [
        ["e", "x", "s"],
        ["p", "x", "y"],
        ["e", "b", "s"],
    ]
    result = split_by_feature(rows, 1)
    assert len(result) == 2
    assert len(result["x"]) == 2
    assert len(result["b"]) == 1
    assert result["b"][0] == ["e", "b", "s"]
    print("  PASS  split_by_feature")


def run_all():
    print("test_data")
    test_count_values()
    test_majority_class()
    test_split_by_feature()
    print()


if __name__ == "__main__":
    run_all()
