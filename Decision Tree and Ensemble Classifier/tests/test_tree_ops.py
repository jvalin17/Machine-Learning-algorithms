"""Tests for src.utils.tree_ops"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.utils.tree_ops import TreeNode, build_tree, predict_one, prune, tree_to_text


SIMPLE_DATA = [
    ["e", "a", "x"],
    ["e", "a", "y"],
    ["p", "b", "x"],
    ["p", "b", "y"],
]

MIXED_DATA = [
    ["e", "a", "x"],
    ["p", "a", "y"],
    ["p", "b", "x"],
    ["p", "b", "y"],
]


def test_build_pure_split():
    tree = build_tree(SIMPLE_DATA, [1, 2], label_col=0,
                      depth=0, max_depth=10, min_samples=1)
    assert tree.label is None
    assert tree.feature == 1
    assert tree.children["a"].label == "e"
    assert tree.children["b"].label == "p"
    print("  PASS  build pure split")


def test_build_depth_limit():
    tree = build_tree(MIXED_DATA, [1, 2], label_col=0,
                      depth=0, max_depth=1, min_samples=1)
    assert tree.label is None
    for child in tree.children.values():
        assert child.label is not None
    print("  PASS  build depth limit")


def test_build_min_samples():
    tree = build_tree(SIMPLE_DATA, [1, 2], label_col=0,
                      depth=0, max_depth=10, min_samples=100)
    assert tree.label is not None
    print("  PASS  build min_samples stop")


def test_predict_one():
    tree = build_tree(SIMPLE_DATA, [1, 2], label_col=0,
                      depth=0, max_depth=10, min_samples=1)
    assert predict_one(tree, ["?", "a", "x"]) == "e"
    assert predict_one(tree, ["?", "b", "y"]) == "p"
    print("  PASS  predict_one")


def test_predict_unseen_value():
    tree = build_tree(SIMPLE_DATA, [1, 2], label_col=0,
                      depth=0, max_depth=10, min_samples=1)
    result = predict_one(tree, ["?", "c", "x"])
    assert result is not None
    print("  PASS  predict unseen value (fallback)")


def test_prune():
    tree = build_tree(MIXED_DATA, [1, 2], label_col=0,
                      depth=0, max_depth=10, min_samples=1)
    validation = [
        ["p", "a", "x"],
        ["p", "a", "y"],
        ["p", "b", "x"],
    ]
    pruned = prune(tree, validation, label_col=0)
    assert predict_one(pruned, ["?", "a", "x"]) is not None
    print("  PASS  prune produces valid tree")


def test_tree_to_text():
    tree = build_tree(SIMPLE_DATA, [1, 2], label_col=0,
                      depth=0, max_depth=10, min_samples=1)
    text = tree_to_text(tree, headers=["label", "shape", "color"])
    assert "shape" in text
    assert "-> e" in text
    assert "-> p" in text
    print("  PASS  tree_to_text")


def run_all():
    print("test_tree_ops")
    test_build_pure_split()
    test_build_depth_limit()
    test_build_min_samples()
    test_predict_one()
    test_predict_unseen_value()
    test_prune()
    test_tree_to_text()
    print()


if __name__ == "__main__":
    run_all()
