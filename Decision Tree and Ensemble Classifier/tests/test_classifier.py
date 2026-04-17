"""Tests for src.classifier (end-to-end)"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.classifier import DecisionTree


TRAIN = [
    ["e", "a", "x"],
    ["e", "a", "y"],
    ["e", "c", "x"],
    ["p", "b", "x"],
    ["p", "b", "y"],
    ["p", "b", "z"],
]

TEST = [
    ["e", "a", "x"],
    ["p", "b", "y"],
    ["e", "c", "x"],
]


def test_fit_and_predict():
    tree = DecisionTree()
    tree.fit(TRAIN, headers=["label", "shape", "surface"], label_col=0)
    preds = tree.predict(TEST)
    assert preds == ["e", "p", "e"]
    print("  PASS  fit and predict")


def test_predict_one():
    tree = DecisionTree()
    tree.fit(TRAIN, label_col=0)
    assert tree.predict_one(["?", "a", "x"]) == "e"
    assert tree.predict_one(["?", "b", "z"]) == "p"
    print("  PASS  predict_one")


def test_score():
    tree = DecisionTree()
    tree.fit(TRAIN, label_col=0)
    assert tree.score(TEST) == 1.0
    print("  PASS  score")


def test_evaluate():
    tree = DecisionTree()
    tree.fit(TRAIN, label_col=0)
    result = tree.evaluate(TEST)
    assert result["accuracy"] == 1.0
    assert result["per_class"]["e"]["f1"] == 1.0
    print("  PASS  evaluate")


def test_display():
    tree = DecisionTree()
    tree.fit(TRAIN, headers=["label", "shape", "surface"], label_col=0)
    text = tree.display()
    assert "shape" in text
    print("  PASS  display")


def test_max_depth():
    tree = DecisionTree(max_depth=1)
    tree.fit(TRAIN, label_col=0)
    for child in tree.root.children.values():
        assert child.label is not None
    print("  PASS  max_depth constraint")


def test_prune():
    tree = DecisionTree(max_depth=10)
    tree.fit(TRAIN, label_col=0)
    tree.prune(TEST)
    assert tree.score(TEST) == 1.0
    print("  PASS  prune")


def test_unseen_feature_value():
    tree = DecisionTree()
    tree.fit(TRAIN, label_col=0)
    result = tree.predict_one(["?", "z", "x"])
    assert result is not None
    print("  PASS  unseen feature value handled")


def run_all():
    print("test_classifier")
    test_fit_and_predict()
    test_predict_one()
    test_score()
    test_evaluate()
    test_display()
    test_max_depth()
    test_prune()
    test_unseen_feature_value()
    print()


if __name__ == "__main__":
    run_all()
