"""Tests for src.ensemble"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.ensemble import EnsembleClassifier


# Both folds share the same feature values so every tree has seen them
FOLD1 = [
    ["e", "a", "x"],
    ["e", "a", "y"],
    ["p", "b", "x"],
    ["p", "b", "y"],
]

FOLD2 = [
    ["e", "a", "x"],
    ["e", "a", "y"],
    ["p", "b", "x"],
    ["p", "b", "y"],
]

TEST = [
    ["e", "a", "x"],
    ["p", "b", "y"],
]


def test_fit_and_predict():
    ens = EnsembleClassifier(max_depth=10)
    ens.fit([FOLD1, FOLD2], label_col=0)
    preds = ens.predict(TEST)
    assert preds == ["e", "p"]
    print("  PASS  fit and predict")


def test_predict_one():
    ens = EnsembleClassifier()
    ens.fit([FOLD1, FOLD2], label_col=0)
    assert ens.predict_one(["?", "a", "x"]) == "e"
    assert ens.predict_one(["?", "b", "y"]) == "p"
    print("  PASS  predict_one")


def test_score():
    ens = EnsembleClassifier()
    ens.fit([FOLD1, FOLD2], label_col=0)
    assert ens.score(TEST) == 1.0
    print("  PASS  score")


def test_cross_validate():
    ens = EnsembleClassifier()
    ens.fit([FOLD1, FOLD2], label_col=0)
    results = ens.cross_validate([FOLD1, FOLD2])
    assert len(results) == 2
    for fold_idx, acc in results:
        assert acc == 1.0  # same distribution, should be perfect
    print("  PASS  cross_validate")


def test_two_trees_created():
    ens = EnsembleClassifier()
    ens.fit([FOLD1, FOLD2], label_col=0)
    assert len(ens.trees) == 2
    print("  PASS  two trees created")


def test_three_folds():
    fold3 = [["e", "a", "y"], ["p", "b", "x"]]
    ens = EnsembleClassifier()
    ens.fit([FOLD1, FOLD2, fold3], label_col=0)
    assert len(ens.trees) == 3
    assert ens.predict_one(["?", "a", "x"]) == "e"
    print("  PASS  three folds")


def test_unseen_value():
    ens = EnsembleClassifier()
    ens.fit([FOLD1, FOLD2], label_col=0)
    result = ens.predict_one(["?", "z", "x"])
    assert result is not None
    print("  PASS  unseen value handled")


def run_all():
    print("test_ensemble")
    test_fit_and_predict()
    test_predict_one()
    test_score()
    test_cross_validate()
    test_two_trees_created()
    test_three_folds()
    test_unseen_value()
    print()


if __name__ == "__main__":
    run_all()
