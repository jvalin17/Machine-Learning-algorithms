"""Tests for src.metrics.evaluation"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.metrics.evaluation import (
    accuracy, confusion_matrix, precision_recall_f1, format_confusion_matrix,
)


def _approx(a, b, tolerance=1e-9):
    return abs(a - b) < tolerance


def test_accuracy_perfect():
    assert accuracy(["e", "p", "e"], ["e", "p", "e"]) == 1.0
    print("  PASS  accuracy perfect")


def test_accuracy_partial():
    assert _approx(accuracy(["e", "p", "e"], ["e", "e", "e"]), 2.0 / 3)
    print("  PASS  accuracy partial")


def test_accuracy_empty():
    assert accuracy([], []) == 0.0
    print("  PASS  accuracy empty")


def test_confusion_matrix():
    preds   = ["e", "e", "p", "p", "e"]
    actuals = ["e", "p", "p", "p", "e"]
    labels  = ["e", "p"]

    cm = confusion_matrix(preds, actuals, labels)
    assert cm["e"]["e"] == 2
    assert cm["p"]["e"] == 1
    assert cm["p"]["p"] == 2
    assert cm["e"]["p"] == 0
    print("  PASS  confusion_matrix")


def test_precision_recall_f1():
    cm = {"e": {"e": 2, "p": 0}, "p": {"e": 1, "p": 2}}

    p, r, f = precision_recall_f1(cm, "e")
    assert _approx(p, 2.0 / 3)
    assert _approx(r, 1.0)
    print("  PASS  precision_recall_f1 for 'e'")

    p, r, f = precision_recall_f1(cm, "p")
    assert _approx(p, 1.0)
    assert _approx(r, 2.0 / 3)
    print("  PASS  precision_recall_f1 for 'p'")


def test_format_confusion_matrix():
    cm = {"e": {"e": 25, "p": 0}, "p": {"e": 0, "p": 25}}
    text = format_confusion_matrix(cm, ["e", "p"])
    assert "25" in text
    assert "0" in text
    print("  PASS  format_confusion_matrix")


def run_all():
    print("test_metrics")
    test_accuracy_perfect()
    test_accuracy_partial()
    test_accuracy_empty()
    test_confusion_matrix()
    test_precision_recall_f1()
    test_format_confusion_matrix()
    print()


if __name__ == "__main__":
    run_all()
