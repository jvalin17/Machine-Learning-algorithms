"""Decision Tree Classifier — ID3 (Entropy + Information Gain).

No external libraries. Zero imports beyond this package.

Usage:
    from src import DecisionTree, load_csv

    headers, rows = load_csv("data.csv")
    tree = DecisionTree(max_depth=8)
    tree.fit(rows, headers, label_col=0)

    prediction = tree.predict_one(new_row)
    score = tree.score(test_rows)
    report = tree.evaluate(test_rows)
"""

from src.classifier import DecisionTree
from src.ensemble import EnsembleClassifier
from src.utils.data import load_csv

__all__ = ["DecisionTree", "EnsembleClassifier", "load_csv"]
