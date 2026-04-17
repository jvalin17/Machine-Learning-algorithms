"""Ensemble Classifier — K-Fold Cross-Validated Decision Trees.

Trains one tree per fold, evaluates each on the held-out fold,
and provides a majority-vote ensemble for new predictions.
"""

from src.classifier import DecisionTree
from src.utils.data import load_csv, count_values


class EnsembleClassifier:
    """Ensemble of DecisionTree classifiers trained on different data folds.

    Usage:
        ensemble = EnsembleClassifier(max_depth=10, min_samples=2)
        ensemble.fit(folds, headers, label_col=0)

        label = ensemble.predict_one(row)
        labels = ensemble.predict(rows)
        acc = ensemble.score(test_rows)
    """

    def __init__(self, max_depth=None, min_samples=2):
        self.max_depth = max_depth
        self.min_samples = min_samples
        self.trees = []
        self.headers = None
        self.label_col = None
        self.labels = None

    def fit(self, folds, headers=None, label_col=0):
        """Train one tree per fold, using remaining folds as training data.

        Args:
            folds:     list of row-lists, one per fold.
                       e.g. [fold1_rows, fold2_rows, ...]
            headers:   optional column name strings.
            label_col: index of the target column (default 0).

        Returns:
            self (for chaining).
        """
        self.headers = headers
        self.label_col = label_col
        self.trees = []

        all_rows = []
        for fold in folds:
            all_rows += fold
        self.labels = sorted(set(row[label_col] for row in all_rows))

        for i in range(len(folds)):
            # train on all folds except fold i
            train_rows = []
            for j in range(len(folds)):
                if j != i:
                    train_rows += folds[j]

            tree = DecisionTree(max_depth=self.max_depth, min_samples=self.min_samples)
            tree.fit(train_rows, headers=headers, label_col=label_col)
            self.trees.append(tree)

        return self

    def predict_one(self, row):
        """Classify a single row using majority vote across all trees."""
        votes = [tree.predict_one(row) for tree in self.trees]
        counts = count_values(votes)

        best_label = None
        best_count = -1
        for label, count in counts.items():
            if count > best_count:
                best_count = count
                best_label = label
        return best_label

    def predict(self, rows):
        """Classify multiple rows. Returns a list of predicted labels."""
        return [self.predict_one(row) for row in rows]

    def score(self, rows):
        """Return ensemble accuracy (0.0 to 1.0) against labeled rows."""
        predictions = self.predict(rows)
        correct = 0
        for pred, row in zip(predictions, rows):
            if pred == row[self.label_col]:
                correct += 1
        return correct / len(rows) if rows else 0.0

    def cross_validate(self, folds):
        """Return per-fold accuracy: each tree tested on its held-out fold.

        Returns:
            list of (fold_index, accuracy) tuples.
        """
        results = []
        for i in range(len(folds)):
            acc = self.trees[i].score(folds[i])
            results.append((i, acc))
        return results
