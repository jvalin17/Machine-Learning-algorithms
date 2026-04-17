"""Public API — the DecisionTree class."""

from src.utils.tree_ops import build_tree, predict_one, prune, tree_to_text
from src.metrics.evaluation import accuracy, confusion_matrix, precision_recall_f1


class DecisionTree:
    """ID3 Decision Tree Classifier.

    Usage:
        tree = DecisionTree(max_depth=5, min_samples=3)
        tree.fit(train_rows, headers, label_col=0)

        label = tree.predict_one(row)
        labels = tree.predict(rows)
        acc = tree.score(test_rows)
        report = tree.evaluate(test_rows)
    """

    def __init__(self, max_depth=None, min_samples=2):
        self.max_depth = max_depth
        self.min_samples = min_samples
        self.root = None
        self.headers = None
        self.label_col = None
        self.labels = None

    def fit(self, rows, headers=None, label_col=0):
        """Build the tree from training data.

        Args:
            rows:      list of row-lists (each row is a list of strings).
            headers:   optional column name strings.
            label_col: index of the target column (default 0).

        Returns:
            self (for chaining).
        """
        self.headers = headers
        self.label_col = label_col
        self.labels = sorted(set(row[label_col] for row in rows))

        feature_cols = [i for i in range(len(rows[0])) if i != label_col]

        self.root = build_tree(
            rows, feature_cols, label_col,
            depth=0,
            max_depth=self.max_depth,
            min_samples=self.min_samples,
        )
        return self

    def prune(self, validation_rows):
        """Reduced-error post-pruning using a held-out validation set.

        Returns:
            self (for chaining).
        """
        if self.root is not None:
            self.root = prune(self.root, validation_rows, self.label_col)
        return self

    def predict_one(self, row):
        """Classify a single row. Returns the predicted label string."""
        return predict_one(self.root, row)

    def predict(self, rows):
        """Classify multiple rows. Returns a list of predicted labels."""
        return [predict_one(self.root, row) for row in rows]

    def score(self, rows):
        """Return accuracy (0.0 to 1.0) against labeled rows."""
        predictions = self.predict(rows)
        actuals = [row[self.label_col] for row in rows]
        return accuracy(predictions, actuals)

    def evaluate(self, rows):
        """Full evaluation: accuracy, confusion matrix, per-class metrics.

        Returns:
            dict with keys "accuracy", "confusion_matrix", "per_class".
        """
        predictions = self.predict(rows)
        actuals = [row[self.label_col] for row in rows]

        acc = accuracy(predictions, actuals)
        matrix = confusion_matrix(predictions, actuals, self.labels)

        per_class = {}
        for label in self.labels:
            p, r, f = precision_recall_f1(matrix, label)
            per_class[label] = {"precision": p, "recall": r, "f1": f}

        return {"accuracy": acc, "confusion_matrix": matrix, "per_class": per_class}

    def display(self):
        """Return a human-readable string of the tree structure."""
        if self.root is None:
            return "(empty tree)"
        return tree_to_text(self.root, self.headers)
