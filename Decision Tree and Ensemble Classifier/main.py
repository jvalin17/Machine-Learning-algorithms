"""Train and evaluate the decision tree and ensemble on mushroom data.

Run:  python3 main.py
"""

from src import DecisionTree, EnsembleClassifier, load_csv
from src.metrics import format_confusion_matrix


# Map generic CSV headers to meaningful feature names
FEATURE_NAMES = ["class", "cap_shape", "cap_surface", "cap_color", "bruises", "odor"]

# Map coded values to readable labels
VALUE_LABELS = {
    "class":       {"e": "edible", "p": "poisonous"},
    "cap_shape":   {"b": "bell", "c": "conical", "x": "convex",
                    "f": "flat", "k": "knobbed", "s": "sunken"},
    "cap_surface": {"f": "fibrous", "g": "grooves", "y": "scaly", "s": "smooth"},
    "cap_color":   {"n": "brown", "b": "buff", "c": "cinnamon", "g": "gray",
                    "r": "green", "p": "pink", "u": "purple", "e": "red",
                    "w": "white", "y": "yellow", "t": "tan", "f": "fawn"},
    "bruises":     {"t": "yes", "f": "no"},
    "odor":        {"a": "almond", "l": "anise", "c": "creosote", "y": "fishy",
                    "f": "foul", "m": "musty", "n": "none", "p": "pungent",
                    "s": "spicy"},
}

CLASS_NAMES = {"e": "edible", "p": "poisonous"}


def readable_tree(node, headers, indent=""):
    """Build a human-readable tree string using feature and value names."""
    if node.label is not None:
        return indent + "-> " + CLASS_NAMES.get(node.label, node.label) + "\n"

    feature_name = headers[node.feature] if headers else str(node.feature)
    value_map = VALUE_LABELS.get(feature_name, {})

    lines = indent + "[" + feature_name + "]\n"
    for value in sorted(node.children):
        readable_val = value_map.get(value, value)
        child = node.children[value]
        lines += indent + "  " + feature_name + " = " + readable_val + "\n"
        lines += readable_tree(child, headers, indent + "    ")
    return lines


def print_evaluation(label, result, labels):
    """Print confusion matrix and per-class metrics."""
    print("-" * 50)
    print("  Confusion Matrix (%s)" % label)
    print("-" * 50)
    label_names = [CLASS_NAMES.get(l, l) for l in labels]
    col_width = max(len(n) for n in label_names) + 2

    print(" " * (col_width + 4) + "Predicted")
    print(" " * (col_width + 4) + "".join(n.rjust(col_width) for n in label_names))

    matrix = result["confusion_matrix"]
    for lbl, name in zip(labels, label_names):
        row = ("  " + name).ljust(col_width + 4)
        for pred_label in labels:
            row += str(matrix[lbl][pred_label]).rjust(col_width)
        print(row)
    print()

    print("-" * 50)
    print("  Per-class Metrics (%s)" % label)
    print("-" * 50)
    print("  %-12s  %9s  %9s  %9s" % ("Class", "Precision", "Recall", "F1"))
    print("  " + "-" * 43)
    for lbl in labels:
        m = result["per_class"][lbl]
        name = CLASS_NAMES.get(lbl, lbl)
        print("  %-12s  %9.3f  %9.3f  %9.3f"
              % (name, m["precision"], m["recall"], m["f1"]))
    print()


def run_decision_tree():
    """Single decision tree on full train/test split."""
    _, train_rows = load_csv("resources/mushroom_train.csv")
    _, test_rows = load_csv("resources/mushroom_test.csv")

    tree = DecisionTree(max_depth=10, min_samples=2)
    tree.fit(train_rows, headers=FEATURE_NAMES, label_col=0)

    print("=" * 50)
    print("  Decision Tree — Mushroom Classification")
    print("=" * 50)
    print()
    print(readable_tree(tree.root, FEATURE_NAMES))

    train_acc = tree.score(train_rows)
    test_acc = tree.score(test_rows)

    print("-" * 50)
    print("  Accuracy")
    print("-" * 50)
    print("  Training set : %.2f%%  (%d / %d)"
          % (train_acc * 100, int(train_acc * len(train_rows)), len(train_rows)))
    print("  Test set     : %.2f%%  (%d / %d)"
          % (test_acc * 100, int(test_acc * len(test_rows)), len(test_rows)))
    print()

    result = tree.evaluate(test_rows)
    print_evaluation("test set", result, tree.labels)


def run_ensemble():
    """Ensemble classifier using 2-fold cross-validation."""
    _, fold1 = load_csv("resources/ensemble_fold_1.csv")
    _, fold2 = load_csv("resources/ensemble_fold_2.csv")
    folds = [fold1, fold2]

    ensemble = EnsembleClassifier(max_depth=10, min_samples=2)
    ensemble.fit(folds, headers=FEATURE_NAMES, label_col=0)

    print("=" * 50)
    print("  Ensemble Classifier — 2-Fold Cross-Validation")
    print("=" * 50)
    print()

    # per-fold accuracy (each tree tested on its held-out fold)
    cv_results = ensemble.cross_validate(folds)
    print("-" * 50)
    print("  Cross-Validation Accuracy")
    print("-" * 50)
    for fold_idx, acc in cv_results:
        print("  Fold %d : %.2f%%  (%d / %d)"
              % (fold_idx + 1, acc * 100, int(acc * len(folds[fold_idx])),
                 len(folds[fold_idx])))

    avg_acc = sum(acc for _, acc in cv_results) / len(cv_results)
    print("  Average: %.2f%%" % (avg_acc * 100))
    print()

    # ensemble majority-vote accuracy on all data
    all_rows = fold1 + fold2
    ensemble_acc = ensemble.score(all_rows)
    print("-" * 50)
    print("  Ensemble Majority-Vote Accuracy (all data)")
    print("-" * 50)
    print("  %.2f%%  (%d / %d)"
          % (ensemble_acc * 100, int(ensemble_acc * len(all_rows)), len(all_rows)))
    print()


def main():
    run_decision_tree()
    print()
    run_ensemble()


if __name__ == "__main__":
    main()
