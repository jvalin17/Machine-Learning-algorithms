"""Train and evaluate the decision tree on mushroom data.

Run:  python3 main.py
"""

from src import DecisionTree, load_csv
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


def main():
    _, train_rows = load_csv("resources/mushroom_train.csv")
    _, test_rows = load_csv("resources/mushroom_test.csv")

    tree = DecisionTree(max_depth=10, min_samples=2)
    tree.fit(train_rows, headers=FEATURE_NAMES, label_col=0)

    # --- tree structure ---
    print("=" * 50)
    print("  Decision Tree — Mushroom Classification")
    print("=" * 50)
    print()
    print(readable_tree(tree.root, FEATURE_NAMES))

    # --- accuracy ---
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

    # --- detailed evaluation ---
    result = tree.evaluate(test_rows)

    print("-" * 50)
    print("  Confusion Matrix (test set)")
    print("-" * 50)
    labels = tree.labels
    label_names = [CLASS_NAMES.get(l, l) for l in labels]
    col_width = max(len(n) for n in label_names) + 2

    # header row
    print(" " * (col_width + 4) + "Predicted")
    header = " " * (col_width + 4) + "".join(n.rjust(col_width) for n in label_names)
    print(header)

    # data rows
    matrix = result["confusion_matrix"]
    for label, name in zip(labels, label_names):
        row = ("  " + name).ljust(col_width + 4)
        for pred_label in labels:
            row += str(matrix[label][pred_label]).rjust(col_width)
        print(row)
    print()

    # --- per-class metrics ---
    print("-" * 50)
    print("  Per-class Metrics (test set)")
    print("-" * 50)
    print("  %-12s  %9s  %9s  %9s" % ("Class", "Precision", "Recall", "F1"))
    print("  " + "-" * 43)
    for label in labels:
        m = result["per_class"][label]
        name = CLASS_NAMES.get(label, label)
        print("  %-12s  %9.3f  %9.3f  %9.3f"
              % (name, m["precision"], m["recall"], m["f1"]))
    print()


if __name__ == "__main__":
    main()
