"""Train and evaluate the decision tree and ensemble on mushroom data.

Run:  python3 main.py
"""

from src import DecisionTree, EnsembleClassifier, load_csv
from src.utils.display import Printer

# ---------------------------------------------------------------------------
# Mushroom dataset labels
# ---------------------------------------------------------------------------

FEATURE_NAMES = ["class", "cap_shape", "cap_surface", "cap_color", "bruises", "odor"]

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

LABEL_COL = 0

out = Printer()


def class_name(code):
    """Translate a class code like 'e' to 'edible'."""
    return VALUE_LABELS["class"].get(code, code)


# ---------------------------------------------------------------------------
# Decision Tree
# ---------------------------------------------------------------------------

def run_decision_tree():
    """Train a single decision tree and report results."""
    _, train_rows = load_csv("resources/mushroom_train.csv")
    _, test_rows = load_csv("resources/mushroom_test.csv")

    tree = DecisionTree(max_depth=10, min_samples=2)
    tree.fit(train_rows, headers=FEATURE_NAMES, label_col=LABEL_COL)

    out.heading("Decision Tree — Mushroom Classification")
    out.tree(tree.root, FEATURE_NAMES, value_labels=VALUE_LABELS, label_fn=class_name)
    out.blank()

    out.section("Accuracy")
    out.accuracy("Training set", tree.score(train_rows), len(train_rows))
    out.accuracy("Test set", tree.score(test_rows), len(test_rows))
    out.blank()

    evaluation = tree.evaluate(test_rows)

    out.section("Confusion Matrix (test set)")
    out.confusion_matrix(evaluation, tree.labels, label_fn=class_name)

    out.section("Per-class Metrics (test set)")
    out.class_metrics(evaluation, tree.labels, label_fn=class_name)


# ---------------------------------------------------------------------------
# Ensemble
# ---------------------------------------------------------------------------

def run_ensemble():
    """Train a 2-fold cross-validated ensemble and report results."""
    _, fold_1_rows = load_csv("resources/ensemble_fold_1.csv")
    _, fold_2_rows = load_csv("resources/ensemble_fold_2.csv")
    folds = [fold_1_rows, fold_2_rows]

    ensemble = EnsembleClassifier(max_depth=10, min_samples=2)
    ensemble.fit(folds, headers=FEATURE_NAMES, label_col=LABEL_COL)

    out.heading("Ensemble Classifier — 2-Fold Cross-Validation")

    cv_results = ensemble.cross_validate(folds)

    out.section("Cross-Validation Accuracy")
    for fold_index, fold_accuracy in cv_results:
        out.accuracy("Fold %d" % (fold_index + 1), fold_accuracy, len(folds[fold_index]))

    average_accuracy = sum(acc for _, acc in cv_results) / len(cv_results)
    out.accuracy_plain("Average", average_accuracy * 100)
    out.blank()

    all_rows = fold_1_rows + fold_2_rows
    ensemble_accuracy = ensemble.score(all_rows)

    out.section("Ensemble Majority-Vote Accuracy (all data)")
    out.accuracy("Combined", ensemble_accuracy, len(all_rows))
    out.blank()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    run_decision_tree()
    out.blank()
    run_ensemble()


if __name__ == "__main__":
    main()
