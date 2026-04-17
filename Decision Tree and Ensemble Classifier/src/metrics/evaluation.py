"""Accuracy, confusion matrix, precision, recall, F1. No imports."""


def accuracy(predictions, actuals):
    """Fraction of matching predictions (0.0 to 1.0)."""
    if not predictions:
        return 0.0
    correct = 0
    for pred, actual in zip(predictions, actuals):
        if pred == actual:
            correct += 1
    return correct / len(predictions)


def confusion_matrix(predictions, actuals, labels):
    """Build a confusion matrix as {actual: {predicted: count}}.

    *labels* defines the row/column order.
    """
    matrix = {}
    for label in labels:
        matrix[label] = {}
        for label2 in labels:
            matrix[label][label2] = 0

    for pred, actual in zip(predictions, actuals):
        matrix[actual][pred] = matrix[actual].get(pred, 0) + 1

    return matrix


def format_confusion_matrix(matrix, labels):
    """Return a formatted string table for a confusion matrix."""
    col_width = max(len(str(label)) for label in labels) + 2
    header = " " * col_width + "".join(str(l).rjust(col_width) for l in labels)
    lines = [header]
    for actual in labels:
        row = str(actual).rjust(col_width)
        for predicted in labels:
            row += str(matrix[actual][predicted]).rjust(col_width)
        lines.append(row)
    return "\n".join(lines)


def precision_recall_f1(matrix, positive_label):
    """Compute precision, recall, and F1 for a given class label.

    Returns:
        (precision, recall, f1) — each a float in [0.0, 1.0].
    """
    tp = matrix[positive_label][positive_label]

    fp = 0
    for actual in matrix:
        if actual != positive_label:
            fp += matrix[actual].get(positive_label, 0)

    fn = 0
    for predicted in matrix[positive_label]:
        if predicted != positive_label:
            fn += matrix[positive_label][predicted]

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return precision, recall, f1
