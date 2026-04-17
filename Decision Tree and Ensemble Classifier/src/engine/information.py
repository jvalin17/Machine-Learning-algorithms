"""Entropy, information gain, and feature selection."""

from src.engine.math_core import log2
from src.utils.data import count_values, split_by_feature


def entropy(rows, label_col):
    """Shannon entropy (base 2) of the label column in *rows*.

    Returns 0.0 for empty input. Maximum is log2(num_classes).
    """
    total = len(rows)
    if total == 0:
        return 0.0

    counts = count_values(row[label_col] for row in rows)
    result = 0.0
    for count in counts.values():
        proportion = count / total
        if proportion > 0.0:
            result -= proportion * log2(proportion)
    return result


def information_gain(rows, feature_col, label_col):
    """Information gain from splitting *rows* on *feature_col*.

    IG = H(parent) - weighted_sum(H(child_i))
    """
    parent_entropy = entropy(rows, label_col)
    total = len(rows)

    partitions = split_by_feature(rows, feature_col)

    child_entropy = 0.0
    for subset in partitions.values():
        weight = len(subset) / total
        child_entropy += weight * entropy(subset, label_col)

    return parent_entropy - child_entropy


def best_feature(rows, feature_cols, label_col):
    """Return (column_index, gain) for the feature with highest information gain."""
    best_gain = -1.0
    best_col = None

    for col in feature_cols:
        gain = information_gain(rows, col, label_col)
        if gain > best_gain:
            best_gain = gain
            best_col = col

    return best_col, best_gain
