"""Tree node, building, prediction, pruning, and display."""

from src.utils.data import count_values, majority_class, split_by_feature
from src.engine.information import best_feature


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

class TreeNode:
    """A single node in the decision tree.

    Leaf node:     TreeNode(label="e")
    Internal node: TreeNode(feature=2, children={"s": child, ...}, default="p")
    """

    __slots__ = ("feature", "children", "label", "default")

    def __init__(self, label=None, feature=None, children=None, default=None):
        self.label = label
        self.feature = feature
        self.children = children
        self.default = default


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

def build_tree(rows, feature_cols, label_col, depth, max_depth, min_samples):
    """Recursively build an ID3 tree. Returns a TreeNode."""

    # leaf: empty data
    if not rows:
        return TreeNode(label=None)

    # leaf: pure node (all same label)
    labels = count_values(row[label_col] for row in rows)
    if len(labels) == 1:
        return TreeNode(label=next(iter(labels)))

    # leaf: depth limit or too few samples
    if max_depth is not None and depth >= max_depth:
        return TreeNode(label=majority_class(rows, label_col))

    if len(rows) < min_samples:
        return TreeNode(label=majority_class(rows, label_col))

    # leaf: no features left
    if not feature_cols:
        return TreeNode(label=majority_class(rows, label_col))

    # pick best feature
    split_col, gain = best_feature(rows, feature_cols, label_col)

    if gain <= 0.0:
        return TreeNode(label=majority_class(rows, label_col))

    # split and recurse
    remaining = [c for c in feature_cols if c != split_col]
    partitions = split_by_feature(rows, split_col)

    children = {}
    for value, subset in partitions.items():
        children[value] = build_tree(
            subset, remaining, label_col, depth + 1, max_depth, min_samples
        )

    return TreeNode(
        feature=split_col,
        children=children,
        default=majority_class(rows, label_col),
    )


# ---------------------------------------------------------------------------
# Predict
# ---------------------------------------------------------------------------

def predict_one(node, row):
    """Walk the tree to classify a single row. Returns the predicted label."""
    while node.label is None:
        value = row[node.feature]
        child = node.children.get(value)

        if child is None:
            return node.default
        node = child

    return node.label


# ---------------------------------------------------------------------------
# Prune (reduced-error pruning)
# ---------------------------------------------------------------------------

def prune(node, validation_rows, label_col):
    """Post-prune the tree using a validation set.

    Replaces a subtree with a leaf if accuracy on *validation_rows*
    does not decrease.
    """
    if node.label is not None:
        return node

    if not validation_rows:
        return node

    # recurse into children first
    partitions = split_by_feature(validation_rows, node.feature)
    for value, child in node.children.items():
        subset = partitions.get(value, [])
        node.children[value] = prune(child, subset, label_col)

    # count correct predictions with the subtree
    subtree_correct = 0
    for row in validation_rows:
        if predict_one(node, row) == row[label_col]:
            subtree_correct += 1

    # count correct predictions if we replace with a leaf
    leaf_label = majority_class(validation_rows, label_col)
    leaf_correct = 0
    for row in validation_rows:
        if row[label_col] == leaf_label:
            leaf_correct += 1

    # prune if leaf is at least as good
    if leaf_correct >= subtree_correct:
        return TreeNode(label=leaf_label)

    return node


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def tree_to_text(node, headers=None, indent=""):
    """Return a human-readable multi-line string of the tree."""
    if node.label is not None:
        return indent + "-> " + str(node.label) + "\n"

    name = headers[node.feature] if headers else str(node.feature)
    lines = indent + "[" + name + "]\n"
    for value in sorted(node.children):
        child = node.children[value]
        lines += indent + "  " + name + " = " + str(value) + "\n"
        lines += tree_to_text(child, headers, indent + "    ")
    return lines
