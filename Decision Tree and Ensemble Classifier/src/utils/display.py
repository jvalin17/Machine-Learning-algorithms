"""Centralized display utility. All formatted output goes through Printer.

This makes it easy to swap the output target (terminal, file, log, UI)
by changing a single write method.
"""


class Printer:
    """Single entry point for all formatted output.

    By default writes to stdout. Pass a custom *write_fn* to redirect
    output (e.g. to a file, a list, or a GUI widget).
    """

    def __init__(self, write_fn=None, width=50):
        self._write = write_fn if write_fn else self._default_write
        self._width = width

    @staticmethod
    def _default_write(text):
        print(text)

    # ----- primitives -----

    def line(self, text=""):
        """Print a single line of text."""
        self._write(text)

    def blank(self):
        """Print an empty line."""
        self._write("")

    def heading(self, title):
        """Print a major section heading."""
        self._write("=" * self._width)
        self._write("  " + title)
        self._write("=" * self._width)
        self._write("")

    def section(self, title):
        """Print a sub-section divider."""
        self._write("-" * self._width)
        self._write("  " + title)
        self._write("-" * self._width)

    # ----- domain formatters -----

    def accuracy(self, dataset_name, score, total):
        """Print one accuracy row."""
        correct = int(score * total)
        self._write("  %-14s: %.2f%%  (%d / %d)"
                     % (dataset_name, score * 100, correct, total))

    def accuracy_plain(self, label, percentage):
        """Print an accuracy row without a count."""
        self._write("  %-14s: %.2f%%" % (label, percentage))

    def confusion_matrix(self, evaluation, class_labels, label_fn=None):
        """Print a labeled confusion matrix.

        *label_fn* translates a class code to a readable name.
        If None, codes are printed as-is.
        """
        if label_fn is None:
            label_fn = str

        names = [label_fn(label) for label in class_labels]
        col_width = max(len(name) for name in names) + 2
        matrix = evaluation["confusion_matrix"]

        self._write(" " * (col_width + 4) + "Predicted")
        self._write(" " * (col_width + 4)
                     + "".join(name.rjust(col_width) for name in names))

        for actual_label, actual_name in zip(class_labels, names):
            row_text = ("  " + actual_name).ljust(col_width + 4)
            for predicted_label in class_labels:
                row_text += str(matrix[actual_label][predicted_label]).rjust(col_width)
            self._write(row_text)

        self._write("")

    def class_metrics(self, evaluation, class_labels, label_fn=None):
        """Print per-class precision, recall, and F1."""
        if label_fn is None:
            label_fn = str

        self._write("  %-12s  %9s  %9s  %9s"
                     % ("Class", "Precision", "Recall", "F1"))
        self._write("  " + "-" * 43)

        for label in class_labels:
            metrics = evaluation["per_class"][label]
            self._write("  %-12s  %9.3f  %9.3f  %9.3f"
                         % (label_fn(label),
                            metrics["precision"], metrics["recall"], metrics["f1"]))
        self._write("")

    def tree(self, node, headers, value_labels=None, label_fn=None, indent=""):
        """Print a human-readable decision tree.

        *value_labels* is a dict of {feature_name: {code: readable_name}}.
        *label_fn* translates leaf class codes to readable names.
        """
        if label_fn is None:
            label_fn = str
        if value_labels is None:
            value_labels = {}

        text = self._format_tree(node, headers, value_labels, label_fn, indent)
        self._write(text)

    def _format_tree(self, node, headers, value_labels, label_fn, indent):
        """Recursively build a tree string."""
        if node.label is not None:
            return indent + "-> " + label_fn(node.label)

        feature = headers[node.feature] if headers else str(node.feature)
        value_map = value_labels.get(feature, {})

        output = indent + "[" + feature + "]\n"
        children = sorted(node.children)
        for i, code in enumerate(children):
            readable_value = value_map.get(code, code)
            child_node = node.children[code]
            output += indent + "  " + feature + " = " + readable_value + "\n"
            child_text = self._format_tree(
                child_node, headers, value_labels, label_fn, indent + "    "
            )
            output += child_text
            if i < len(children) - 1:
                output += "\n"

        return output
