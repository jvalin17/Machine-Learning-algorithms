"""Data loading, counting, and partitioning. No imports."""


def load_csv(filepath):
    """Read a CSV file. Returns (headers, rows).

    headers: list of column name strings.
    rows:    list of lists, one per data row.
    """
    with open(filepath, "r") as file:
        raw = file.read()

    lines = raw.strip().split("\n")
    headers = lines[0].split(",")
    rows = [line.split(",") for line in lines[1:] if line.strip()]
    return headers, rows


def count_values(items):
    """Return {value: count} for an iterable of hashable items."""
    counts = {}
    for item in items:
        counts[item] = counts.get(item, 0) + 1
    return counts


def majority_class(rows, label_col):
    """Return the most frequent label in *rows*."""
    counts = count_values(row[label_col] for row in rows)
    best_label = None
    best_count = -1
    for label, count in counts.items():
        if count > best_count:
            best_count = count
            best_label = label
    return best_label


def split_by_feature(rows, feature_col):
    """Partition rows into {value: [matching_rows]} by a feature column."""
    partitions = {}
    for row in rows:
        key = row[feature_col]
        if key not in partitions:
            partitions[key] = []
        partitions[key].append(row)
    return partitions
