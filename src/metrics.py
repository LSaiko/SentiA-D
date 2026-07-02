"""Classification metrics. Thin wrapper over sklearn — no reinvention."""
from collections import Counter

from sklearn.metrics import precision_recall_fscore_support, accuracy_score


def label_distribution(labels: list[str]) -> dict[str, int]:
    return dict(Counter(labels))


def edge_case_count(flags: list[bool]) -> int:
    return sum(flags)


def score(y_true: list[str], y_pred: list[str]) -> dict:
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": p,
        "recall": r,
        "f1": f1,
    }
