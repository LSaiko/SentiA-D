from src.metrics import edge_case_count, label_distribution, score


def test_label_distribution():
    labels = ["POSITIVE", "POSITIVE", "POSITIVE", "NEGATIVE"]
    assert label_distribution(labels) == {"POSITIVE": 3, "NEGATIVE": 1}


def test_edge_case_count():
    flags = [True, False, True, False, False]  # 2 edge / 5 total
    assert edge_case_count(flags) == 2


def test_perfect_score():
    m = score(["POSITIVE", "NEGATIVE"], ["POSITIVE", "NEGATIVE"])
    assert m["accuracy"] == 1.0 and m["f1"] == 1.0


def test_half_wrong():
    m = score(["POSITIVE", "NEGATIVE"], ["POSITIVE", "POSITIVE"])
    assert m["accuracy"] == 0.5
