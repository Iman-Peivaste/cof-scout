from cof_scout.analysis.statistics import tie_adjusted_superiority


def test_tie_adjusted_superiority():
    assert tie_adjusted_superiority([2], [1, 2, 3]) == 0.5
