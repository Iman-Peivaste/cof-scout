from pathlib import Path

import pandas as pd

from cof_scout.figures.paper import endpoint_summary


def test_endpoint_summary_matches_archived_reference():
    root = Path(__file__).resolve().parents[1] / "data" / "small"
    source = pd.read_csv(root / "figure_2_endpoint_source.csv")
    expected = pd.read_csv(root / "figure_2_endpoint_summary.csv")
    actual = endpoint_summary(source)
    pd.testing.assert_frame_equal(actual, expected, check_dtype=False)
