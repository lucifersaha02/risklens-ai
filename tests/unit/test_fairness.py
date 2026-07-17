"""Unit tests for responsible-AI subgroup diagnostics."""

import numpy as np
import pandas as pd

from risklens.fairness.evaluation import disparity_summary, subgroup_table


def diagnostic_frame() -> pd.DataFrame:
    """Return a small two-group diagnostic sample."""
    return pd.DataFrame(
        {
            "group": ["A"] * 4 + ["B"] * 4,
            "TARGET": [0, 0, 1, 1, 0, 0, 1, 1],
            "probability": [0.1, 0.4, 0.7, 0.9, 0.2, 0.6, 0.3, 0.8],
        }
    )


def test_subgroup_table_reports_operating_metrics() -> None:
    table = subgroup_table(diagnostic_frame(), "group", 0.5, minimum_group_size=2)
    group_a = table[table["group"] == "A"].iloc[0]
    assert group_a["rows"] == 4
    assert group_a["recall"] == 1.0
    assert group_a["false_positive_rate"] == 0.0
    assert group_a["meets_minimum_group_size"]


def test_small_groups_are_marked_not_deleted() -> None:
    table = subgroup_table(diagnostic_frame(), "group", 0.5, minimum_group_size=10)
    assert len(table) == 2
    assert not table["meets_minimum_group_size"].any()


def test_disparity_summary_reports_max_min_gap() -> None:
    table = subgroup_table(diagnostic_frame(), "group", 0.5, minimum_group_size=2)
    summary = disparity_summary(table)
    expected_gap = float(table["false_positive_rate"].max()) - float(
        table["false_positive_rate"].min()
    )
    assert summary["eligible_groups"] == 2
    assert summary["gaps"]["false_positive_rate_max_min_gap"] == expected_gap


def test_roc_auc_is_bounded_for_each_group() -> None:
    table = subgroup_table(diagnostic_frame(), "group", 0.5, minimum_group_size=2)
    assert np.logical_and(table["roc_auc"] >= 0, table["roc_auc"] <= 1).all()
