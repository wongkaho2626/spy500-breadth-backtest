import pandas as pd

import qqq_vector_factor_grid as grid


def test_feature_grid_contains_every_non_empty_subset() -> None:
    subsets = grid.feature_subsets()

    assert len(subsets) == 2 ** len(grid.FEATURES) - 1
    assert len(set(subsets)) == len(subsets)
    assert all(subset for subset in subsets)


def test_selection_prefers_precision_then_sharpe_then_fewer_features() -> None:
    baseline = {
        "early": {
            "cagr": 0.10,
            "max_drawdown": -0.30,
            "round_trips_per_year": 1.0,
        }
    }
    candidates = pd.DataFrame(
        [
            {
                "features": "a|b",
                "feature_count": 2,
                "threshold": 0.2,
                "early_cagr": 0.10,
                "early_sharpe": 1.0,
                "early_max_drawdown": -0.20,
                "early_round_trips_per_year": 1.0,
                "early_protected_crashes": 1,
                "early_true_vector_exits": 1,
                "early_precision": 0.5,
            },
            {
                "features": "a",
                "feature_count": 1,
                "threshold": 0.3,
                "early_cagr": 0.10,
                "early_sharpe": 0.9,
                "early_max_drawdown": -0.20,
                "early_round_trips_per_year": 1.0,
                "early_protected_crashes": 1,
                "early_true_vector_exits": 1,
                "early_precision": 1.0,
            },
        ]
    )

    selected = grid.select_configuration(candidates, baseline, "early")

    assert selected is not None
    assert selected["features"] == "a"


def test_opposite_validation_requires_crash_and_guardrails() -> None:
    baseline = {
        "late": {
            "cagr": 0.10,
            "max_drawdown": -0.30,
            "round_trips_per_year": 1.0,
        }
    }
    selected = pd.Series(
        {
            "late_protected_crashes": 1,
            "late_cagr": 0.09,
            "late_max_drawdown": -0.25,
            "late_precision": 0.5,
        }
    )

    result = grid._opposite_validation(selected, "early", baseline)

    assert result["passed"]


def test_consensus_leader_requires_protection_in_both_halves() -> None:
    baseline = {
        "early": {"cagr": 0.10, "max_drawdown": -0.30},
        "late": {"cagr": 0.10, "max_drawdown": -0.30},
    }
    candidates = pd.DataFrame(
        [
            {
                "features": "a",
                "feature_count": 1,
                "threshold": 0.2,
                "early_protected_crashes": 1,
                "late_protected_crashes": 0,
                "early_false_vector_exits": 0,
                "late_false_vector_exits": 0,
                "early_cagr": 0.10,
                "late_cagr": 0.10,
                "early_max_drawdown": -0.20,
                "late_max_drawdown": -0.20,
            },
            {
                "features": "b",
                "feature_count": 1,
                "threshold": 0.3,
                "early_protected_crashes": 1,
                "late_protected_crashes": 1,
                "early_false_vector_exits": 1,
                "late_false_vector_exits": 1,
                "early_cagr": 0.09,
                "late_cagr": 0.09,
                "early_max_drawdown": -0.25,
                "late_max_drawdown": -0.25,
            },
        ]
    )

    selected = grid.descriptive_consensus_leader(candidates, baseline)

    assert selected is not None
    assert selected["features"] == "b"
