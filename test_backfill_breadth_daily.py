from __future__ import annotations

import pandas as pd

import backfill_breadth_daily as backfill
import build_breadth_daily as build


def test_yfinance_symbol_maps_class_shares() -> None:
    assert backfill.yfinance_symbol("BRK.B") == "BRK-B"


def test_reconstruct_uses_point_in_time_membership_and_available_denominator() -> None:
    dates = pd.to_datetime(["1996-01-02", "1996-01-03"])
    ratios = pd.DataFrame(
        {"AAA": [1.1, 0.9], "BBB": [0.8, 1.2], "CCC": [1.2, 1.2]},
        index=dates,
    )
    membership = pd.DataFrame(
        {
            "date": pd.to_datetime(["1996-01-01", "1996-01-03"]),
            "tickers": ["AAA,BBB,MISSING", "AAA,CCC"],
        }
    )

    result = backfill.reconstruct_breadth(ratios, membership).set_index("Date")

    assert result.loc[dates[0], "breadth"] == 50.0
    assert result.loc[dates[0], "available_count"] == 2
    assert result.loc[dates[0], "constituent_count"] == 3
    assert result.loc[dates[1], "breadth"] == 50.0
    assert result.loc[dates[1], "membership_snapshot_date"] == pd.Timestamp("1996-01-03")


def test_splice_preserves_authoritative_rows() -> None:
    reference = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2002-01-02", "2002-01-03"]),
            "breadth": [65.91, 69.77],
            "source": ["MMTH-mapped", "MMTH-mapped"],
        }
    )
    reconstructed = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2002-01-01", "2002-01-02"]),
            "breadth": [60.123, 1.0],
        }
    )

    result = backfill.splice_backfill(reference, reconstructed)

    assert result["breadth"].tolist() == [60.12, 65.91, 69.77]
    assert result["source"].tolist() == [backfill.SOURCE_NAME, "MMTH-mapped", "MMTH-mapped"]


def test_regular_builder_preserves_validated_backfill(tmp_path, monkeypatch) -> None:
    output = tmp_path / "breadth_daily.csv"
    pd.DataFrame(
        {
            "Date": ["12/31/2001", "01/02/2002"],
            "breadth": [55.51, 65.91],
            "source": [backfill.SOURCE_NAME, "MMTH-mapped"],
        }
    ).to_csv(output, index=False)
    monkeypatch.setattr(build, "OUT_FILE", output)

    preserved = build.load_existing_backfill(pd.Timestamp("2002-01-02"))

    assert preserved.index.tolist() == [pd.Timestamp("2001-12-31")]
    assert preserved.iloc[0].to_dict() == {
        "breadth": 55.51,
        "source": backfill.SOURCE_NAME,
    }
