"""Run the tracked QQQ trajectory challenger with QQQ-style output.

This is a presentation/runner layer around the already-audited research engine.
It intentionally does not redefine any signal, execution, or cost rules.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import qqq_backtest as qbt
import qqq_breadth_only_trajectory_extension as strategy
import qqq_breadth_tqqq_ma200_satellite as proxy_source
import tqqq_backtest as tqbt


ROOT = Path(__file__).parent
INITIAL_CAPITAL = 10_000.0
TRAJECTORY_LOOKBACK = 20
BREADTH_WEIGHT = 0.70
TREND_WEIGHT = 0.30
TQQQ_BOOST_WEIGHT = 0.10

CHART_FILE = "qqq_trajectory_backtest_performance.png"
EQUITY_FILE = "qqq_trajectory_backtest_equity.csv"
TRADES_FILE = "qqq_trajectory_backtest_trades.csv"
DECISIONS_FILE = "qqq_trajectory_backtest_decisions.csv"


def run_backtest(df: pd.DataFrame, tqqq_proxy: pd.DataFrame) -> dict[str, Any]:
    """Run the fixed tracked configuration without changing the research engine."""
    return strategy.run_ensemble(
        df,
        tqqq_proxy,
        TRAJECTORY_LOOKBACK,
    )


def load_inputs(start_year: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    """Load all history first so indicators and the pre-inception proxy stay valid."""
    df = qbt.load_data()
    loaded_tqqq = tqbt.load_tqqq_data()[["price", "open"]]
    proxy, daily_drag = proxy_source.load_tqqq_proxy(df, 1.0, loaded=loaded_tqqq)

    if start_year is not None:
        mask = df.index.year >= start_year
        df = df.loc[mask]
        proxy = proxy.reindex(df.index)
    if df.empty:
        raise ValueError(f"no observations available for start year {start_year}")
    if proxy[["price", "open"]].isna().any().any():
        raise ValueError("TQQQ proxy does not cover every backtest session")
    return df, proxy, daily_drag


def _series_metrics(values: pd.Series) -> dict[str, float]:
    returns = values.pct_change().dropna()
    years = (values.index[-1] - values.index[0]).days / 365.25
    drawdown = values / values.cummax() - 1
    downside = returns[returns < 0]
    volatility = float(returns.std(ddof=1))
    cagr = float((values.iloc[-1] / values.iloc[0]) ** (1 / years) - 1)
    max_drawdown = float(drawdown.min())
    downside_deviation = float(np.sqrt(np.mean(np.square(downside))) * np.sqrt(252))
    monthly = values.resample("ME").last().pct_change().dropna()
    return {
        "total_return": float(values.iloc[-1] / values.iloc[0] - 1),
        "cagr": cagr,
        "max_drawdown": max_drawdown,
        "ulcer_index": float(np.sqrt(np.mean(np.square(drawdown)))),
        "sharpe": float(returns.mean() / volatility * np.sqrt(252)) if volatility else 0.0,
        "sortino": float(returns.mean() * 252 / downside_deviation) if downside_deviation else 0.0,
        "calmar": cagr / abs(max_drawdown) if max_drawdown else 0.0,
        "positive_months": float((monthly > 0).mean()),
    }


def _display_metrics(
    equity: pd.Series,
    exposure: pd.Series,
    independent_events: int | None = None,
) -> dict[str, str]:
    metrics = _series_metrics(equity)
    output = {
        "Total Return": f"{metrics['total_return']:.1%}",
        "CAGR": f"{metrics['cagr']:.1%}",
        "Max Drawdown": f"{metrics['max_drawdown']:.1%}",
        "Ulcer Index": f"{metrics['ulcer_index']:.2%}",
        "Sharpe Ratio": f"{metrics['sharpe']:.2f}",
        "Sortino Ratio": f"{metrics['sortino']:.2f}",
        "Calmar Ratio": f"{metrics['calmar']:.2f}",
        "Positive Months": f"{metrics['positive_months']:.1%}",
        "Exposure": f"{exposure.mean():.1%}",
        "Final Value": f"${equity.iloc[-1]:,.0f}",
    }
    if independent_events is not None:
        output["Independent Events"] = str(independent_events)
    return output


def print_configuration(df: pd.DataFrame, daily_drag: float) -> None:
    print(f"Date range  : {df.index[0].date()} → {df.index[-1].date()} ({len(df)} trading days)")
    print("Portfolio   : initially 70% frozen breadth timing + 30% NDX-above-MA200 trend sleeve")
    print("Washout buy : 10 percentage points of total portfolio use TQQQ; remainder uses NDX")
    print("Trajectory  : at session 60, extend TQQQ to session 80 when breadth > breadth 20 sessions ago")
    print("Core sells  : frozen bearish divergence OR climax top OR 25% trailing stop")
    print("Trend sells : NDX closes below MA200")
    print("Execution   : close signal → next-session OPEN fill")
    print(f"Costs       : ${qbt.COMMISSION:.0f} commission + {qbt.SLIPPAGE * 100:.2f}% slippage per leg")
    print(f"TQQQ proxy  : actual since {tqbt.TQQQ_INCEPTION}; synthetic before inception (calibrated drag {daily_drag * 252:.2%}/yr)")


def print_component_trades(run: dict[str, Any]) -> None:
    breadth_trades, breadth_open = run["breadth"][1], run["breadth"][2]
    trend_trades, trend_open = run["trend"][1], run["trend"][2]

    print("\n── 70% breadth sleeve trades ──")
    qbt.print_trades(breadth_trades, breadth_open)
    print("\n── 30% MA200 sleeve trades ──")
    qbt.print_trades(trend_trades, trend_open)


def print_extension_decisions(decisions: list[dict[str, Any]]) -> None:
    print("\n── TQQQ trajectory decisions ──")
    if not decisions:
        print("\nNo completed session-60 decisions.")
        return
    header = (
        f"\n{'#':>3}  {'Entry':10}  {'Decision':10}  {'Breadth':>8}  "
        f"{'20d ago':>8}  {'Change':>8}  {'Target':>7}  {'Rotation fill':13}"
    )
    print(header)
    print("-" * len(header))
    for number, decision in enumerate(decisions, 1):
        target = "80 days" if decision["extended_to_80"] else "60 days"
        rotation = decision.get("scheduled_rotation_date")
        rotation_text = rotation.strftime("%Y-%m-%d") if rotation is not None else "(after data)"
        print(
            f"{number:>3}  {decision['entry_date'].strftime('%Y-%m-%d'):10}  "
            f"{decision['decision_date'].strftime('%Y-%m-%d'):10}  "
            f"{decision['decision_breadth']:>8.2f}  "
            f"{decision['decision_past_breadth']:>8.2f}  "
            f"{decision['decision_breadth_change']:>+8.2f}  "
            f"{target:>7}  {rotation_text:13}"
        )


def print_current_state(df: pd.DataFrame, run: dict[str, Any]) -> None:
    last_date = df.index[-1]
    total = float(run["equity"].iloc[-1])
    breadth_value = float(run["breadth"][0].iloc[-1])
    trend_value = float(run["trend"][0].iloc[-1])
    breadth_open = run["breadth"][2]
    trend_position = bool(run["trend"][3].iloc[-1])
    above_ma200 = bool(df["price"].iloc[-1] > df["ma200"].iloc[-1])

    print(f"\n── Current portfolio state  (as of {last_date:%Y-%m-%d}) ──\n")
    print(f"  Total value       ${total:,.0f}")
    print(f"  Breadth sleeve    ${breadth_value:,.0f} ({breadth_value / total:.1%})  "
          f"{'IN' if breadth_open else 'CASH'}")
    print(f"  MA200 sleeve      ${trend_value:,.0f} ({trend_value / total:.1%})  "
          f"{'IN' if trend_position else 'CASH'}")

    desired_trend = "IN" if above_ma200 else "CASH"
    current_trend = "IN" if trend_position else "CASH"
    if desired_trend != current_trend:
        print(f"  MA200 next open   pending {'BUY' if above_ma200 else 'SELL'}")
    else:
        print(f"  MA200 next open   no change ({current_trend})")

    if breadth_open and breadth_open.get("tqqq_boosted"):
        boost_active = breadth_open.get("rotation_date") is None
        print(f"  TQQQ boost        {'ACTIVE' if boost_active else 'already rotated to NDX'}")
        decision_date = breadth_open.get("extension_decision_date")
        if decision_date is not None and decision_date > last_date:
            print(f"  Next trajectory   session-60 decision on {decision_date:%Y-%m-%d}")
        elif breadth_open.get("extended_to_80") is not None:
            target = 80 if breadth_open["extended_to_80"] else 60
            print(f"  Last trajectory   target {target} sessions")
    else:
        print("  TQQQ boost        inactive")


def save_artifacts(
    output_dir: Path,
    df: pd.DataFrame,
    benchmark: pd.Series,
    run: dict[str, Any],
) -> tuple[Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    equity_path = output_dir / EQUITY_FILE
    trades_path = output_dir / TRADES_FILE
    decisions_path = output_dir / DECISIONS_FILE

    pd.DataFrame(
        {
            "strategy": run["equity"],
            "buy_hold_ndx": benchmark,
            "breadth_sleeve": run["breadth"][0],
            "ma200_sleeve": run["trend"][0],
            "portfolio_in_market": run["position"].astype(int),
            "breadth_in_market": run["breadth"][3].astype(int),
            "ma200_in_market": run["trend"][3].astype(int),
        }
    ).rename_axis("Date").to_csv(equity_path)

    trade_frames = []
    for component, records in (("breadth", run["breadth"][1]), ("ma200", run["trend"][1])):
        frame = pd.DataFrame(records)
        if not frame.empty:
            frame.insert(0, "component", component)
            trade_frames.append(frame)
    if trade_frames:
        trades = pd.concat(trade_frames, ignore_index=True).sort_values("exit_date")
    else:
        trades = pd.DataFrame(columns=["component", "entry_date", "exit_date"])
    trades.to_csv(trades_path, index=False)
    pd.DataFrame(run["extension_decisions"]).to_csv(decisions_path, index=False)
    return equity_path, trades_path, decisions_path


def plot_results(
    output_dir: Path,
    df: pd.DataFrame,
    benchmark: pd.Series,
    run: dict[str, Any],
) -> Path:
    fig, axes = plt.subplots(
        3,
        1,
        figsize=(16, 12),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1.5, 1.2]},
    )
    ax_equity, ax_breadth, ax_price = axes
    fig.suptitle(
        "QQQ/NDX Trajectory Backtest\n"
        "Initially 70% breadth timing + 30% MA200 sleeve; 10% washout TQQQ boost; "
        "session-60 breadth trajectory chooses session 60/80 rotation",
        fontsize=10,
        fontweight="bold",
    )

    ax_equity.plot(benchmark.index, benchmark, label="Buy & Hold NDX", color="#2196F3", linewidth=1.4)
    ax_equity.plot(run["equity"].index, run["equity"], label="Trajectory strategy", color="#FF5722", linewidth=1.5)
    ax_equity.set_ylabel("Portfolio Value ($)")
    ax_equity.yaxis.set_major_formatter(plt.FuncFormatter(lambda value, _: f"${value:,.0f}"))
    ax_equity.legend(loc="upper left", fontsize=8)
    ax_equity.grid(True, alpha=0.3)

    ax_breadth.plot(df.index, df["breadth"], color="#7B1FA2", linewidth=1.0, label="S&P 500 % above MA200")
    ax_breadth.axhline(qbt.BUY_B200_THRESH, color="green", linestyle="--", linewidth=0.9, label=f"Washout < {qbt.BUY_B200_THRESH:.0f}%")
    ax_breadth.axhline(qbt.DIVERGENCE_BREADTH_CAP, color="red", linestyle="--", linewidth=0.9, label=f"Divergence cap < {qbt.DIVERGENCE_BREADTH_CAP:.0f}%")
    decisions = run["extension_decisions"]
    if decisions:
        extended_dates = [row["decision_date"] for row in decisions if row["extended_to_80"]]
        short_dates = [row["decision_date"] for row in decisions if not row["extended_to_80"]]
        if extended_dates:
            ax_breadth.scatter(extended_dates, df["breadth"].reindex(extended_dates), marker="^", color="green", s=65, zorder=5, label="Extend to 80")
        if short_dates:
            ax_breadth.scatter(short_dates, df["breadth"].reindex(short_dates), marker="v", color="orange", s=65, zorder=5, label="Rotate at 60")
    ax_breadth.set_ylabel("Breadth (%)")
    ax_breadth.legend(loc="upper left", fontsize=7)
    ax_breadth.grid(True, alpha=0.3)

    ax_price.plot(df.index, df["price"], color="#455A64", linewidth=1.0, label="NASDAQ-100")
    ax_price.plot(df.index, df["ma200"], color="orange", linewidth=0.9, linestyle="--", label="MA200")
    marker_specs = (
        (run["breadth"][1], run["breadth"][2], "green", "red", "Breadth"),
        (run["trend"][1], run["trend"][2], "#00ACC1", "#FB8C00", "MA200 sleeve"),
    )
    for records, open_trade, buy_color, sell_color, label in marker_specs:
        entries = [row["entry_date"] for row in records]
        if open_trade:
            entries.append(open_trade["entry_date"])
        exits = [row["exit_date"] for row in records]
        if entries:
            ax_price.scatter(entries, df["price"].reindex(entries), marker="^", color=buy_color, s=40, zorder=5, label=f"{label} buy")
        if exits:
            ax_price.scatter(exits, df["price"].reindex(exits), marker="v", color=sell_color, s=40, zorder=5, label=f"{label} sell")
    ax_price.set_ylabel("NDX")
    ax_price.set_xlabel("Date")
    ax_price.legend(loc="upper left", fontsize=7, ncol=2)
    ax_price.grid(True, alpha=0.3)
    ax_price.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax_price.xaxis.set_major_locator(mdates.YearLocator(2))
    fig.autofmt_xdate()

    chart_path = output_dir / CHART_FILE
    plt.tight_layout()
    plt.savefig(chart_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return chart_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tracked QQQ trajectory challenger backtest (QQQ-style report)"
    )
    parser.add_argument(
        "--start-year",
        type=int,
        default=None,
        metavar="YEAR",
        help="Reset the portfolio at the first observation in YEAR (default: full history)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT,
        help="Directory for chart and CSV outputs (default: script directory)",
    )
    parser.add_argument("--no-plot", action="store_true", help="Do not create the PNG chart")
    parser.add_argument("--no-save", action="store_true", help="Do not write the CSV artifacts")
    args = parser.parse_args()

    print("Loading data...")
    df, proxy, daily_drag = load_inputs(args.start_year)
    print_configuration(df, daily_drag)
    run = run_backtest(df, proxy)
    benchmark = qbt.run_benchmark(df)

    strategy_metrics = _display_metrics(
        run["equity"], run["position"], len(run["clustered_trades"])
    )
    benchmark_metrics = _display_metrics(
        benchmark, pd.Series(True, index=benchmark.index)
    )
    qbt.print_metrics(strategy_metrics, benchmark_metrics)
    print_component_trades(run)
    print_extension_decisions(run["extension_decisions"])
    print_current_state(df, run)
    qbt.print_sell_proximity(df, run["breadth"][2])

    if not args.no_save:
        paths = save_artifacts(args.output_dir, df, benchmark, run)
        print("\nCSV files saved:")
        for path in paths:
            print(f"  {path}")
    if not args.no_plot:
        chart_path = plot_results(args.output_dir, df, benchmark, run)
        print(f"\nChart saved → {chart_path}")

    print("\nResearch challenger only; frozen qqq_backtest.py signals are unchanged.")


if __name__ == "__main__":
    main()
