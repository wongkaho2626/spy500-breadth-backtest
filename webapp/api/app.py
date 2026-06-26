"""
Flask backend for QQQ Portfolio Backtest webapp.
Wraps qqq_portfolio_backtest.py and exposes a JSON API.
"""
import json
import sys
import threading
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from flask import Flask, Response, render_template, request

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

import qqq_portfolio_backtest as _qbt  # noqa: E402


class _NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return None if (np.isnan(obj) or np.isinf(obj)) else float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, pd.Timestamp):
            return obj.strftime("%Y-%m-%d")
        return super().default(obj)


app = Flask(__name__)


def _json(data, status=200):
    return Response(json.dumps(data, cls=_NumpyEncoder), status=status, mimetype="application/json")

# ─── Data cache ───────────────────────────────────────────────────────────────
_data: tuple | None = None
_data_lock = threading.Lock()
_data_error: str | None = None


def _load_data() -> None:
    global _data, _data_error
    with _data_lock:
        if _data is not None:
            return
        try:
            print("[webapp] Loading market data…")
            result = _qbt.load_data()
            _data = result
            df = result[0]
            print(f"[webapp] Data ready: {df.index[0].date()} → {df.index[-1].date()}")
        except Exception as exc:
            _data_error = str(exc)
            print(f"[webapp] Data load failed: {exc}")


# Preload in background so first request isn't slow
threading.Thread(target=_load_data, daemon=True).start()

# Serialisation lock — run_strategy uses module-level globals for weights
_run_lock = threading.Lock()


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _series_to_list(s: pd.Series) -> list[dict]:
    return [
        {"date": d.strftime("%Y-%m-%d"), "value": round(float(v), 4)}
        for d, v in s.items()
        if not pd.isna(v)
    ]


def _annual_returns(strat: pd.Series, bench: pd.Series) -> list[dict]:
    rows = []
    for yr in sorted(set(strat.index.year)):
        sp = strat[strat.index.year == yr]
        bp = bench[bench.index.year == yr]
        if len(sp) < 2 or len(bp) < 2:
            continue
        rows.append({
            "year":      yr,
            "strategy":  round((sp.iloc[-1] / sp.iloc[0] - 1) * 100, 2),
            "benchmark": round((bp.iloc[-1] / bp.iloc[0] - 1) * 100, 2),
        })
    return rows


def _clean(v):
    """Make a value JSON-serialisable."""
    if isinstance(v, pd.Timestamp):
        return v.strftime("%Y-%m-%d")
    if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
        return None
    if isinstance(v, np.bool_):
        return bool(v)
    if isinstance(v, (np.integer, np.floating)):
        return float(v)
    return v


def _clean_trade(t: dict) -> dict:
    return {k: _clean(v) for k, v in t.items()}


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/info")
def info():
    with _data_lock:
        if _data_error:
            return _json({"ready": False, "error": _data_error})
        if _data is None:
            return _json({"ready": False})
        df = _data[0]
        return _json({
            "ready": True,
            "start": df.index[0].strftime("%Y-%m-%d"),
            "end":   df.index[-1].strftime("%Y-%m-%d"),
        })


@app.route("/api/backtest", methods=["POST"])
def run_backtest():
    try:
        p = request.json or {}

        # ── Weights
        raw = {
            "qqq":   float(p.get("qqq",   60)),
            "stock": float(p.get("stock", 30)),
            "tqqq":  float(p.get("tqqq",  10)),
            "spy":   float(p.get("spy",    0)),
            "soxx":  float(p.get("soxx",   0)),
        }
        total_w = sum(raw.values())
        if total_w <= 0:
            return _json({"error": "Weights must sum to a positive number."}), 400
        norm = {k: v / total_w for k, v in raw.items()}

        initial_capital      = float(p.get("initial_capital",      10_000))
        monthly_contribution = float(p.get("monthly_contribution",      0))
        yearly_contribution  = float(p.get("yearly_contribution",        0))
        cooldown_days        = int(p.get("cooldown_days",              30))
        start_date_str       = p.get("start_date") or None
        end_date_str         = p.get("end_date")   or None

        # ── Ensure data loaded
        _load_data()
        if _data_error:
            return _json({"error": f"Data load failed: {_data_error}"}), 500
        if _data is None:
            return _json({"error": "Data is still loading, please try again."}), 503

        with _run_lock:
            df, top_holdings, aligned_stocks, aligned_tqqq, aligned_spy, aligned_soxx = _data

            # Set module-level weight globals
            _qbt.QQQ_WEIGHT   = norm["qqq"]
            _qbt.STOCK_WEIGHT = norm["stock"]
            _qbt.TQQQ_WEIGHT  = norm["tqqq"]
            _qbt.SPY_WEIGHT   = norm["spy"]
            _qbt.SOXX_WEIGHT  = norm["soxx"]

            # Date filtering
            start_date = pd.Timestamp(start_date_str) if start_date_str else None
            end_date   = pd.Timestamp(end_date_str)   if end_date_str   else None
            if start_date is not None and start_date < df.index[0]:
                start_date = None
            if end_date is not None and end_date > df.index[-1]:
                end_date = None

            force_entry_on_start = False
            force_ticker: str | None = None
            if start_date is not None:
                df_pre = df[df.index < start_date]
                if not df_pre.empty:
                    force_entry_on_start, force_ticker = _qbt._position_at_date(
                        df_pre, top_holdings, cooldown_days=cooldown_days
                    )

            df_s = df.copy()
            if start_date is not None:
                df_s = df_s[df_s.index >= start_date]
            if end_date is not None:
                df_s = df_s[df_s.index <= end_date]

            def _slice(s):
                return s[s.index.isin(df_s.index)] if s is not None else None

            as_s   = {t: _slice(ser) for t, ser in aligned_stocks.items()}
            tqqq_s = _slice(aligned_tqqq)
            spy_s  = _slice(aligned_spy)
            soxx_s = _slice(aligned_soxx)

            strategy, trades, open_trade, total_contrib = _qbt.run_strategy(
                df_s, top_holdings, as_s, tqqq_s, spy_s, soxx_s,
                cooldown_days=cooldown_days,
                initial_capital=initial_capital,
                monthly_contribution=monthly_contribution,
                yearly_contribution=yearly_contribution,
                force_entry_on_start=force_entry_on_start,
                force_ticker=force_ticker,
            )
            benchmark = _qbt.run_benchmark(df_s, initial_capital=initial_capital)

        strat_m = _qbt.compute_metrics(strategy, trades)
        bench_m = _qbt.compute_metrics(benchmark)

        # Sell proximity
        sell_prox = None
        if open_trade:
            last  = df_s.iloc[-1]
            past  = df_s.iloc[max(0, len(df_s) - 1 - _qbt.DIVERGENCE_WINDOW)]
            pr    = (last["price"] - past["price"]) / past["price"] * 100
            bf    = past["breadth"] - last["breadth"]
            cap_ok = float(last["breadth"]) < _qbt.DIVERGENCE_BREADTH_CAP
            sell_prox = {
                "price_rise_pct":      round(pr, 2),
                "breadth_fall_pts":    round(bf, 2),
                "breadth_current":     round(float(last["breadth"]), 2),
                "price_rise_needed":   _qbt.DIVERGENCE_PRICE_RISE,
                "breadth_fall_needed": _qbt.DIVERGENCE_BREADTH_FALL,
                "breadth_cap":         _qbt.DIVERGENCE_BREADTH_CAP,
                "price_rise_met":      pr >= _qbt.DIVERGENCE_PRICE_RISE,
                "breadth_fall_met":    bf >= _qbt.DIVERGENCE_BREADTH_FALL,
                "breadth_cap_met":     cap_ok,
            }

        return _json({
            "success":       True,
            "metrics":       {"strategy": strat_m, "benchmark": bench_m},
            "chart_data": {
                "portfolio": _series_to_list(strategy),
                "benchmark": _series_to_list(benchmark),
                "breadth":   _series_to_list(df_s["breadth"]),
                "ndx":       _series_to_list(df_s["price"]),
            },
            "trades":         [_clean_trade(t) for t in trades],
            "open_trade":     _clean_trade(open_trade) if open_trade else None,
            "sell_proximity": sell_prox,
            "annual_returns": _annual_returns(strategy, benchmark),
            "total_contrib":  round(total_contrib, 2),
            "weights":        {k: round(v * 100, 1) for k, v in norm.items()},
            "params": {
                "buy_b200_thresh":         _qbt.BUY_B200_THRESH,
                "vix_buy_thresh":          _qbt.VIX_BUY_THRESH,
                "divergence_window":       _qbt.DIVERGENCE_WINDOW,
                "divergence_price_rise":   _qbt.DIVERGENCE_PRICE_RISE,
                "divergence_breadth_fall": _qbt.DIVERGENCE_BREADTH_FALL,
                "divergence_breadth_cap":  _qbt.DIVERGENCE_BREADTH_CAP,
            },
        })

    except Exception as exc:
        traceback.print_exc()
        return _json({"error": str(exc)}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5051, use_reloader=False)
