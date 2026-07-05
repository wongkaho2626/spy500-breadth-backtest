# Frozen parameters — 2026-07-05

Snapshot of the canonical `qqq_backtest.py` configuration, frozen for genuine
out-of-sample evaluation. Everything before this date is in-sample (parameters
were tuned, and the trend re-entry rule was designed, using data through
2026-07-02). **Performance from this date forward is the only clean OOS
evidence.** Do not re-tune these values against windows that disappoint;
evaluate the frozen configuration as-is.

## Strategy constants (qqq_backtest.py)

| Constant | Frozen value |
|---|---|
| `BUY_B200_THRESH` | 26.0 |
| `VIX_BUY_THRESH` | 30.0 |
| `MA200_WINDOW` | 200 |
| `DIVERGENCE_WINDOW` | 60 |
| `DIVERGENCE_PRICE_RISE` | 3.0 |
| `DIVERGENCE_BREADTH_FALL` | 20.0 |
| `DIVERGENCE_BREADTH_CAP` | 60.0 |
| `EXT10_PCT` | 5.0 |
| `CLIMAX_VOTE_WINDOW` | 10 |
| `TRAILING_STOP_PCT` | 25.0 |
| `COMMISSION` / `SLIPPAGE` | $1 / 0.05% per side |
| `COOLDOWN_DAYS` | 15 |
| `EXECUTION_LAG` / `FILL_PRICE` | 1 / `open` (next-day open) |

Rule set: washout entry (breadth < 26 + VIX/MA200 vote) OR trend re-entry
(fresh MA200 recross after a climax-top exit or above the prior exit price);
exits on bearish divergence, climax top, or 25% trailing stop.

## State at freeze (data through 2026-07-02)

- Position: **IN** — entered 2025-04-07 at NDX 16,771.77 (VIX trigger),
  +76.0% as of 2026-07-02 (NDX 29,546.74).
- In-sample record, 2002-01-02 → 2026-07-02: CAGR 20.5%, Sharpe 1.12,
  MDD −32.2%, 17 closed trades. Real-breadth era only (2007+): CAGR 22.2%,
  Sharpe 1.17.

## Validation status at freeze (see session audits, 2026-07-05)

- Significance: daily t = 5.5, PSR ≈ 1.00, DSR ≥ 0.89 under punitive
  multiple-testing assumptions. Excess return vs B&H only borderline
  (t ≈ 1.9) — the edge is risk reduction.
- Cross-asset check of the trend re-entry gate (ON vs OFF, same rules):
  ΔCAGR +1.4 (NDX), +2.8 (SPX), +0.3 (SOXX), +4.2 pts (Russell 3000) —
  generalizes across indices, weakest on the volatile sector ETF.
- Known caveats: NDX price index (no dividends; T-bill yield on cash not
  modelled — adding both raises 2007+ CAGR ≈ +1.4 pts, Sharpe 1.21);
  pre-2007 breadth is synthetic; expect a drawdown worse than −32% eventually
  (block-bootstrap 5th percentile ≈ −39%).

## How to evaluate OOS later

Run `python qqq_backtest.py --start-year 2026` (or slice results from
2026-07-05) and compare against this document. The strategy earns continued
trust if its forward Sharpe stays above ~0.6–0.7 (half the in-sample 1.12,
the usual live-decay allowance) over a meaningful stretch (3+ years or ≥5
completed trades).
