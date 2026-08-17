# QQQ 70 / NDX Top-1 30 DCA Rolling Backtest Verification

Analysis date: 2026-08-17  
Backtest data: 1996-01-02 to 2026-08-14, 7,702 aligned trading sessions  
Primary inputs: `qqq70stock30_dca_rolling.csv`,
`qqq70stock30_dca_nonoverlap.csv`, and
`qqq70stock30_dca_overlap_diagnostics.csv`

## Backtest Score: 20 / 100 — Reject

The displayed returns are strong, but the evidence does not yet justify a
tradeable-edge conclusion. More than 99.5% of daily-step rolling windows overlap,
no horizon survives multiple-testing correction on the canonical non-overlapping
sample, the strategy has no frozen out-of-sample segment, and the 1996–2001 breadth
reconstruction has unresolved missing-history/survivorship risk.

The exact strategy does not export a daily equity series or trade log. Missing
sub-metrics are therefore excluded rather than scored as failures; the available
score is normalised from a reduced 40-point denominator.

| Component | Awarded | Measured maximum | Original maximum | Reason |
|---|---:|---:|---:|---|
| A. Statistical validity | 4 | 15 | 30 | 1y excess t-stat fails; 30 independent annual windows are borderline |
| B. Risk-adjusted performance | N/A | 0 | 25 | Exact daily return/equity series unavailable |
| C. Robustness and OOS | 4 | 18 | 25 | Bootstrap evidence mixed; no WFA/OOS; sensitivity unavailable |
| D. Trade quality and consistency | 5 | 7 | 20 | 80% of non-overlapping 1y strategy windows positive; no exact trade log |
| **Reduced raw score** | **13** | **40** |  | **32.5 / 100 after normalisation** |
| Caps applied |  |  |  | Unresolved survivorship/missing-history bias: 20; no OOS/WFA: 55 |
| **Final score** |  |  | **20 / 100** | Lowest applicable cap |

## Executive Summary

The overlap correction materially weakens the headline evidence. The 1-year mean
excess return is only +6.12 percentage points on 30 non-overlapping observations;
its t-test, Newey–West interval, and bootstrap interval all fail to exclude zero.
The 3-year result is nominally positive, but it does not survive Benjamini–Hochberg
false-discovery correction across the tested horizons.

Long horizons are descriptive only. There are six independent 5-year windows,
three 10-year windows, and one 20-year window. Thousands of overlapping starts do
not turn these into thousands of independent observations.

## 1. Data Intake and Performance

| Item | Assessment |
|---|---|
| Input type | Rolling ending values/returns, canonical non-overlapping outcomes, overlap diagnostics |
| Strategy | Breadth/trend-driven long/cash timing; 70% NDX price proxy and 30% annual NDX Top-1 stock; annual DCA |
| Frequency | Daily signals and next-session open execution |
| Period | 1996-01-02 to 2026-08-14 |
| Parameters | At least 11 signal/risk parameters, plus portfolio weights and contribution schedule |
| OOS/WFA | None identified |

The reported multi-year return is return on total deployed capital after multiple
cash contributions. It is not a time-weighted CAGR and must not be labelled CAGR.

| Horizon | Daily-step windows | Strategy average return | NDX buy-and-hold | Strategy losing windows |
|---:|---:|---:|---:|---:|
| 1y | 7,451 | 22.6% | 17.3% | 14.7% |
| 3y | 6,947 | 70.1% | 47.4% | 1.7% |
| 5y | 6,443 | 119.6% | 59.2% | 0.0% |
| 10y | 5,183 | 316.4% | 147.2% | 0.0% |
| 20y | 2,663 | 1,692.0% | 468.0% | 0.0% |

CAGR, annualised volatility, Sharpe, Sortino, Calmar, Omega, VaR/CVaR,
drawdown duration, Ulcer Index, and rolling Sharpe cannot be calculated reliably
from ending-value summaries. A cash-flow-adjusted daily equity series is required.

## 2. Statistical Significance

The primary test uses the canonical non-overlapping excess returns. The FDR column
corrects the ten horizons with at least three observations using
Benjamini–Hochberg. None remains significant at 5%.

| Horizon | Independent n | Mean excess | t-stat | Raw p | FDR p | Non-overlap bootstrap 95% CI |
|---:|---:|---:|---:|---:|---:|---:|
| 1y | 30 | +6.12 pp | 1.19 | 0.242 | 0.346 | -4.01 to +15.75 pp |
| 3y | 10 | +33.11 pp | 2.90 | 0.018 | 0.172 | +11.75 to +55.05 pp |
| 5y | 6 | +56.38 pp | 1.65 | 0.160 | 0.280 | -5.51 to +116.95 pp |
| 10y | 3 | +265.02 pp | 2.12 | 0.168 | 0.280 | +114.96 to +513.11 pp |
| 20y | 1 | +743.07 pp | N/A | N/A | N/A | N/A |

The 1-year excess distribution is non-normal (skew -1.13, excess kurtosis 5.34,
Jarque–Bera p < 0.0001), reinforcing the bootstrap result over a naive Gaussian
t-test. Its overlapping-series Newey–West estimate is +5.26 pp with a 95% interval
of -1.71 to +12.23 pp (p = 0.139).

PSR is unavailable because the exact daily return series is absent. DSR is also
unavailable because neither an exact daily Sharpe nor the complete number and
dependence structure of all tried specifications is recorded. The repository does
contain a 21-allocation full-sample grid search and other sweeps, so treating the
70/30 choice as pre-specified would be inappropriate.

## 3. Bias Assessment

| Bias | Status | Evidence |
|---|---|---|
| Core signal lookahead | Absent | Signal index is lagged one bar and orders fill at the next open |
| Top-holding availability lookahead | Cannot verify | NDX holdings have only a `Year`; the SPY fallback uses the latest report dated before each calendar year |
| Survivorship/missing-history | Present | 1996–2001 reconstructed breadth median coverage is 44.83%; unavailable historical constituents are excluded from the denominator |
| Data snooping/overfitting | Present | 70/30 sits inside a 0–100% allocation grid and other same-sample sweeps; no DSR or OOS confirmation |
| Transaction-cost underestimate | Cannot verify | $1 commission and 5 bps slippage are modelled, but no 2×/5×/10× stress, spread, market impact, or capacity test |
| Liquidity bias | Cannot verify | No order-size-versus-ADV test |
| Frequency mismatch | Absent | Daily data are aligned; sparse pre-2007 breadth was replaced by a daily composite |
| Look-forward features | Absent in inspected signal code | Rolling indicators use current/past values and execution is lagged |
| Regime overfitting | Cannot verify | No isolated exact-strategy crisis/low-volatility report |

The earlier strategy-definition inconsistency is resolved in the latest run.
NASDAQ-100 Top-1 holdings begin in 2001; for 1996–2000 the rolling engine now uses
the SPY Top-1 known from the latest holdings snapshot preceding the calendar year
(GE for 1996–1998 and MSFT for 1999–2000), rather than leaving 30% in cash.

The benchmark and QQQ leg use the NDX price index, while stock CSVs were downloaded
with `auto_adjust=True`. This mixes a price-only index with dividend-adjusted stock
returns and is not an executable QQQ-versus-stock comparison.

## 4. Robustness

| Test | Result |
|---|---|
| Overlap correction | Completed: canonical non-overlap cohorts plus Newey–West diagnostics |
| Bootstrap | Mixed; 1y and 5y intervals include zero; long-horizon samples are too small |
| Multiple-testing correction | No tested horizon significant at 5% FDR |
| Walk-forward/OOS | Not performed |
| Monte Carlo trade sequencing/MDD | Not available without exact trade log/equity path |
| Parameter sensitivity | Not performed for signal thresholds |
| Cost stress | Not performed |
| Regime stability | Not performed for the exact DCA strategy |

## 5. Red Flags

1. Early breadth uses only 37.37%–44.83% typical constituent coverage and excludes
   missing histories from the denominator.
2. Annual NDX top holdings lack point-in-time effective dates.
3. There is no frozen OOS or walk-forward test after parameter/allocation searches.
4. Multi-year DCA ROI is easy to misread as CAGR.
5. NDX price-index returns and adjusted stock returns are inconsistent.

## 6. Improvement Recommendations

1. Attach an as-of/publication date to every annual NDX holding and only permit its use
   after that date.
2. Export the exact strategy's daily equity, external cash flows, positions, and
   trades. Compute time-weighted return and XIRR separately.
3. Freeze the complete specification on a pre-2017 development sample and reserve
   2017–2026 for purged OOS/walk-forward validation.
4. Run threshold sensitivity and 1×/2×/5×/10× cost stress tests. Apply DSR using the
   full recorded trial count.
5. Use actual adjusted QQQ prices for both strategy and benchmark, with consistent
   dividend, fee, and corporate-action treatment.
6. Either obtain institution-grade point-in-time constituent histories or report a
   clean 2007+ S5TH-only result beside the reconstructed full-history result.

## 7. Verdict

**Backtest Score: 20 / 100 — Reject.** The strategy may contain an economically
interesting signal, but the current evidence is exploratory. The next valid
decision point is after completing a frozen, purged OOS run with an exact daily
equity/trade record.
