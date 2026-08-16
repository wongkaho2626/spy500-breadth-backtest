# QQQ Backtest Verification Report — SPY constituent MA200 backfill

Run date: 2026-08-17  
Data window: 1996-01-02 to 2026-08-14, 7,702 sessions  
Execution: close signal, next-session NDX open fill  
Breadth sources: 1,511 `SPY-constituents-MA200` rows, 1,256 `MMTH-mapped`
rows, and 4,935 actual `S5TH` rows.

## Backtest Score: 20 / 100 — Reject

The raw evidence score is 72, but the requested full-history run has unresolved
survivorship/missing-history bias in its 1996-2001 breadth input. The installed
scoring rubric therefore caps the result at 20. The 25 completed trades also
trigger the separate 40-point small-sample cap, and no genuinely untouched OOS
period exists after the 2026-07-05 parameter freeze.

| Component | Score | Max |
|---|---:|---:|
| A. Statistical Validity & Significance | 27 | 30 |
| B. Risk-Adjusted Performance | 13 | 25 |
| C. Robustness & Out-of-Sample | 16 | 25 |
| D. Trade Quality & Consistency | 16 | 20 |
| **Raw total** | **72** | **100** |
| Caps applied | survivorship/missing-history 20; fewer than 30 trades 40; no true OOS 55 | |
| **Final score** | **20** | **100** |

## Executive Summary

The expanded run produces a 17.94% CAGR, 1.00 Sharpe, and -35.24% maximum
drawdown versus 13.72%, 0.61, and -82.90% for NDX price buy-and-hold. Absolute
strategy returns are statistically different from zero, parameter perturbations
remain profitable, and 10x stated costs leave a 16.98% CAGR.

The result is not yet reliable as a tradeable QQQ estimate. Early reconstructed
breadth has only 37.37% minimum and 44.83% median constituent coverage; the code
uses an untradeable NDX price index rather than QQQ total returns; and just 25
trades, four of them losses, drive the trade-quality statistics. The 2007+
actual-S5TH segment is the cleanest evidence and remains materially stronger
than the reconstructed early period.

## 1. Performance Metrics

| Metric | Strategy | NDX price benchmark | Assessment |
|---|---:|---:|---|
| Total return | 15,536.26% | 5,027.85% | Strategy higher |
| CAGR | 17.94% | 13.72% | +4.22 percentage points |
| Annual volatility | 18.13% | 27.50% | Lower |
| Sharpe | 1.002 | 0.606 | Good, not exceptional |
| Sortino | 1.515 | 0.877 | Good |
| Calmar | 0.509 | 0.166 | Meets minimum threshold |
| Maximum drawdown | -35.24% | -82.90% | Material strategy risk remains |
| Ulcer Index | 8.37% | 41.35% | Strong improvement |
| Lake ratio | 80.08% | 92.57% | Strategy is underwater most sessions |
| 95% daily VaR / CVaR | 1.73% / 2.78% | 2.78% / 4.04% | Lower tail risk |
| 99% daily VaR / CVaR | 3.44% / 4.38% | 4.67% / 6.18% | Lower tail risk |
| Completed / open trades | 25 / 1 | — | Below 30-trade minimum |
| Session time in market | 62.40% | 100% | CLI's 58% excludes the open trade |

### Breadth-source stability

Each row below restarts the strategy in cash at the beginning of the source
segment, so it is an isolated stability comparison rather than a slice of the
continuous portfolio.

| Breadth segment | Trades | CAGR | Sharpe | Max DD | Benchmark CAGR |
|---|---:|---:|---:|---:|---:|
| SPY reconstruction, 1996-2001 | 7 | 12.30% | 0.810 | -35.24% | 17.95% |
| MMTH mapped proxy, 2002-2006 | 4 | 11.18% | 0.773 | -17.42% | 1.76% |
| Actual S5TH, 2007+ | 12 | 22.19% | 1.165 | -32.19% | 15.57% |
| Existing combined series, 2002+ | 17 | 20.47% | 1.116 | -32.18% | 12.62% |

## 2. Statistical Significance

| Test | Result | Interpretation |
|---|---:|---|
| Daily mean-return t-stat | 5.54, p=3.1e-8 | Absolute return differs from zero |
| PSR versus zero | 99.999999% | Strong absolute Sharpe evidence |
| DSR, assumed 1,000 trials | 98.94% | Passes the stated multiple-test assumption |
| DSR benchmark Sharpe | 0.589 | Observed Sharpe 1.002 exceeds it |
| Paired excess vs NDX | +1.51%/year, t=0.405, p=0.686 | No significant raw excess mean |
| HAC market regression | alpha 10.94%/year, p=2.3e-6; beta 0.435 | Significant low-beta risk-adjusted alpha |
| ADF on daily returns | -15.48, p=2.6e-28 | Returns are stationary |
| Jarque-Bera | 39,074.9, p≈0 | Returns are strongly non-normal |
| Ljung-Box, 20 lags | p=1.6e-11 | Serial dependence is present |

Return skewness is +0.47 and excess kurtosis is 11.00. Normal-theory inference
must therefore be read alongside the block-bootstrap results. The apparent
conflict between insignificant raw excess and significant HAC alpha is expected:
the alpha regression adjusts for the strategy's low 0.435 market beta, whereas
the paired test asks only whether the unadjusted average daily return difference
is positive.

## 3. Bias Assessment

| Bias | Verdict | Evidence |
|---|---|---|
| Lookahead | Absent | Close-derived signals fill at the following session's open. |
| Survivorship / missing histories | **Present** | 1996-2001 breadth drops constituents without a valid Yahoo history; minimum coverage is 37.37%. Sparse S5TH validation (r=0.947, MAE 4.74 points) mitigates but does not remove daily threshold error. |
| Overfitting / data snooping | **Present** | At least 11 principal settings plus MACD spans were developed against data through 2026; the baseline is frequently a local full-sample sensitivity optimum. |
| Transaction-cost underestimate | Absent for stated turnover | 10x commission/slippage retains 16.98% CAGR and 0.958 Sharpe. |
| Liquidity / fill realism | Cannot verify | The engine fills at the NDX index open, not an executable QQQ quote, and has no ADV model. |
| Data-frequency mismatch | Absent | Daily close signals and next-session fills are aligned. |
| Feature look-forward | Absent | Rolling features use current and prior observations only. |
| Regime concentration | Partly present | Crisis protection is strong, but only 25 completed trades and a few multi-year winners dominate the record. |
| Instrument / total-return mismatch | **Present** | `NASDAQ100.csv` is a price index proxy; QQQ dividends, fund expenses, tracking difference, and cash interest are omitted. This can overstate relative results because buy-and-hold misses more dividend exposure. |

## 4. Robustness

### Pseudo walk-forward

Six main breadth parameters were selected from 729 combinations per expanding
fold while the remaining strategy rules stayed fixed. These are historical
holdouts, not genuinely untouched OOS observations.

| IS → OOS | IS Sharpe | OOS Sharpe | Efficiency | OOS trades | Current-rule OOS Sharpe |
|---|---:|---:|---:|---:|---:|
| 1996-2010 → 2011-2017 | 0.832 | 0.809 | 0.972 | 2 | 1.099 |
| 1996-2017 → 2018-2026 | 0.931 | 1.445 | 1.551 | 6 | 1.509 |

Efficiency is strong, but two and six OOS trades are too few for high confidence.

### Bootstrap and sensitivity

- Twenty-session block bootstrap, 1,000 runs: Sharpe 95% interval
  0.681-1.318; CAGR 95% interval 11.23%-25.12%; MDD 5th/median/95th
  percentiles -45.55%/-31.39%/-23.14%.
- Completed-trade bootstrap, 1,000 runs: terminal multiple 5th/median/95th
  percentiles 10.94x/85.20x/856.96x; MDD percentiles
  -43.94%/-24.07%/-10.81%. The huge terminal range illustrates the thin trade
  sample rather than precision.
- One-at-a-time ±20%/±50% sensitivity remains profitable throughout. Across
  all tested settings Sharpe ranges from 0.677 to 1.011, while the worst MDD is
  -53.85%. The surface is not cliff-edge, although many baseline settings sit
  near the full-sample optimum.
- Cost stress at 1x/2x/5x/10x produces CAGR
  17.94%/17.83%/17.51%/16.98% and Sharpe 1.002/0.998/0.983/0.958.

### Crisis behaviour

| Regime | Strategy return / MDD | Benchmark return / MDD |
|---|---:|---:|
| Dot-com bust | -10.17% / -35.24% | -82.40% / -82.90% |
| Global financial crisis | +0.84% / -32.18% | -51.92% / -53.71% |
| COVID crash | -14.72% / -14.87% | -27.90% / -28.03% |
| 2022 bear market | -6.18% / -21.77% | -35.21% / -35.21% |

## 5. Red Flags

1. The full 1996 result uses a structurally incomplete breadth denominator;
   threshold crossings can move when unavailable delisted stocks are restored.
2. `qqq_backtest.py` is an NDX price-index proxy, not a QQQ total-return and
   executable-price simulation.
3. Twenty-five completed trades are too few for stable profit-factor, Kelly,
   and win-rate estimates. PF 12.35, 84% wins, and 77% full Kelly should not be
   used directly for sizing.
4. There is no meaningful post-freeze sample after 2026-07-05. Historical
   walk-forward partitions were visible during strategy development.
5. Only 45.65% of calendar months are positive; the 252-session rolling Sharpe
   ranges from -2.35 to +3.22 with 0.88 standard deviation.

## 6. Improvement Recommendations

1. Make 2007+ actual S5TH the primary evidence window and label 1996-2006 as
   auxiliary/proxy analysis until institutional delisted-price histories are
   available.
2. Replace NDX price opens with adjusted QQQ closes for marks, unadjusted QQQ
   opens for fills, explicit distributions, expense ratio, and a T-bill cash
   return. Re-run the entire audit without retuning.
3. Keep the 2026-07-05 rules frozen and accumulate at least 30 independent
   completed trades or a multi-year prospective record before treating PF,
   Kelly, or the 84% win rate as stable.
4. Pre-register one narrow robustness test for the 26% washout threshold and
   do not select a replacement using the same 1996-2026 sample.
5. Correct the CLI time-in-market metric to include the current open position;
   actual session exposure is 62.40%, not the printed 58.0%.

## 7. Verdict

The full-history backtest receives **20/100 — Reject** under the required hard
gates. This does not mean the breadth strategy has no useful signal: the 2007+
actual-S5TH segment has 22.19% CAGR and 1.165 Sharpe, robustness tests are
directionally favourable, and crisis drawdowns are much better than NDX.
It means the new 1996 extension cannot yet increase confidence in a tradeable
edge because its missing-history bias and thin independent trade count dominate
the evidential quality.
