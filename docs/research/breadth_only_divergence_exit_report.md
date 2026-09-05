# Backtest Verification Report — breadth-only 60-session decline exit

## Verdict: Reject

## Executive Summary

Removing the NDX price-rise vote produces a challenger score of **52 / 100 (Needs work)** versus baseline **40 / 100 (Weak)**.  The decision follows the pre-registered Calmar objective and guardrails.

## Backtest Scores

| Component | Baseline | Challenger | Max |
|---|---:|---:|---:|
| A. Statistical validity | 23 | 20 | 30 |
| B. Risk-adjusted performance | 11 | 4 | 25 |
| C. Robustness / OOS | 25 | 15 | 25 |
| D. Trade quality / consistency | 16 | 13 | 20 |
| **Raw total** | **75** | **52** | **100** |
| Hard cap | 40 | 100 | |
| **Final score** | **40** | **52** | **100** |

## Performance and risk

| Metric | Baseline | Challenger | Delta |
|---|---:|---:|---:|
| CAGR | 20.26% | 7.54% | -12.72% |
| Volatility | 18.20% | 12.68% | -5.52% |
| Sharpe | 1.107 | 0.638 | -0.469 |
| Sortino | 0.941 | 0.370 | -0.571 |
| Calmar | 0.630 | 0.218 | -0.411 |
| Maximum drawdown | -32.18% | -34.54% | -2.35% |
| Ulcer Index | 5.80% | 8.16% | +2.36% |
| Time underwater | 85.23% | 91.63% | +6.40% |
| Completed trades | 17 | 46 | +29 |
| Exposure | 73.19% | 34.13% | -39.06% |
| Win rate | 94.12% | 67.39% | -26.73% |
| Payoff ratio | 3.14 | 1.67 | |
| Profit factor | 50.21 | 3.45 | |
| Expectancy | 31.29% | 3.94% | -27.34% |

## Statistical significance

- Effective observations: 5458 / 5146.
- t-stat: 5.151 / 2.883; PSR: 1.0000 / 0.9980.
- DSR after 4,604 trials: 0.9336 / 0.2160.
- Jarque-Bera p: 0 / 0; Ljung-Box p: 3.09e-13 / 1.88e-13.
- Paired annual mean: -12.06%; HAC t=-5.143, p=0.000.
- Block-bootstrap 95% interval: [-0.16962369567686272, -0.07542388438471777].

## Robustness

- Historical-half efficiency: 0.763 / 0.642 (pseudo-OOS).
- Sensitivity Calmar: fall_15=0.259, fall_20=0.218, fall_25=0.180.
- Odd/even paired means: -10.93% / -13.13%.
- Trade bootstrap simulations: 5,000.
- 5x/10x-cost paired means: -12.59% / -13.26%.

## Signal inventory

- Raw active days: 824.
- Raw episode onsets: 94.
- Raw days newly admitted by removing price rise: 761.
- Executed breadth-only sells: 40.
- Executed sells newly admitted by removing price rise: 38.

## Bias assessment

| Bias | Status | Evidence |
|---|---|---|
| Lookahead / frequency mismatch | Absent | Close signal; next-session-open fill |
| Survivorship | Cannot fully verify | Aggregate index and breadth series |
| Data snooping | Present, material | 4,604 trials penalised |
| Costs | Included | 1x/2x/5x/10x commission and slippage |
| Liquidity | Low concern | Liquid index proxy; NDX price-index approximation remains |
| Synthetic breadth | Present before 2007 | 2007+ reported separately |
| Clean forward OOS | Insufficient | Too few post-freeze observations/trades |
| Regime overfit | Tested | Halves, odd/even, real-breadth, sensitivity |

## Guardrails

- baseline_parity: PASS
- calmar_improved: FAIL
- max_drawdown_not_worse: FAIL
- cagr_within_two_points: FAIL
- expectancy_not_worse: FAIL
- turnover_guardrail: FAIL
- historical_halves_calmar_nonnegative: FAIL
- real_breadth_calmar_nonnegative: FAIL
- five_x_paired_return_positive: FAIL
- sensitivity_not_cliff_edge: FAIL

## Current signal

As of 2026-09-01, breadth is 62.60% versus 58.05% 60 sessions earlier, a fall of -4.55 points.  The breadth-only signal is inactive.

## Red Flags

1. Removing a confirmation vote can substantially increase exits and whipsaw.
2. Pre-2007 breadth is synthetic and all pre-freeze comparisons are historical robustness evidence.
3. A higher final score caused only by clearing the 30-trade cap is not an economic improvement.

## Improvement Recommendations

1. Follow the pre-registered decision; do not modify the frozen baseline on historical evidence alone.
2. Use the raw-event CSV to inspect every qualifying date and the trade CSV for actual fills.

## Decision

The verdict follows the primary Calmar objective and every guardrail without retuning.

Research evidence only; not investment advice.
