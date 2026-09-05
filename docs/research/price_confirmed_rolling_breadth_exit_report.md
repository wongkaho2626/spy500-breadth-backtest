# Backtest Verification Report — price-confirmed rolling breadth exit

## Verdict: Reject

## Executive Summary

The challenger scores **40 / 100 (Weak)** versus baseline **40 / 100 (Weak)**.  The decision follows the pre-registered Calmar objective and guardrails.

## Backtest Scores

| Component | Baseline | Challenger | Max |
|---|---:|---:|---:|
| A. Statistical validity | 23 | 23 | 30 |
| B. Risk-adjusted performance | 11 | 8 | 25 |
| C. Robustness / OOS | 25 | 18 | 25 |
| D. Trade quality / consistency | 16 | 13 | 20 |
| **Raw total** | **75** | **62** | **100** |
| Hard cap | 40 | 40 | |
| **Final score** | **40** | **40** | **100** |

## Performance and risk

| Metric | Baseline | Challenger | Delta |
|---|---:|---:|---:|
| CAGR | 20.26% | 16.60% | -3.67% |
| Volatility | 18.20% | 17.23% | -0.98% |
| Sharpe | 1.107 | 0.979 | -0.128 |
| Sortino | 0.941 | 0.793 | -0.149 |
| Calmar | 0.630 | 0.475 | -0.155 |
| Maximum drawdown | -32.18% | -34.98% | -2.79% |
| Ulcer Index | 5.80% | 6.35% | +0.55% |
| Time underwater | 85.23% | 87.52% | +2.29% |
| Completed trades | 17 | 21 | +4 |
| Exposure | 73.19% | 64.26% | -8.93% |
| Win rate | 94.12% | 90.48% | -3.64% |
| Payoff ratio | 3.14 | 1.69 | |
| Profit factor | 50.21 | 16.07 | |
| Expectancy | 31.29% | 19.99% | -11.30% |

## Statistical significance

- Effective observations: 5458 / 5647.
- t-stat: 5.151 / 4.636; PSR: 1.0000 / 1.0000.
- DSR after 4,606 trials: 0.9336 / 0.8438.
- Jarque-Bera p: 0 / 0; Ljung-Box p: 3.09e-13 / 3.3e-09.
- Paired annual mean: -3.28%; HAC t=-2.582, p=0.010.
- Block-bootstrap 95% interval: [-0.058196289339465385, -0.00860584087141619].

## Robustness

- Historical-half efficiency: 0.763 / 0.905 (pseudo-OOS).
- Sensitivity Calmar: drawdown_15=0.301, drawdown_20=0.475, drawdown_25=0.465.
- Odd/even paired means: -4.37% / -2.24%.
- Trade bootstrap simulations: 5,000.
- 5x/10x-cost paired means: -3.35% / -3.44%.

## Bias assessment

| Bias | Status | Evidence |
|---|---|---|
| Lookahead / frequency mismatch | Absent | Price and rolling breadth end at close; next-open fill |
| Survivorship | Cannot fully verify | Aggregate index and breadth series |
| Data snooping | Present, material | 4,606 trials penalised |
| Costs | Included | 1x/2x/5x/10x commission and slippage |
| Liquidity | Low concern | Liquid index proxy; NDX price-index approximation |
| Synthetic breadth | Present before 2007 | 2007+ reported separately |
| Clean forward OOS | Insufficient | Too few post-freeze observations/trades |
| Regime overfit | Tested | Halves, odd/even, real-breadth, sensitivity |

## Guardrails

- baseline_parity: PASS
- calmar_improved: FAIL
- max_drawdown_not_worse: FAIL
- cagr_within_two_points: FAIL
- expectancy_not_worse: FAIL
- turnover_guardrail: PASS
- historical_halves_calmar_nonnegative: FAIL
- real_breadth_calmar_nonnegative: FAIL
- five_x_paired_return_positive: FAIL
- sensitivity_not_cliff_edge: FAIL

## Current signal

As of 2026-09-01, NDX 60-session return is +0.41%, rolling breadth max is 72.75%, and current breadth is 62.60% (drawdown 10.15 points).  The signal is inactive.

## Red Flags

1. Historical robustness is not clean forward OOS and many related rules were tested.
2. A rolling breadth peak may remain stale and create extra exits despite price confirmation.
3. Final-score comparisons must account for the baseline's fewer-than-30-trades cap.

## Improvement Recommendations

1. Follow the pre-registered verdict and leave the frozen baseline unchanged unless all guardrails pass.
2. Track any passing historical challenger forward from the frozen boundary before adoption.

## Decision

The verdict follows the primary Calmar objective and every guardrail without retuning.

Research evidence only; not investment advice.
