# Backtest Verification Report — rolling breadth-drawdown exit

## Verdict: Reject

## Executive Summary

The rolling-max challenger scores **57 / 100 (Needs work)** versus baseline **40 / 100 (Weak)**.  The verdict follows the pre-registered Calmar objective and guardrails.

## Backtest Scores

| Component | Baseline | Challenger | Max |
|---|---:|---:|---:|
| A. Statistical validity | 23 | 22 | 30 |
| B. Risk-adjusted performance | 11 | 7 | 25 |
| C. Robustness / OOS | 25 | 15 | 25 |
| D. Trade quality / consistency | 16 | 13 | 20 |
| **Raw total** | **75** | **57** | **100** |
| Hard cap | 40 | 100 | |
| **Final score** | **40** | **57** | **100** |

## Performance and risk

| Metric | Baseline | Challenger | Delta |
|---|---:|---:|---:|
| CAGR | 20.26% | 7.34% | -12.92% |
| Volatility | 18.20% | 11.13% | -7.07% |
| Sharpe | 1.107 | 0.693 | -0.414 |
| Sortino | 0.941 | 0.376 | -0.566 |
| Calmar | 0.630 | 0.299 | -0.330 |
| Maximum drawdown | -32.18% | -24.53% | +7.66% |
| Ulcer Index | 5.80% | 6.92% | +1.12% |
| Time underwater | 85.23% | 84.04% | -1.19% |
| Completed trades | 17 | 49 | +32 |
| Exposure | 73.19% | 27.87% | -45.32% |
| Win rate | 94.12% | 69.39% | -24.73% |
| Payoff ratio | 3.14 | 1.97 | |
| Profit factor | 50.21 | 4.47 | |
| Expectancy | 31.29% | 3.85% | -27.44% |

## Statistical significance

- Effective observations: 5458 / 5034.
- t-stat: 5.151 / 3.099; PSR: 1.0000 / 0.9992.
- DSR after 4,605 trials: 0.9336 / 0.2987.
- Jarque-Bera p: 0 / 0; Ljung-Box p: 3.09e-13 / 3.51e-07.
- Paired annual mean: -12.43%; HAC t=-4.970, p=0.000.
- Block-bootstrap 95% interval: [-0.17073608364860476, -0.07366290846046179].

## Robustness

- Historical-half efficiency: 0.763 / 0.513 (pseudo-OOS).
- Sensitivity Calmar: drawdown_15=0.606, drawdown_20=0.299, drawdown_25=0.215.
- Odd/even paired means: -12.93% / -11.96%.
- Trade bootstrap simulations: 5,000.
- 5x/10x-cost paired means: -13.01% / -13.75%.

## Bias assessment

| Bias | Status | Evidence |
|---|---|---|
| Lookahead / frequency mismatch | Absent | Rolling window ends at signal close; next-open fill |
| Survivorship | Cannot fully verify | Aggregate index and breadth series |
| Data snooping | Present, material | 4,605 trials penalised |
| Costs | Included | 1x/2x/5x/10x commission and slippage |
| Liquidity | Low concern | Liquid index proxy; NDX price-index approximation |
| Synthetic breadth | Present before 2007 | 2007+ reported separately |
| Clean forward OOS | Insufficient | Too few post-freeze observations/trades |
| Regime overfit | Tested | Halves, odd/even, real-breadth, sensitivity |

## Guardrails

- baseline_parity: PASS
- calmar_improved: FAIL
- max_drawdown_not_worse: PASS
- cagr_within_two_points: FAIL
- expectancy_not_worse: FAIL
- turnover_guardrail: FAIL
- historical_halves_calmar_nonnegative: FAIL
- real_breadth_calmar_nonnegative: FAIL
- five_x_paired_return_positive: FAIL
- sensitivity_not_cliff_edge: FAIL

## Current signal

As of 2026-09-01, rolling 60-session breadth max is 72.75% and current breadth is 62.60%, a drawdown of 10.15 points.  The signal is inactive.

## Red Flags

1. A rolling peak can keep the condition active long after the first breakdown, increasing whipsaw.
2. Pre-2007 breadth is synthetic; pre-freeze results are not clean forward OOS.
3. A score increase caused only by clearing the 30-trade cap is not an economic improvement.

## Improvement Recommendations

1. Follow the pre-registered verdict and leave the frozen baseline unchanged.
2. Inspect raw events and actual fills separately; repeated active days are not independent signals.

## Decision

The verdict follows the primary Calmar objective and every guardrail without retuning.

Research evidence only; not investment advice.
