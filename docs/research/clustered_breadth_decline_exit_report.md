# Backtest Verification Report — clustered breadth-decline exit

## Verdict: Reject

## Executive Summary

The count>=4 challenger scores **62 / 100 (Needs work)** versus the frozen baseline **40 / 100 (Weak)**.  The decision follows the pre-registered Calmar objective and guardrails, not CAGR alone.

The baseline raw score is 75, above the challenger's 62; its displayed final score is capped at 40 only because it has fewer than 30 completed trades.  The challenger's higher capped score therefore is not evidence of an economic improvement.

## Backtest Scores

| Component | Baseline | Challenger | Max |
|---|---:|---:|---:|
| A. Statistical validity | 23 | 22 | 30 |
| B. Risk-adjusted performance | 11 | 9 | 25 |
| C. Robustness / OOS | 25 | 18 | 25 |
| D. Trade quality / consistency | 16 | 13 | 20 |
| **Raw total** | **75** | **62** | **100** |
| Hard cap | 40 | 100 | |
| **Final score** | **40** | **62** | **100** |

## Performance and risk

| Metric | Baseline | Challenger | Delta |
|---|---:|---:|---:|
| CAGR | 20.26% | 5.96% | -14.30% |
| Annual volatility | 18.20% | 9.40% | -8.80% |
| Sharpe | 1.107 | 0.664 | -0.443 |
| Sortino | 0.941 | 0.276 | -0.665 |
| Calmar | 0.630 | 0.321 | -0.309 |
| Maximum drawdown | -32.18% | -18.59% | +13.59% |
| Ulcer Index | 5.80% | 5.91% | +0.11% |
| Time underwater | 85.23% | 93.68% | +8.45% |
| Completed trades | 17 | 50 | +33 |
| Exposure | 73.19% | 14.77% | -58.42% |
| Round trips / year | 0.69 | 2.03 | +1.34 |
| Win rate | 94.12% | 70.00% | -24.12% |
| Payoff ratio | 3.14 | 1.72 | |
| Profit factor | 50.21 | 4.01 | |
| Expectancy | 31.29% | 3.11% | -28.18% |

NDX price-index buy-and-hold over the same dates: CAGR 12.45%, Sharpe 0.621, maximum drawdown -53.71%.

## Distribution and consistency

| Metric | Baseline | Challenger |
|---|---:|---:|
| Daily VaR 95% | -1.77% | -0.30% |
| Daily CVaR 95% | -2.73% | -1.27% |
| Skewness | 0.313 | 2.705 |
| Excess kurtosis | 9.85 | 73.21 |
| Positive months | 53.04% | 19.93% |
| Rolling 252d Sharpe min / max / std | -1.21 / 3.12 / 0.75 | -2.21 / 2.91 / 0.88 |

## Statistical significance

- Baseline/challenger effective observations: 5458 / 5409.
- Mean-return t-stat: 5.151 / 3.075; PSR vs zero: 1.0000 / 0.9993.
- DSR probability after 4,603 related trials: 0.9336 / 0.3227.
- Jarque-Bera p: 0 / 0; Ljung-Box(10) p: 3.09e-13 / 1.5e-07.
- Paired annual mean difference: -13.91%; HAC t=-4.666, p=0.000.
- 21-session block-bootstrap 95% interval: [-0.19431115256225265, -0.077933687889773].

## Robustness

- Historical-half efficiency (pseudo-OOS): baseline 0.763, challenger 0.995.
- Sensitivity Calmar: count_3=0.248, count_4=0.321, count_5=0.282.
- Odd/even-year paired annual means: -22.05% / -6.20%.
- Trade bootstrap (5,000 simulations) terminal-return p05/p50/p95: 1.12 / 3.19 / 7.59; MDD p05/p50/p95: -22.34% / -12.29% / -6.69%.
- 5x-cost paired annual mean: -14.50%; 10x: -15.26%.

### Historical direction splits

| Slice | CAGR delta | Sharpe delta | Calmar delta | MDD delta |
|---|---:|---:|---:|---:|
| early_period | -10.34% | -0.283 | -0.079 | +16.72% |
| late_period | -18.18% | -0.577 | -0.762 | +3.94% |
| real_breadth_period | -16.07% | -0.462 | -0.349 | +13.59% |

### Cost stress

| Cost multiplier | Baseline CAGR | Challenger CAGR | Paired annual mean |
|---|---:|---:|---:|
| 1x | 20.26% | 5.96% | -13.91% |
| 2x | 20.17% | 5.72% | -14.06% |
| 5x | 19.88% | 5.00% | -14.50% |
| 10x | 19.41% | 3.79% | -15.26% |

## Bias assessment

| Bias | Status | Evidence |
|---|---|---|
| Lookahead / frequency mismatch | Absent | Ten closes through t; fill at t+1 open |
| Survivorship | Cannot fully verify | Aggregate NDX and aggregate S&P breadth; no constituent reconstruction |
| Data snooping | Present, material | 4,603 related trials in DSR penalty |
| Transaction costs | Included | $1 and 0.05% per side, stressed at 1x/2x/5x/10x |
| Liquidity | Low concern | Liquid index proxy, but NDX price-index execution is still an approximation |
| Synthetic breadth | Present before 2007 | 2007+ real-breadth result reported separately |
| Clean forward OOS | Insufficient | Post-freeze window is too short for five completed trades |
| Regime overfit | Tested, unresolved if inconsistent | Historical halves, odd/even years, named crises, and sensitivity reported |

## Guardrails

- baseline_parity: PASS
- calmar_improved: FAIL
- max_drawdown_not_worse: PASS
- cagr_within_one_point: FAIL
- expectancy_not_worse: FAIL
- completed_trades_within_25pct: FAIL
- historical_halves_calmar_nonnegative: FAIL
- real_breadth_calmar_nonnegative: FAIL
- five_x_paired_return_positive: FAIL
- sensitivity_not_cliff_edge: FAIL

## Current signal

As of 2026-09-01, the raw rule is ACTIVE: 4 qualifying declines, breadth slope -0.683 points/session, latest breadth return -5.62%.

## Red Flags

1. The rule cuts exposure from 73% to 15% and misses too much of the long-run NDX advance.
2. Forty-six of fifty challenger exits come from the new rule, creating repeated exit/re-entry churn.
3. CAGR, Sharpe, Sortino, Calmar, expectancy, time underwater, both historical halves, and 2007+ real-breadth evidence all deteriorate.
4. The paired underperformance is statistically clear and survives block bootstrap and every cost stress.
5. Clean post-freeze evidence is only 41 daily observations and contains no completed forward round trip.

## Improvement Recommendations

1. Do not add this condition as an unconditional sell rule to the frozen strategy.
2. Keep today's trigger as a diagnostic warning only; do not treat it as validated execution evidence.
3. If a later research round is requested, pre-register a different economic role such as a regime-gated warning or exposure throttle rather than retuning this failed exit on the same history.

## Decision

The result is rejected if any pre-registered guardrail fails.  Even a historical pass would remain a research challenger until meaningful clean post-freeze evidence accumulates.

Research evidence only; not investment advice.
