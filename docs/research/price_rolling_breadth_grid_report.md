# Backtest Verification Report — price × rolling breadth grid

## Verdict: Reject

## Executive Summary

The pre-registered 210-cell robust winner is P=2%, D=30 points, C=70%. It scores **40 / 100 (Weak)** versus baseline **40 / 100 (Weak)**.

## Selected combinations

| Selection | P | D | C | Full CAGR | Full Calmar | Early Calmar | Late Calmar | Trades |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| robust_consensus | 2% | 30 | 70% | 18.98% | 0.590 | 0.560 | 0.869 | 14 |
| early_selected | 2% | 30 | 70% | 18.98% | 0.590 | 0.560 | 0.869 | 14 |
| late_selected | 0% | 30 | 30% | 18.24% | 0.389 | 0.316 | 0.942 | 12 |
| full_sample | 1% | 30 | 50% | 19.24% | 0.598 | 0.540 | 0.919 | 15 |

## Backtest Scores

| Component | Baseline | Challenger | Max |
|---|---:|---:|---:|
| A. Statistical validity | 23 | 23 | 30 |
| B. Risk-adjusted performance | 11 | 11 | 25 |
| C. Robustness / OOS | 25 | 18 | 25 |
| D. Trade quality / consistency | 16 | 18 | 20 |
| **Raw total** | **75** | **70** | **100** |
| Hard cap | 40 | 40 | |
| **Final score** | **40** | **40** | **100** |

## Performance and risk — robust candidate

| Metric | Baseline | Challenger | Delta |
|---|---:|---:|---:|
| CAGR | 20.26% | 18.98% | -1.28% |
| Sharpe | 1.107 | 1.007 | -0.100 |
| Sortino | 0.941 | 0.893 | -0.048 |
| Calmar | 0.630 | 0.590 | -0.040 |
| Maximum drawdown | -32.18% | -32.18% | -0.00% |
| Ulcer Index | 5.80% | 6.53% | +0.73% |
| Time underwater | 85.23% | 85.14% | -0.10% |
| Completed trades | 17 | 14 | -3 |
| Exposure | 73.19% | 81.53% | +8.33% |
| Profit factor | 50.21 | 57.86 | |
| Expectancy | 31.29% | 43.89% | +12.61% |

## Statistical significance

- DSR after 4,816 trials: 0.9321 / 0.8563.
- Paired annual mean: -0.90%; HAC t=-0.742, p=0.458.
- Block-bootstrap 95% interval: [-0.033419816934964715, 0.01308955646255987].

## Robustness

- Candidate boundary: True.
- Stable immediate neighbours: 4 / 5 (80%).
- Historical-half efficiency: 0.763 / 0.948 (pseudo-OOS).
- Odd/even paired means: -2.26% / +0.38%.
- 5x/10x-cost paired means: -0.85% / -0.79%.

## Bias assessment

| Bias | Status | Evidence |
|---|---|---|
| Lookahead / frequency mismatch | Absent | Close features; next-open fill |
| Survivorship | Cannot fully verify | Aggregate index and breadth series |
| Data snooping | Present, material | 210 new cells; 4,816 total trials in DSR |
| Costs | Included | 1x/2x/5x/10x stress |
| Liquidity | Low concern | Liquid index proxy; NDX price-index approximation |
| Synthetic breadth | Present before 2007 | 2007+ reported separately |
| Clean forward OOS | Insufficient | Grid and latest history are now seen |
| Regime overfit | Tested, not eliminated | Halves, reverse direction, neighbours, odd/even |

## Guardrails

- baseline_parity: PASS
- calmar_improved: FAIL
- max_drawdown_not_worse: FAIL
- cagr_within_two_points: PASS
- expectancy_not_worse: PASS
- historical_halves_calmar_nonnegative: FAIL
- real_breadth_calmar_nonnegative: FAIL
- five_x_paired_return_positive: FAIL
- candidate_not_on_boundary: FAIL
- local_neighbour_stability: FAIL

## Current signal

As of 2026-09-01, the selected signal is inactive: NDX 60-session return +0.41%, breadth drawdown 10.15 points, current breadth 62.60%.

## Red Flags

1. This is a 210-cell search on already-seen historical data.
2. A full-sample winner is descriptive, not clean OOS evidence.
3. Boundary or locally unstable winners indicate the grid has found a ridge rather than a robust plateau.

## Improvement Recommendations

1. Do not expand the grid after seeing these results in the same round.
2. Keep any passing robust candidate isolated and forward-track it from the frozen boundary.

## Decision

The verdict follows the pre-registered maximin selection and guardrails.

Research evidence only; not investment advice.
