# Backtest Verification Report — Timed washout TQQQ boost

## Verdict: Reject

The fixed 60-session challenger scores **79 / 100 (Promising)** versus the frozen baseline **40 / 100 (Weak)**.

## Backtest Scores

| Component | Frozen baseline | Challenger | Max |
|---|---:|---:|---:|
| A. Statistical validity | 23 | 26 | 30 |
| B. Risk-adjusted performance | 11 | 14 | 25 |
| C. Robustness / OOS | 25 | 21 | 25 |
| D. Trade quality / consistency | 16 | 18 | 20 |
| **Raw total** | **75** | **79** | **100** |
| Hard cap | 40 | 100 | |
| **Final score** | **40** | **79** | **100** |

## Performance

| Metric | Frozen baseline | Challenger | Delta |
|---|---:|---:|---:|
| CAGR | 20.20% | 21.16% | +0.95% |
| Sharpe | 1.104 | 1.131 | +0.027 |
| Sortino | 0.937 | 1.093 | +0.156 |
| Calmar | 0.628 | 0.723 | +0.095 |
| Maximum drawdown | -32.18% | -29.28% | +2.90% |
| Ulcer Index | 5.81% | 5.87% | +0.06% |
| Positive months | 53.06% | 62.24% | +9.18% |
| Clustered events | 17 | 48 | +31 |
| Completed rotations | 0 | 7 | |

## Statistical significance and robustness

- Paired annual mean: +0.85%; HAC t=1.556, p=0.1198.
- Block-bootstrap 95% interval: [-0.0012663235320334793, 0.019494327594087837].
- Sensitivity scores: 40_sessions=79, 60_sessions=79, 80_sessions=83.
- 5x-cost paired annual mean: +0.78%.
- 3x proxy-drag score: 79; paired annual mean +0.81%.

## Bias assessment

| Bias | Status | Evidence |
|---|---|---|
| Lookahead / frequency mismatch | Absent | All signals, including age rotation, fill next-session open |
| Survivorship | Cannot fully verify | Aggregate NDX and breadth series |
| Data snooping | Present, material | At least 4,599 related trials; DSR penalty applied |
| Trade independence | Adjusted | Component exits within 21 sessions form one event |
| Costs | Included | Each entry, exit and two-leg rotation; 1x/2x/5x/10x stress |
| Synthetic TQQQ | Present before 2010 | Actual post-inception; 1x and punitive 3x drag |
| Synthetic breadth | Present before 2007 | 2007+ result reported separately |
| Clean forward OOS | Insufficient | Post-freeze sample remains too short |

## Guardrails

- frozen_and_unlimited_parity: PASS
- at_least_30_clustered_events: PASS
- component_correlation_below_0_95: PASS
- final_score_at_least_80: FAIL
- cagr_improved: PASS
- sharpe_improved: PASS
- calmar_improved: PASS
- max_drawdown_within_two_points: PASS
- paired_mean_positive: PASS
- paired_bootstrap_excludes_zero: FAIL
- all_periods_positive: FAIL
- five_x_paired_return_positive: PASS
- sensitivity_stable: PASS
- profit_factor_above_1_2: PASS
- positive_expectancy: PASS
- three_x_drag_paired_positive: PASS
- three_x_drag_drawdown_within_two_points: PASS
- three_x_drag_score_at_least_80: FAIL

## Decision

The decision follows all pre-registered score, risk, timing, period, sensitivity, cost and synthetic-data guardrails. Historical success can only justify forward tracking and does not modify the frozen baseline.

Research evidence only; not investment advice.
