# Backtest Verification Report — MA200 ensemble + washout TQQQ boost

## Verdict: Reject

The fixed 70/30 ensemble with a 10% washout-only TQQQ allocation scores **77 / 100 (Promising)** versus the frozen baseline **40 / 100 (Weak)**.

## Backtest Scores

| Component | Frozen baseline | Challenger | Max |
|---|---:|---:|---:|
| A. Statistical validity | 23 | 26 | 30 |
| B. Risk-adjusted performance | 11 | 11 | 25 |
| C. Robustness / OOS | 25 | 22 | 25 |
| D. Trade quality / consistency | 16 | 18 | 20 |
| **Raw total** | **75** | **77** | **100** |
| Hard cap | 40 | 100 | |
| **Final score** | **40** | **77** | **100** |

## Performance

| Metric | Frozen baseline | Challenger | Delta |
|---|---:|---:|---:|
| CAGR | 20.20% | 24.78% | +4.57% |
| Sharpe | 1.104 | 1.096 | -0.008 |
| Sortino | 0.937 | 1.028 | +0.091 |
| Calmar | 0.628 | 0.723 | +0.096 |
| Maximum drawdown | -32.18% | -34.25% | -2.07% |
| Ulcer Index | 5.81% | 7.59% | +1.78% |
| Positive months | 53.06% | 62.59% | +9.52% |
| Clustered events | 17 | 48 | +31 |

## Robustness

- Paired annual mean: +4.64%; HAC t=4.083, p=0.000.
- Block-bootstrap 95% interval: [0.024795132214308877, 0.06939426902843668].
- Sensitivity scores: 5%=77, 10%=77, 15%=77.
- 5x-cost paired annual mean: +4.63%.
- 3x proxy-drag score: 77; paired annual mean +4.51%.

## Bias assessment

| Bias | Status | Evidence |
|---|---|---|
| Lookahead / frequency mismatch | Absent | Close signals fill next-session open |
| Survivorship | Cannot fully verify | Aggregate NDX and breadth series |
| Data snooping | Present, material | At least 4,598 related trials; DSR penalty applied |
| Trade independence | Adjusted | Component exits within 21 sessions form one event |
| Costs | Included | Every traded leg; 1x/2x/5x/10x stress |
| Synthetic TQQQ | Present before 2010 | Actual post-inception; 1x and punitive 3x drag |
| Synthetic breadth | Present before 2007 | 2007+ period reported separately |
| Clean forward OOS | Insufficient | Post-freeze sample is not meaningful |

## Guardrails

- baseline_and_zero_boost_parity: PASS
- at_least_30_clustered_events: PASS
- component_correlation_below_0_95: PASS
- final_score_at_least_80: FAIL
- cagr_improved: PASS
- sharpe_improved: FAIL
- calmar_improved: PASS
- max_drawdown_within_two_points: FAIL
- paired_mean_positive: PASS
- paired_bootstrap_excludes_zero: PASS
- all_periods_positive: FAIL
- five_x_paired_return_positive: PASS
- sensitivity_stable: FAIL
- profit_factor_above_1_2: PASS
- positive_expectancy: PASS
- three_x_drag_paired_positive: PASS
- three_x_drag_drawdown_within_two_points: FAIL
- three_x_drag_score_at_least_80: FAIL

## Decision

The decision follows every pre-registered performance, inference, period, sensitivity, cost and synthetic-data guardrail. A historical pass can only be forward-tracked and does not change the frozen baseline.

Research evidence only; not investment advice.
