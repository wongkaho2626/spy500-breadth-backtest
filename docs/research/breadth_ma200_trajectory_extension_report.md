# Backtest Verification Report — Trajectory-gated boost extension

## Verdict: Reject

The fixed 20-session trajectory gate scores **79 / 100 (Promising)** versus the frozen baseline **40 / 100 (Weak)**.

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
| CAGR | 20.20% | 21.31% | +1.11% |
| Sharpe | 1.104 | 1.133 | +0.029 |
| Sortino | 0.937 | 1.093 | +0.156 |
| Calmar | 0.628 | 0.728 | +0.100 |
| Maximum drawdown | -32.18% | -29.28% | +2.90% |
| Ulcer Index | 5.81% | 5.87% | +0.06% |
| Positive months | 53.06% | 62.24% | +9.18% |
| Clustered events | 17 | 48 | +31 |
| Extension decisions / extended | 0 | 8 / 3 | |

## Robustness

- Paired annual mean: +0.99%; HAC t=1.773, p=0.0762.
- Block-bootstrap 95% interval: [-0.0005706289684719319, 0.02075393612567282].
- Sensitivity scores: 10_sessions=79, 20_sessions=79, 40_sessions=79.
- Odd/even paired means: +1.58% / +0.43%.
- 5x-cost paired mean: +0.93%.
- 3x proxy-drag score: 79.

## Bias assessment

| Bias | Status | Evidence |
|---|---|---|
| Lookahead / frequency mismatch | Absent | Session-60 close gate; next-open rotation |
| Survivorship | Cannot fully verify | Aggregate NDX and breadth series |
| Data snooping | Present, material | 4,601 related trials; DSR applied |
| Trade independence | Adjusted | Exits within 21 sessions clustered |
| Costs | Included | Entry, exit and two-leg rotation; up to 10x stress |
| Synthetic TQQQ | Present before 2010 | Actual-only period plus 1x/3x drag |
| Synthetic breadth | Present before 2007 | 2007+ period separate |
| Clean forward OOS | Insufficient | Post-freeze sample too short |

## Guardrails

- frozen_force_short_force_long_parity: PASS
- at_least_30_clustered_events: PASS
- component_correlation_below_0_95: PASS
- final_score_at_least_80: FAIL
- no_hard_cap: PASS
- cagr_improved: PASS
- sharpe_improved: PASS
- calmar_improved: PASS
- max_drawdown_below_30pct: PASS
- max_drawdown_within_two_points: PASS
- paired_mean_positive: PASS
- paired_hac_p_below_0_05: FAIL
- paired_bootstrap_excludes_zero: FAIL
- all_periods_positive: FAIL
- odd_even_years_positive: PASS
- five_x_paired_return_positive: PASS
- sensitivity_stable: PASS
- profit_factor_above_1_2: PASS
- positive_expectancy: PASS
- three_x_drag_paired_positive: PASS
- three_x_drag_drawdown_below_30pct: PASS
- three_x_drag_score_at_least_80: FAIL

## Decision

The decision follows every pre-registered score, regime, timing, sensitivity, cost and proxy guardrail. A historical pass supports forward tracking only and does not alter the frozen baseline.

Research evidence only; not investment advice.
