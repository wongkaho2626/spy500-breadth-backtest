# Backtest Verification Report — Breadth-only trajectory extension

## Verdict: Track as research challenger

The fixed 20-session breadth-only gate scores **83 / 100 (Tradeable)** versus the frozen baseline **40 / 100 (Weak)**.

## Backtest Scores

| Component | Frozen baseline | Challenger | Max |
|---|---:|---:|---:|
| A. Statistical validity | 23 | 26 | 30 |
| B. Risk-adjusted performance | 11 | 14 | 25 |
| C. Robustness / OOS | 25 | 25 | 25 |
| D. Trade quality / consistency | 16 | 18 | 20 |
| **Raw total** | **75** | **83** | **100** |
| Hard cap | 40 | 100 | |
| **Final score** | **40** | **83** | **100** |

## Performance

| Metric | Frozen baseline | Challenger | Delta |
|---|---:|---:|---:|
| CAGR | 20.20% | 21.53% | +1.32% |
| Sharpe | 1.104 | 1.140 | +0.036 |
| Sortino | 0.937 | 1.100 | +0.163 |
| Calmar | 0.628 | 0.733 | +0.105 |
| Maximum drawdown | -32.18% | -29.37% | +2.82% |
| Ulcer Index | 5.81% | 5.88% | +0.07% |
| Positive months | 53.06% | 62.24% | +9.18% |
| Clustered events | 17 | 48 | +31 |
| Decisions / extensions | 0 | 8 / 5 | |

## Robustness

- Paired annual mean: +1.18%; HAC t=2.096, p=0.0360.
- Block-bootstrap 95% interval: [0.0012474962811685833, 0.02283668243378816].
- Sensitivity scores: 10_sessions=83, 20_sessions=83, 40_sessions=83.
- Odd/even paired means: +1.62% / +0.76%.
- 5x-cost paired mean: +1.12%.
- 3x proxy-drag score: 83.

## Bias assessment

| Bias | Status | Evidence |
|---|---|---|
| Lookahead / frequency mismatch | Absent | Session-60 close breadth gate; next-open rotation |
| Survivorship | Cannot fully verify | Aggregate NDX and breadth series |
| Data snooping | Present, material | 4,602 trials; DSR applied |
| Trade independence | Adjusted | Exits within 21 sessions clustered |
| Costs | Included | Entry, exit and rotation; 1x/2x/5x/10x stress |
| Synthetic TQQQ | Present before 2010 | Actual-only period plus 1x/3x drag |
| Synthetic breadth | Present before 2007 | 2007+ period separate |
| Clean forward OOS | Insufficient | Post-freeze sample too short |

## Guardrails

- all_parity_controls: PASS
- at_least_30_clustered_events: PASS
- component_correlation_below_0_95: PASS
- final_score_at_least_80: PASS
- no_hard_cap: PASS
- cagr_improved: PASS
- sharpe_improved: PASS
- calmar_improved: PASS
- max_drawdown_below_30pct: PASS
- max_drawdown_within_two_points: PASS
- paired_mean_positive: PASS
- paired_hac_p_below_0_05: PASS
- paired_bootstrap_excludes_zero: PASS
- all_periods_positive: PASS
- odd_even_years_positive: PASS
- five_x_paired_return_positive: PASS
- sensitivity_stable: PASS
- profit_factor_above_1_2: PASS
- positive_expectancy: PASS
- three_x_drag_paired_positive: PASS
- three_x_drag_drawdown_below_30pct: PASS
- three_x_drag_score_at_least_80: PASS

## Decision

The decision follows all pre-registered score, regime, timing, sensitivity, cost and proxy guardrails. A historical pass supports forward tracking only and does not change the frozen baseline.

Research evidence only; not investment advice.
