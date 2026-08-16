# Backtest Verification Report — Monthly MA200 × breadth regime exit

## Verdict: Reject

The 50% month-end challenger scores **61 / 100 (Needs work)** versus the frozen baseline **40 / 100 (Weak)**.

## Backtest Scores

| Component | Baseline | Challenger | Max |
|---|---:|---:|---:|
| A. Statistical validity | 23 | 22 | 30 |
| B. Risk-adjusted performance | 11 | 11 | 25 |
| C. Robustness / OOS | 25 | 15 | 25 |
| D. Trade quality / consistency | 16 | 13 | 20 |
| **Raw total** | **75** | **61** | **100** |
| Hard cap | 40 | 100 | |
| **Final score** | **40** | **61** | **100** |

## Performance

| Metric | Baseline | Challenger | Delta |
|---|---:|---:|---:|
| CAGR | 20.20% | 13.28% | -6.93% |
| Sharpe | 1.104 | 0.897 | -0.207 |
| Sortino | 0.937 | 0.649 | -0.288 |
| Calmar | 0.628 | 0.500 | -0.127 |
| Maximum drawdown | -32.18% | -26.53% | +5.65% |
| Ulcer Index | 5.81% | 8.04% | +2.23% |
| Positive months | 53.06% | 42.86% | -10.20% |
| Completed trades | 17 | 38 | +21 |
| Profit factor | 50.21 | 8.45 | |
| Expectancy | 31.29% | 9.50% | |

## Statistical significance and robustness

- Paired annual mean difference: -6.45%; HAC t=-3.019, p=0.003.
- 21-session block-bootstrap 95% interval: [-0.10696965986059048, -0.022304253888328435].
- Fixed-rule historical-half efficiency: baseline 0.766, challenger 0.547.  These are pseudo-OOS robustness checks, not clean forward evidence.
- Sensitivity Calmar values: breadth_40=0.400, breadth_50=0.500, breadth_60=0.481.
- 5x-cost paired annual mean: -6.83%.

## Bias assessment

| Bias | Status | Evidence |
|---|---|---|
| Lookahead / frequency mismatch | Absent | Month-end close signal, next-session-open fill |
| Survivorship | Cannot fully verify | Aggregate index and breadth; historical constituent reconstruction is not used |
| Data snooping | Present, material | At least 4,595 related trials; DSR penalty applied |
| Costs | Included | 1x/2x/5x/10x commission and slippage stress |
| Synthetic breadth | Present before 2007 | 2007+ real-breadth period reported separately |
| Clean forward OOS | Insufficient | No completed post-2026-07-05 forward round trip |

## Guardrails

- baseline_parity: PASS
- final_score_at_least_80: FAIL
- calmar_improved: FAIL
- max_drawdown_not_worse: PASS
- cagr_within_two_points: FAIL
- positive_expectancy: PASS
- profit_factor_above_1_2: PASS
- historical_halves_calmar_positive: FAIL
- turnover_guardrail: FAIL
- five_x_paired_return_positive: FAIL
- real_breadth_direction_positive: FAIL
- sensitivity_not_cliff_edge: FAIL

## Decision

The decision follows the pre-registered rule without moving the target.  A historical score at or above 80 is still only research evidence until sufficient clean post-freeze observations accumulate.

Research evidence only; not investment advice.
