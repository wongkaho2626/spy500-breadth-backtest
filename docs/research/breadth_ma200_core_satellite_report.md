# Backtest Verification Report — Breadth + MA200 core-satellite

## Verdict: Reject

The fixed 70/30 ensemble scores **76 / 100 (Promising)** versus the frozen baseline **40 / 100 (Weak)**.

## Backtest Scores

| Component | Baseline | 70/30 ensemble | Max |
|---|---:|---:|---:|
| A. Statistical validity | 23 | 26 | 30 |
| B. Risk-adjusted performance | 11 | 14 | 25 |
| C. Robustness / OOS | 25 | 18 | 25 |
| D. Trade quality / consistency | 16 | 18 | 20 |
| **Raw total** | **75** | **76** | **100** |
| Hard cap | 40 | 100 | |
| **Final score** | **40** | **76** | **100** |

## Performance

| Metric | Baseline | Ensemble | Delta |
|---|---:|---:|---:|
| CAGR | 20.20% | 18.71% | -1.50% |
| Sharpe | 1.104 | 1.109 | +0.004 |
| Sortino | 0.937 | 1.047 | +0.110 |
| Calmar | 0.628 | 0.682 | +0.055 |
| Maximum drawdown | -32.18% | -27.42% | +4.76% |
| Ulcer Index | 5.81% | 5.44% | -0.37% |
| Positive months | 53.06% | 62.24% | +9.18% |
| Completed / clustered events | 17 | 48 | +31 |
| Profit factor | 50.21 | 29.16 | |
| Expectancy | 31.29% | 10.07% | |

## Independence and robustness

- Component daily-return correlation: 0.577.
- Raw component exits / 21-session clusters: 85 / 48.
- Paired annual mean: -1.50%; HAC t=-2.766, p=0.006.
- Block-bootstrap 95% interval: [-0.025578197014795544, -0.004412030044066251].
- Historical-half efficiency: baseline 0.766; ensemble 0.764 (pseudo-OOS only).
- Sensitivity Sharpes: 20%=1.110, 30%=1.109, 40%=1.103.
- 5x-cost paired annual mean: -1.55%.

## Bias assessment

| Bias | Status | Evidence |
|---|---|---|
| Lookahead / frequency mismatch | Absent | Both sleeves signal on close and fill next-session open |
| Survivorship | Cannot fully verify | Aggregate NDX and breadth series |
| Data snooping | Present, material | At least 4,596 related trials; DSR penalty applied |
| Trade independence | Adjusted | Component exits within 21 sessions counted as one event |
| Costs | Included | Every component transition; 1x/2x/5x/10x stress |
| Synthetic breadth | Present before 2007 | 2007+ result reported separately |
| Clean forward OOS | Insufficient | No completed post-freeze ensemble evaluation period |

## Guardrails

- baseline_parity: PASS
- at_least_30_clustered_events: PASS
- component_correlation_below_0_95: PASS
- final_score_at_least_80: FAIL
- sharpe_improved: PASS
- calmar_improved: PASS
- max_drawdown_not_worse: PASS
- cagr_within_two_points: PASS
- historical_halves_positive: FAIL
- real_breadth_positive: PASS
- five_x_paired_return_positive: FAIL
- sensitivity_stable: FAIL
- profit_factor_above_1_2: PASS
- positive_expectancy: PASS

## Decision

The decision follows the pre-registered score and economic guardrails.  A historical score of 80 or more would still require clean forward tracking before adoption.

Research evidence only; not investment advice.
