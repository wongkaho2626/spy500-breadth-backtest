# Backtest Verification Report — Breadth + TQQQ MA200 satellite

## Verdict: Reject

The fixed 85/15 ensemble scores **73 / 100 (Promising)** versus the frozen baseline **40 / 100 (Weak)**.  Under 3× pre-inception drag it scores **73 / 100**.

## Backtest Scores

| Component | Baseline | 85/15 ensemble | Max |
|---|---:|---:|---:|
| A. Statistical validity | 23 | 26 | 30 |
| B. Risk-adjusted performance | 11 | 11 | 25 |
| C. Robustness / OOS | 25 | 18 | 25 |
| D. Trade quality / consistency | 16 | 18 | 20 |
| **Raw total** | **75** | **73** | **100** |
| Hard cap | 40 | 100 | |
| **Final score** | **40** | **73** | **100** |

## Performance

| Metric | Baseline | Ensemble | Delta |
|---|---:|---:|---:|
| CAGR | 20.20% | 20.93% | +0.72% |
| Sharpe | 1.104 | 1.015 | -0.089 |
| Sortino | 0.937 | 0.936 | -0.001 |
| Calmar | 0.628 | 0.662 | +0.034 |
| Maximum drawdown | -32.18% | -31.62% | +0.57% |
| Ulcer Index | 5.81% | 7.40% | +1.59% |
| Positive months | 53.06% | 60.88% | +7.82% |
| Completed / clustered events | 17 | 48 | +31 |
| Profit factor | 50.21 | 17.70 | |
| Expectancy | 31.29% | 11.96% | |

## Independence and robustness

- Component daily-return correlation: 0.576.
- Raw component exits / 21-session clusters: 85 / 48.
- Paired annual mean: +1.14%; HAC t=0.887, p=0.375.
- Block-bootstrap 95% interval: [-0.013841203183061415, 0.03679944621893794].
- Historical-half efficiency: baseline 0.766; ensemble 0.813 (pseudo-OOS only).
- Sensitivity scores: 10%=76, 15%=73, 20%=73.
- 5x-cost paired annual mean: +0.62%.
- 3× proxy-drag paired annual mean: +0.68%; MDD -31.78%; score 73.

## Bias assessment

| Bias | Status | Evidence |
|---|---|---|
| Lookahead / frequency mismatch | Absent | Both sleeves signal on close and fill next-session open |
| Survivorship | Cannot fully verify | Aggregate NDX and breadth series |
| Data snooping | Present, material | At least 4,597 related trials; DSR penalty applied |
| Trade independence | Adjusted | Component exits within 21 sessions counted as one event |
| Costs | Included | Every component transition; 1x/2x/5x/10x stress |
| Synthetic TQQQ | Present before 2010 | Actual post-inception; calibrated 1× and punitive 3× drag tested |
| Synthetic breadth | Present before 2007 | 2007+ result reported separately |
| Clean forward OOS | Insufficient | No completed post-freeze ensemble evaluation period |

## Guardrails

- baseline_parity: PASS
- at_least_30_clustered_events: PASS
- component_correlation_below_0_95: PASS
- final_score_at_least_80: FAIL
- cagr_improved: PASS
- sharpe_improved: FAIL
- calmar_improved: PASS
- max_drawdown_within_two_points: PASS
- paired_mean_positive: PASS
- paired_bootstrap_excludes_zero: FAIL
- historical_halves_positive: FAIL
- real_breadth_positive: FAIL
- five_x_paired_return_positive: PASS
- sensitivity_stable: FAIL
- profit_factor_above_1_2: PASS
- positive_expectancy: PASS
- three_x_drag_paired_positive: PASS
- three_x_drag_drawdown_within_two_points: PASS
- three_x_drag_score_at_least_80: FAIL

## Decision

The decision follows every pre-registered score, risk, cost, sensitivity, and synthetic-data guardrail.  Historical success can only justify forward tracking, not an immediate frozen-baseline change.

Research evidence only; not investment advice.
