# Backtest Verification Report — Fixed 80-session confirmation

## Verdict: Reject

The selected fixed 80-session rule scores **83 / 100 (Tradeable)** versus the frozen baseline **40 / 100 (Weak)**.

Selection status: **selected from prior sensitivity; confirmatory, not blind**. Historical confirmation is not clean forward OOS.

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
| CAGR | 20.20% | 21.45% | +1.24% |
| Sharpe | 1.104 | 1.131 | +0.027 |
| Sortino | 0.937 | 1.090 | +0.153 |
| Calmar | 0.628 | 0.730 | +0.102 |
| Maximum drawdown | -32.18% | -29.37% | +2.82% |
| Ulcer Index | 5.81% | 6.13% | +0.32% |
| Positive months | 53.06% | 62.59% | +9.52% |
| Clustered events | 17 | 48 | +31 |

## Confirmation evidence

- Paired annual mean: +1.13%; HAC t=1.966, p=0.0493.
- Block-bootstrap 95% interval: [0.00045639411381643403, 0.022421199241913756].
- Sensitivity scores: 60_sessions=79, 80_sessions=83, 100_sessions=80.
- Odd-year paired mean: +1.67%; even-year: +0.62%.
- 5x-cost paired annual mean: +1.07%.
- 3x proxy-drag score: 80; paired annual mean +1.08%.

## Bias assessment

| Bias | Status | Evidence |
|---|---|---|
| Lookahead / frequency mismatch | Absent | Every entry, exit and age rotation fills next-session open |
| Survivorship | Cannot fully verify | Aggregate NDX and breadth series |
| Data snooping | Present, material | 80 sessions was selected from prior sensitivity; 4,600 trials in DSR |
| Trade independence | Adjusted | Component exits within 21 sessions form one event |
| Costs | Included | Entry, exit and two-leg rotation; 1x/2x/5x/10x stress |
| Synthetic TQQQ | Present before 2010 | Actual-only period plus 1x/3x proxy drag reported |
| Synthetic breadth | Present before 2007 | 2007+ period reported separately |
| Clean forward OOS | Insufficient | Only a very short post-freeze sample exists |

## Guardrails

- frozen_and_prior_engine_parity: PASS
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
- all_periods_positive: FAIL
- odd_even_years_positive: PASS
- five_x_paired_return_positive: PASS
- sensitivity_stable: PASS
- profit_factor_above_1_2: PASS
- positive_expectancy: PASS
- three_x_drag_paired_positive: PASS
- three_x_drag_drawdown_below_30pct: PASS
- three_x_drag_score_at_least_80: PASS

## Decision

A complete historical pass reaches the research-score objective and supports forward tracking only. It does not justify frozen-baseline adoption without meaningful clean forward OOS evidence and an explicit user request.

Research evidence only; not investment advice.
