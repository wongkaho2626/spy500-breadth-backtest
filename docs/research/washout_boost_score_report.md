# Backtest Verification Report — Washout-only 10% TQQQ boost

## Verdict: Reject

Canonical-signal 70/30 portfolio raw/final score: **65 / 40**.  Washout boost raw/final score: **61 / 40**.  The final score is the number that determines the 80-point objective.

## Backtest Scores

| Component | Baseline | 10% boost | Max |
|---|---:|---:|---:|
| A. Statistical validity | 23 | 23 | 30 |
| B. Risk-adjusted performance | 11 | 11 | 25 |
| C. Robustness / OOS | 15 | 11 | 25 |
| D. Trade quality / consistency | 16 | 16 | 20 |
| **Raw total** | **65** | **61** | **100** |
| Hard cap | 40 | 40 | |
| **Final score** | **40** | **40** | **100** |

Both strategies have fewer than 30 independent completed trades.  Under the installed rubric, that is a 40-point hard cap; stronger CAGR cannot remove it.

## Performance

| Metric | Baseline | 10% boost | Delta |
|---|---:|---:|---:|
| CAGR | 23.66% | 26.75% | +3.09% |
| Sharpe | 1.137 | 1.126 | -0.011 |
| Sortino | 0.979 | 0.944 | -0.035 |
| Calmar | 0.658 | 0.655 | -0.003 |
| Maximum drawdown | -35.98% | -40.84% | -4.87% |
| Ulcer Index | 7.73% | 8.56% | +0.83% |
| Positive months | 50.68% | 51.36% | +0.68% |
| Completed trades | 17 | 17 | 0 |

## Validity and robustness

- Baseline parity max equity difference: 0; signatures identical: True.
- Paired HAC t-stat: 3.485; p=0.000; 95% block-bootstrap annual-return interval [0.013403026056159072, 0.048949822720171024].
- Historical-half direction consistent: False.
- 5%/10%/15% sensitivity all retain positive CAGR delta: True.
- 5x-cost CAGR delta: +3.08%.
- 3x pre-inception drag CAGR: 26.68% versus baseline 23.66%.

## Bias assessment

| Bias | Status | Evidence |
|---|---|---|
| Lookahead / frequency mismatch | Absent | Close signal, next-session open fill |
| Survivorship | Cannot fully verify | Aggregate NDX signal plus annual top-1 history; delisted constituent completeness is not proven |
| Data snooping | Present, material | The 10% result was already disclosed and thousands of related trials exist |
| Costs | Included | 1x/2x/5x/10x commission and slippage stress |
| Synthetic data | Present before 2010 | TQQQ proxy uses 3x NDX less overlap-calibrated drag; 3x drag stressed |
| Clean forward OOS | Insufficient | Freeze occurred 2026-07-05; no completed forward round trip |

## Decision

This challenger is rejected as the requested 80-score strategy if any pre-registered guardrail fails.  Historical improvement may still be economically interesting, but it cannot be promoted into the frozen baseline without sufficient independent forward evidence.

Research evidence only; not investment advice.
