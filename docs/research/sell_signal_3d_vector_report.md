# Sell-signal-only 3D vector report

## Decision: Reject

Removing all six trajectory dimensions and keeping only the three continuous inputs of the frozen bearish-divergence sell rule produces a strict causal AUC of 0.467. This is below the 0.5 random-ranking benchmark. The three-dimensional sell vector does not predict a subsequent 20% SPX decline.

## Exact vector

`[NDX 60-session return %, breadth 60-session fall in points, current breadth %]`

The prior NDX slope, VIX, VIX slope, SPX drawdown and drawdown slope features are excluded. Continuous values are used because three booleans would produce only eight possible vectors and many distance ties.

## Method and data

- Data: 2002-01-02 through 2026-07-30, 6,180 sessions.
- Target: minimum SPX return over the following 126 sessions less than or equal to -20%.
- Evaluation: 61 pre-existing transition-signal clusters; 48 have a strict causal breadth-comparable peak reference.
- Positive clusters: 3, all belonging to the same 2020 crash; effective independent positive episodes: 1.
- Scaling: historical median and IQR.
- Causal reference: a peak episode is available only after its -20% breach occurred.
- New parameters and tuned thresholds: none.
- Frozen strategy, costs, next-open execution and trade logic: unchanged.

## Results

| Metric | 3D sell-only vector |
|---|---:|
| Full-sample AUC | 0.833 |
| Strict causal AUC | 0.467 |
| True-cluster median causal similarity | 77.64% |
| False-cluster median causal similarity | 79.07% |

The full-sample result is hindsight-biased because it permits peak vectors that were unknown at the query date. Once future peaks are removed, false signals are slightly more peak-like than the true group. The sell-only vector therefore has no causal ranking edge.

Several false signals receive extremely high causal similarity, including 2017-05-19 at 99.74%, 2021-09-15 at 99.73%, and 2014-09-12 at 99.20%. This overlap prevents a useful similarity threshold.

## Exact canonical sell outcomes

The frozen bearish-divergence rule triggered nine completed exits. Only the 2020-02-25 signal was followed by a 20% SPX decline within 126 sessions.

- Raw rate: 1/9, or 11.11%.
- Jeffreys posterior estimate: 15.00%.
- 90% interval: 1.99%–36.07%.

This is too imprecise to use as a calibrated crash probability.

## Current vector — 2026-07-30

`[+1.6438%, -14.94 points, 68.41%]`

Interpretation:

- NDX return is below the required +3%: not met.
- Breadth did not fall 20 points; it increased 14.94 points: not met.
- Breadth is 68.41%, above the required sub-60% level: not met.
- Frozen bearish-divergence signal: inactive, 0/3 conditions.
- 3D causal peak similarity: 49.91%, an ordinary middle-ranked value rather than a high-risk extreme.
- Nearest historical transition signal: 2016-06-21, standardized distance 0.137; it was not followed by a 20% decline.
- Climax-top and trailing-stop signals are also inactive.

The current three-dimensional sell vector does not indicate a forecastable major decline. The 49.91% figure is a similarity rank, not a 49.91% crash probability.

## Bias and robustness audit

| Check | Status | Evidence |
|---|---|---|
| Feature lookahead | Absent | Only current and 60-session-lagged observations are used. |
| Reference lookahead | Absent in primary result | Strict causal peak eligibility is enforced. |
| Full-sample leakage | Present by design | The 0.833 AUC is reported only to show the hindsight effect. |
| Independent sample adequacy | Failed | Only one independent positive crash episode. |
| Data snooping | High risk | More than 4,590 related vector/sell trials precede this test. |
| Clean forward OOS | Insufficient | No completed post-freeze forward trade. |
| Parameter stability | Not applicable | No parameter was tuned. |
| Transaction costs | Unchanged | No new trade rule was introduced. |

## Predictive evidence score: 3 / 100 — Reject

| Component | Score | Max |
|---|---:|---:|
| A. Statistical validity and significance | 0 | 30 |
| B. Risk-adjusted trading performance | unavailable | 25 |
| C. Robustness and out-of-sample | 2 | 25 |
| D. Signal quality and consistency | 0 | 20 |
| Measured subtotal | 2 | 75 |
| Normalized raw score | 3 | 100 |
| Applicable caps | fewer than 30 independent events → 40; no meaningful OOS → 55 | |
| Final score | **3** | **100** |

The causal AUC below 0.5, one independent positive episode, low exact-signal precision and lack of clean forward validation provide no evidence for a tradeable prediction edge.

This is research evidence, not investment advice.
