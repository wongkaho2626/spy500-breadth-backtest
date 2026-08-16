# Current sell-state vector verification report

## Decision: Reject

Adding the frozen canonical bearish-divergence state to the trajectory vector raises strict causal AUC from 0.615 to 0.696, but all three positive signal clusters belong to the same 2020 crash. This fails the pre-registered independent-episode rule. The feature has weak historical ranking information, but it cannot support a calibrated prediction of a future 20% SPX decline.

## Idea tested

The existing six-dimensional vector is:

`[NDX 60d return, NDX slope, VIX, VIX slope, SPX 252d drawdown, drawdown slope]`

The challenger adds one feature:

`canonical_sell_vote_fraction = conditions met / 3`

The three frozen conditions are NDX 60-session return at least 3%, breadth falling at least 20 points over 60 sessions, and current breadth below 60%. The feature can only be 0, 1/3, 2/3, or 1. No new threshold was selected.

The pre-registered idea card is in `docs/research/current_sell_state_vector_idea.md`.

## Baseline parity and experiment design

- `qqq_backtest.py` and `docs/frozen_params_2026-07-05.md` were not changed.
- The diagnostic calls the canonical engine directly with next-session-open execution, $1 commission, 0.05% slippage, 15-day cooldown, and the same 17 completed trades.
- Existing independent parity evidence remains exact: maximum equity difference 0, identical trade signatures, and identical open trade in `qqq_triple_trajectory_sell_results.json`.
- Target: minimum SPX return over the next 126 sessions less than or equal to -20%.
- Evaluation set: 61 pre-existing triple-trajectory signal clusters; 48 have a comparable strict causal breadth/peak reference.
- Causal scaling uses only observations before each query. A peak episode can be a reference only after its -20% breach had already occurred.
- The 2000 peak cannot be used in the seven-dimensional comparison because comparable breadth data do not exist.

## Predictor comparison

| Metric | 6D baseline | 7D sell-state vector | Difference |
|---|---:|---:|---:|
| Strict causal AUC | 0.615 | 0.696 | +0.081 |
| True-cluster median similarity | 84.42% | 82.70% | -1.72 pts |
| False-cluster median similarity | 73.92% | 65.91% | -8.01 pts |
| Causal clusters | 48 | 48 | — |
| Positive clusters | 3 | 3 | — |
| Independent positive crash episodes | 1 | 1 | — |

The seventh feature helps mainly by pushing some false signals farther from historical peaks. The apparent AUC improvement cannot be separated from the single 2020 episode, so no reliable confidence interval or crash-episode bootstrap can be estimated.

## What the actual canonical sell signals predicted

| Canonical exit reason | Signals | Followed by -20% SPX drop | Raw rate | Jeffreys posterior | 90% interval |
|---|---:|---:|---:|---:|---:|
| Bearish divergence | 9 | 1 | 11.11% | 15.00% | 1.99%–36.07% |
| Climax top | 7 | 2 | 28.57% | 31.25% | 8.81%–59.29% |
| Trailing stop | 1 | 1 | 100.00% | 75.00% | 22.85%–99.85% |
| All canonical exits | 17 | 4 | 23.53% | 25.00% | 10.37%–42.86% |

The trailing-stop row has only one event and is not meaningful evidence. More importantly, the exact bearish-divergence sell signal only preceded one 20% decline in nine occurrences. It is an exit/risk-management rule, not a high-precision crash predictor.

## Current state — 2026-07-30

- Canonical sell vote: 0/3, or 0.0.
- NDX 60-day price condition: not met.
- Breadth 60-day deterioration condition: not met.
- Breadth below 60 condition: not met.
- Bearish divergence, climax top, and trailing stop: all inactive.
- Drawdown from the current trade high: -8.33%, versus the frozen -25% trailing-stop level.
- Six-dimensional causal peak similarity: 22.99%.
- Seven-dimensional causal peak similarity: 21.75%.
- Nearest eligible peak-zone row: 2022-01-12, but the low percentile means the current vector is not especially peak-like.

The current sell state therefore does not predict an imminent 20% decline. The 21.75% value is a similarity percentile, not a 21.75% crash probability.

## Bias and robustness assessment

| Check | Status | Evidence |
|---|---|---|
| Lookahead in features | Absent | All three sell votes use the current close and exactly 60 prior sessions. |
| Lookahead in causal references | Absent | A peak episode becomes eligible only after its historical -20% breach. |
| Signal/fill mismatch | Absent | Strategy performance remains next-session-open; this diagnostic does not create trades. |
| Survivorship bias | Cannot fully verify | NDX and VIX are index series; pre-2007 breadth is a synthetic splice. |
| Data snooping | Present as a major risk | At least 4,590 related trials and only one independent positive crash episode. |
| Clean forward OOS | Insufficient | Only data after 2026-07-05 are clean OOS; no completed forward trade exists. |
| Parameter sensitivity | Not applicable | No new threshold was tuned. |
| Episode bootstrap | Unavailable | One independent positive episode cannot be resampled meaningfully. |
| Cost stress | Not applicable | The vector is diagnostic-only and makes no new trades. |

## Predictive evidence score: 5 / 100 — Reject

This score concerns the new crash-prediction claim, not the profitability of the frozen QQQ strategy.

| Component | Score | Max |
|---|---:|---:|
| A. Statistical validity and significance | 0 | 30 |
| B. Risk-adjusted trading performance | unavailable | 25 |
| C. Robustness and out-of-sample | 4 | 25 |
| D. Signal quality and consistency | 0 | 20 |
| Measured subtotal | 4 | 75 |
| Normalized raw score | 5 | 100 |
| Caps applicable | fewer than 30 independent events → 40; no meaningful OOS → 55 | |
| Final score | **5** | **100** |

Statistical significance and DSR cannot support the challenger because the effective positive sample is one episode and thousands of related variants have already been examined. The deterministic no-threshold construction earns limited robustness credit, but it does not offset the missing independent events.

## Final interpretation

The canonical sell state adds some descriptive breadth information to the vector, but it does not turn the vector into a dependable drop predictor. Keep the frozen sell rule as an execution/risk-control mechanism; do not use the augmented vector to quote a crash probability or introduce a new sell threshold. Meaningful validation requires additional independent crash episodes or a genuinely untouched cross-market panel tested without retuning.

This is research evidence, not investment advice.
