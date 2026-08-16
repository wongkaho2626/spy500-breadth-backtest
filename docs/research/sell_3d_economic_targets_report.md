# Sell-only 3D vector with economic targets

## Decision: Reject

Changing the target from a rare 20% crash to economically aligned post-sell outcomes does not rescue the three-dimensional sell vector. The primary 60-session adverse-return MAE improves by only 0.16% versus an expanding historical median, has near-zero rank correlation, and reverses from a 7.01% improvement in the early period to a 7.43% deterioration from 2014 onward. None of seven secondary targets beats its naive benchmark.

## Design

Vector at each canonical signal close:

`[NDX 60-session return, breadth 60-session fall, current breadth]`

Targets measured from the canonical next-open execution price:

1. Lowest NDX return during the following 60 sessions.
2. Terminal return divided by realized path volatility over 20 and 60 sessions.
3. Cash advantage and whether cash beats holding over 20 and 60 sessions, after a fixed 0.10% round-trip slippage allowance.
4. Oracle re-entry delay to the lowest NDX close in the following 60 sessions.

The model uses three nearest prior exits. Scaling is fitted only on prior eligible events, and an event cannot enter training until its entire 60-session outcome window has ended. Continuous targets are compared with an expanding historical median; binary targets are compared with an expanding historical base rate. No parameter was tuned.

## Data and baseline integrity

- Canonical exits: 17.
- Causal predictions after the minimum four-event training period: 13.
- Exit reasons: nine bearish-divergence, seven climax-top, one trailing-stop.
- Strategy rules, next-open execution, cooldown, commissions and slippage are unchanged.
- No completed clean forward-OOS exit exists after the 2026-07-05 freeze.
- `qqq_backtest.py` and the frozen parameter record were not modified.

## Results

### Primary target: future minimum NDX return over 60 sessions

| Metric | 3D nearest-neighbour | Naive median |
|---|---:|---:|
| MAE | 7.376% | 7.388% |
| Relative improvement | 0.16% | — |
| Spearman correlation with actual outcome | 0.049 | — |

The 0.012 percentage-point MAE difference is economically negligible.

| Historical period | Model MAE | Naive MAE | Model improvement |
|---|---:|---:|---:|
| Through 2013 | 7.819% | 8.409% | +7.01% |
| From 2014 | 6.997% | 6.513% | -7.43% |

The direction reversal fails the required stability test.

### Other continuous targets

| Target | Model MAE | Naive MAE | Improvement | Spearman |
|---|---:|---:|---:|---:|
| 20-session risk-adjusted return | 0.962 | 0.776 | -23.99% | -0.225 |
| 60-session risk-adjusted return | 0.891 | 0.769 | -15.87% | -0.500 |
| Cash advantage over 20 sessions | 8.98% | 6.77% | -32.53% | -0.341 |
| Cash advantage over 60 sessions | 12.99% | 10.53% | -23.30% | -0.599 |
| Oracle re-entry delay | 20.65 sessions | 20.42 sessions | -1.09% | -0.275 |

Every secondary continuous target is worse than the simple historical median, and all rank correlations are negative.

### Cash versus hold classification

| Target | Model Brier | Naive Brier | Improvement | Model accuracy | Naive accuracy |
|---|---:|---:|---:|---:|---:|
| Cash wins over 20 sessions | 0.328 | 0.280 | -17.29% | 53.85% | 53.85% |
| Cash wins over 60 sessions | 0.460 | 0.257 | -79.28% | 30.77% | 46.15% |

The vector probabilities are less calibrated than the expanding base rate, especially over 60 sessions.

## Current state — 2026-07-30

Current 3D vector:

`[+1.6438%, -14.94 breadth points, 68.41% breadth]`

No bearish-divergence, climax-top or trailing-stop signal is active. Applying a post-sell model is therefore outside its intended domain.

For completeness, the failed model's nearest analogues are 2003-03-17, 2022-11-11 and 2002-11-11. Its counterfactual estimates if a sale were forced today are:

- Lowest NDX return during 60 sessions: approximately -5.87%.
- Cash-win probability: 77.68% over 20 sessions and 22.32% over 60 sessions.
- Expected cash advantage: approximately -0.06% over 20 sessions and -7.75% over 60 sessions.
- Oracle re-entry delay: approximately 27 sessions.

These estimates should not be acted upon. The conflicting 20-session probability and expected advantage, plus failed causal validation, show that the small-neighbour forecast is unstable.

## Bias and robustness audit

| Check | Status | Evidence |
|---|---|---|
| Feature lookahead | Absent | Vector uses current and 60-session-lagged data only. |
| Target leakage | Absent | Training event admitted only after its 60-session label window ends. |
| Signal/fill alignment | Correct | Outcomes start from canonical next-session-open execution. |
| Historical stability | Failed | Primary improvement changes sign after 2014. |
| Naive benchmark | Failed | Zero of seven secondary targets beats its naive forecast. |
| Independent sample size | Inadequate | Only 13 causal predictions from 17 exits. |
| Multiple testing | High risk | More than 4,591 related investigations precede this test. |
| Clean forward OOS | Missing | No completed post-freeze exit. |
| Cost model | Partial | Fixed 0.10% slippage included; dollar commission and cash yield omitted. |
| Re-entry target | Hindsight label only | The 60-session trough is valid for training evaluation, not known live. |

## Predictive evidence score: 8 / 100 — Reject

This score evaluates the new three-dimensional prediction claim, not the profitability of the frozen QQQ strategy.

| Component | Score | Max |
|---|---:|---:|
| A. Statistical validity and significance | 0 | 30 |
| B. Risk-adjusted trading performance | unavailable | 25 |
| C. Robustness and out-of-sample | 4 | 25 |
| D. Prediction quality and consistency | 2 | 20 |
| Measured subtotal | 6 | 75 |
| Normalized raw score | 8 | 100 |
| Applicable cap | fewer than 30 independent events → 40; no meaningful OOS → 55 | |
| Final score | **8** | **100** |

## Interpretation

The revised targets are economically better aligned, but the three features are still insufficient. They describe only the bearish-divergence inputs and omit the state that distinguishes the other exit mechanisms: sell reason, entry date, trade-high drawdown, MACD-cross age, extension age, cooldown and market path.

The appropriate conclusion for this research round is to reject the sell-only 3D predictor. A separate future round could test a stateful post-exit vector against the same frozen economic targets, without changing them after seeing these results.

This is research evidence, not investment advice.
