# Stateful sell vector report

## Decision: Reject, with useful secondary evidence

The requested position, exit and trajectory state was added to the sell vector. Four of seven secondary targets improve versus both the previous 3D vector and the naive benchmark. However, the pre-registered primary target—future minimum NDX return over 60 sessions—becomes worse than both comparators and fails the early/late guardrail. The challenger is therefore rejected as a complete predictor.

## Vector construction

The model uses 16 robust-scaled dimensions:

- Sell inputs: NDX 60-session return, breadth 60-session fall, current breadth.
- Sell reason: bearish-divergence, climax-top and trailing-stop one-hot values.
- Position path: return since entry, maximum gain from entry to trade high, current drawdown from trade high.
- Event age: sessions since MACD bearish cross and extension event, capped at 60.
- Active state: climax and trailing-stop flags.
- Trajectory: 20-session slopes of NDX 60-session momentum, VIX and SPX 252-session drawdown.

Raw entry price and trade high are retained in the event audit CSV. They are converted to relative returns for distance calculation because nominal NDX levels across 2002–2026 are not comparable.

## Validation design

- Same 17 canonical exits and same economic targets as the frozen 3D test.
- 13 expanding causal predictions after the minimum four-event training period.
- Historical event admitted only after its full 60-session target window ended.
- Fixed three nearest neighbours; no grid search or tuned threshold.
- Comparators: the previous sell-only 3D vector and expanding median/base-rate forecasts.
- Required to pass: primary beats both comparators in full, early and late periods, plus at least four of seven secondary targets beat both.
- Frozen QQQ trading rules, costs and execution remain unchanged.

## Primary target: future minimum NDX return over 60 sessions

| Model | MAE | Versus naive | Spearman |
|---|---:|---:|---:|
| Stateful 16D | 7.448% | -0.82% | 0.005 |
| Sell-only 3D | 7.376% | +0.16% | 0.049 |
| Naive median | 7.388% | — | — |

Stateful MAE is 0.98% worse than the 3D model on a relative basis and 0.82% worse than naive. Its rank correlation is effectively zero.

| Period | Stateful MAE | 3D MAE | Naive MAE | Result |
|---|---:|---:|---:|---|
| Through 2013 | 8.193% | 7.819% | 8.409% | Beats naive, loses to 3D |
| From 2014 | 6.810% | 6.997% | 6.513% | Beats 3D, loses to naive |

It never beats both comparators in either period, so the primary guardrail fails.

## Secondary targets

| Target | Stateful result vs 3D | Stateful result vs naive | Pass both? |
|---|---:|---:|---|
| 20-session risk-adjusted return MAE | -1.56% | -25.92% | No |
| 60-session risk-adjusted return MAE | +18.36% | +5.41% | Yes |
| 20-session cash-advantage MAE | +28.39% | +5.10% | Yes |
| 60-session cash-advantage MAE | +26.50% | +9.38% | Yes |
| Re-entry-delay MAE | +18.33% | +17.44% | Yes |
| 20-session cash-win Brier | +13.14% | -1.88% | No |
| 60-session cash-win Brier | +27.98% | -29.12% | No |

The stateful information clearly helps estimate the size of cash advantage, 60-session risk-adjusted return and approximate re-entry delay. It does not calibrate the binary cash decision, and it does not improve the primary adverse-return forecast.

This split matters: regression targets retain information about magnitude, while converting the same outcomes into cash-win yes/no labels loses information in a sample of only 13 predictions.

## Current state — 2026-07-30

Important position values:

- Entry price: 16,771.77.
- Trade high: 30,660.60.
- Return since entry at the current close: +67.58%.
- Maximum gain from entry: +82.81%.
- Current drawdown from trade high: -8.33%.
- MACD-cross age: 38 sessions.
- Extension age: 71 sessions, represented by the fixed 60-session cap.
- Climax and trailing-stop states: inactive.
- NDX momentum slope: -1.1060.
- VIX slope: approximately zero.
- SPX drawdown slope: -0.1052.
- No canonical sell reason is active, so all historical sell-reason one-hot values are zero for the current query.

The current query is outside the model's training domain because every historical row is an actual sell event while today is not. Its three nearest historical exits are 2024-12-18, 2014-08-07 and 2018-03-22, but their standardized distances are large at 3.76–4.32.

If a sale were forced today, the rejected model estimates a 60-session minimum return near -5.78%, negative cash advantage of about -2.30% over 20 sessions and -2.56% over 60 sessions, and an oracle re-entry delay near 37 sessions. These values are not actionable because the primary model failed and there is no active sell signal.

## Bias and robustness audit

| Check | Status | Evidence |
|---|---|---|
| Feature lookahead | Absent | State is reconstructed only through the signal close. |
| Target leakage | Absent | Complete 60-session label required before a prior exit enters training. |
| Raw price non-stationarity | Controlled | Raw prices logged, relative path returns used in distance. |
| Signal/fill alignment | Correct | Outcomes start from canonical next-session-open execution. |
| Dimensionality | Severe risk | 16 dimensions versus 13 causal predictions. |
| Primary stability | Failed | Cannot beat both comparators in full, early or late samples. |
| Secondary breadth | Partial pass | Four of seven targets beat both comparators. |
| Multiple testing | High risk | At least 4,593 related investigations including this challenger. |
| Clean forward OOS | Missing | No completed post-freeze exit. |
| Current applicability | Failed | Current row has no sell reason and lies outside historical sell-event support. |

## Predictive evidence score: 24 / 100 — Reject

This score evaluates the stateful prediction claim, not the frozen QQQ strategy.

| Component | Score | Max |
|---|---:|---:|
| A. Statistical validity and significance | 2 | 30 |
| B. Risk-adjusted trading performance | unavailable | 25 |
| C. Robustness and out-of-sample | 8 | 25 |
| D. Prediction quality and consistency | 8 | 20 |
| Measured subtotal | 18 | 75 |
| Normalized raw score | 24 | 100 |
| Applicable cap | fewer than 30 independent events → 40; no meaningful OOS → 55 | |
| Final score | **24** | **100** |

## Interpretation

The added state is not useless: it materially improves four continuous economic targets. The complete 16-dimensional nearest-neighbour model still fails because the sample is far smaller than the feature space and the primary target remains unpredictable.

Do not add this vector to the frozen strategy. The defensible next research step would be a new, separately pre-registered dimensionality-reduction round that keeps only the state groups supported here—position path, exit mechanism and re-entry timing—rather than another threshold search.

This is research evidence, not investment advice.
