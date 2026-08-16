# Current sell-state vector idea card

Failure mode: The six-dimensional trajectory vector can identify slowing momentum and worsening volatility/drawdown, but it omits the complete canonical bearish-divergence decision state, especially breadth deterioration.

Causal hypothesis: A trajectory vector that is also close to satisfying the frozen canonical bearish-divergence sell rule will rank subsequent 126-session SPX declines of at least 20% better than the six-dimensional vector alone.

Signal change: Diagnostic only. Add one seventh feature, `canonical_sell_vote_fraction`, equal to the fraction of the three frozen divergence conditions currently met: NDX 60-session return at least 3%, breadth 60-session fall at least 20 points, and breadth below 60%. The feature is 0, 1/3, 2/3, or 1. No threshold is tuned.

Entry or exit only: Exit-side diagnostic. Frozen buy and sell execution remain unchanged.

Data available at decision time: Same-day close, current breadth, and observations exactly 60 sessions earlier. All are known at the signal close; no future outcome data enter the feature.

Primary metric: Strict causal peak-similarity AUC on the 61 pre-existing triple-trajectory signal clusters, comparing the frozen six-dimensional vector with the augmented seven-dimensional vector.

Guardrails: No lookahead in features or scaling; historical peak references must already have breached -20% before each query date; no deterioration in causal AUC; conclusions must use independent crash episodes rather than correlated daily rows.

Single threshold and fixed sensitivity values: Frozen sell thresholds only (3%, 20 points, 60% breadth). No new threshold and no grid search.

Expected helpful regimes: Narrowing rallies in which NDX remains firm while broad participation deteriorates before a major SPX decline.

Expected failure regimes: Fast exogenous crashes without prior breadth divergence; healthy mega-cap-led rallies; signals occurring before enough prior crash episodes exist.

Prior related trials counted for DSR: At least 4,589 materially related vector/sell trials documented by the existing research harness, plus this challenger.

Falsification rule: Reject if causal AUC does not exceed the six-dimensional baseline, if any apparent gain is driven only by the single 2020 positive episode, or if actual canonical sell events show poor precision for a subsequent -20% SPX drop.
