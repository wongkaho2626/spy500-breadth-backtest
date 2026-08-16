# Sell-signal-only 3D vector idea card

Failure mode: The previous challenger mixed the frozen sell state with six trajectory features, so any apparent predictive ranking could not be attributed to the sell signal alone.

Causal hypothesis: The three continuous inputs of the frozen bearish-divergence rule contain enough information by themselves to rank subsequent 126-session SPX declines of at least 20%.

Signal change: Diagnostic only. Use exactly three dimensions: NDX 60-session return, S&P 500 breadth 60-session fall in percentage points, and current breadth level. Do not use the prior six trajectory features.

Entry or exit only: Exit-side diagnostic. The frozen strategy is unchanged.

Data available at decision time: Same-day NDX close and breadth plus values exactly 60 sessions earlier.

Primary metric: Strict causal peak-similarity AUC on the existing triple-trajectory signal clusters for which a previously confirmed, breadth-comparable peak episode is available.

Guardrails: No future scaling or peak references, no new threshold, report actual canonical bearish-divergence precision, and count independent crash episodes rather than daily rows.

Single threshold and fixed sensitivity values: Existing frozen 3% / 20-point / 60% thresholds are descriptive only. No threshold or neighbor count is tuned.

Expected helpful regimes: Narrowing rallies with positive NDX momentum, sharply deteriorating breadth, and breadth already below 60%.

Expected failure regimes: Fast external shocks, trailing-stop exits after a crash has started, and momentum reversals without breadth deterioration.

Prior related trials counted for DSR: At least 4,590 materially related vector/sell trials plus this isolated diagnostic.

Falsification rule: Reject as a predictor if causal AUC is at or below 0.5, if the positive class still contains only one independent crash episode, or if exact bearish-divergence precision remains too low to support a calibrated crash probability.
