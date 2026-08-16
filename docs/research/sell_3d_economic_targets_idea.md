# Sell-only 3D vector with economic targets — idea card

Failure mode: A binary target of a 20% SPX decline is rare and poorly aligned with how the frozen QQQ strategy creates value after an exit.

Causal hypothesis: At canonical sell decisions, the three continuous bearish-divergence inputs can forecast economically aligned post-sell outcomes better than an expanding historical naive benchmark.

Signal change: None. Diagnostic only. Use exactly `[NDX 60-session return, breadth 60-session fall, current breadth]` at the signal close.

Targets fixed before model results:

1. Minimum NDX return over the next 60 sessions from the next-open execution price.
2. Realized terminal return divided by realized path volatility over 20 and 60 sessions.
3. Whether cash beats holding NDX over 20 and 60 sessions after a fixed 0.10% round-trip slippage allowance.
4. Oracle re-entry delay: sessions from the sell execution to the lowest NDX close in the following 60 sessions.

Entry or exit only: Exit-side diagnostics; frozen entry, exit and re-entry rules are unchanged.

Data available at decision time: Same-day NDX close and breadth plus observations exactly 60 sessions earlier.

Primary metric: Model-versus-naive MAE improvement for future minimum 60-session NDX return. Secondary metrics are risk-adjusted-return MAE, cash-vs-hold Brier score, and re-entry-delay MAE.

Guardrails: Expanding causal training only; a historical event becomes trainable only after its complete 60-session target window ends; median/IQR scaling fit on training events only; same targets and costs for model and naive benchmark; no strategy performance claim.

Single threshold and fixed sensitivity values: Three nearest prior eligible exits, inverse-distance weight `1/(distance + 0.25)`, minimum four eligible training exits, horizons 20 and 60, and 0.10% round-trip slippage. No grid search or sensitivity selection.

Expected helpful regimes: The vector distinguishes breadth-led topping exits from healthy corrections and identifies when waiting in cash has value.

Expected failure regimes: Climax/trailing exits driven by state absent from the 3D vector, structural regime changes, and a sample too small for stable nearest neighbours.

Prior related trials counted for DSR: At least 4,591 related vector/sell investigations including the preceding sell-only 3D crash test.

Falsification rule: Reject if the primary 60-session adverse-return MAE does not beat the expanding historical median, or if any secondary improvement is inconsistent across targets and supported by fewer than 10 causal predictions.
