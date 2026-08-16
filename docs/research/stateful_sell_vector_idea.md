# Stateful sell vector — idea card

Failure mode: The sell-only 3D vector omits the position path and the state that actually caused each canonical exit, so economically aligned post-exit targets remain unpredictable.

Causal hypothesis: Adding position, exit-mechanism and market-trajectory state at the signal close improves post-exit forecasts relative to both the frozen 3D vector and the expanding naive benchmark.

Signal change: None. This is a diagnostic challenger only.

Stateful vector fixed before results:

- Frozen sell inputs: NDX 60-session return, breadth 60-session fall, current breadth.
- Sell reason one-hot: bearish divergence, climax top, trailing stop.
- Position path: return since entry, maximum gain from entry to trade high, current drawdown from trade high.
- Event ages: sessions since MACD bearish cross and since 10-day extension, each capped at 60.
- Active states: climax active and trailing-stop active.
- Market trajectory: 20-session slopes of NDX 60-session momentum, VIX and SPX 252-session drawdown.

Raw entry price and raw trade high are retained in the audit table but represented in model distance by scale-free returns. This prevents nominal NDX price inflation across decades from dominating similarity.

Entry or exit only: Exit-side diagnostic. Frozen entries, exits, cooldown and execution remain unchanged.

Targets: Identical to the preceding pre-registered economic-target test: future minimum NDX return over 60 sessions, 20/60-session risk-adjusted return, 20/60-session cash advantage and cash-win indicator, and 60-session oracle re-entry delay.

Primary metric: Stateful-vector MAE for future minimum 60-session NDX return versus both the 3D-vector MAE and expanding historical-median MAE.

Guardrails: Exact same events, targets, three-neighbour rule, causal label availability, costs and naive forecasts as the 3D test; stateful primary must beat both comparators in full, early and late periods; at least four of seven secondary targets must improve versus 3D and naive.

Fixed values: Three neighbours, minimum four eligible prior exits, inverse-distance weight `1/(distance + 0.25)`, event-age cap 60, 20/60-session horizons and 0.10% round-trip slippage. No grid search.

Expected helpful regimes: Separating climax, divergence and trailing-stop exits and distinguishing fresh deterioration from stale signals with similar static breadth values.

Expected failure regimes: Sixteen dimensions with only 17 exits, unstable distances, nominal-state reconstruction errors and structural regime change.

Prior related trials counted for DSR: At least 4,592 materially related vector/sell investigations including the sell-only economic-target test.

Falsification rule: Reject unless the primary target beats both comparators in the full, early and late samples and at least four of seven secondary targets also beat both comparators.
