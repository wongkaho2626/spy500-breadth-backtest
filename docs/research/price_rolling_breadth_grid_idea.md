# Price-confirmed rolling breadth exit grid — pre-registered search card

## Failure mode

The single 3% / 20-point / 60% price-confirmed rolling-breadth challenger
underperformed, but one fixed combination does not show whether that failure is
broad or caused by a poorly balanced confirmation threshold.

## Causal hypothesis

Price confirmation, breadth drawdown severity, and the current breadth cap have
an economically coherent interaction: a milder breadth decline may require a
stricter cap or stronger price confirmation, while a severe breadth collapse may
remain informative at a looser cap.  A stable region, rather than an isolated
best cell, would be evidence worth forward tracking.

## Signal family

Replace only the frozen bearish-divergence exit with:

`NDX 60-session return >= P`

and

`max(breadth[t-59:t]) - breadth[t] >= D points`

and

`breadth[t] < C%`.

Signals use the close and fill at the next session open.  Entries, cooldown,
costs, climax-top, and trailing-stop logic remain frozen.

## Fixed grid

- Price-rise threshold `P`: 0%, 1%, 2%, 3%, 4%, 5%, 6%.
- Rolling breadth-drawdown threshold `D`: 10, 15, 20, 25, 30, 35 points.
- Current breadth cap `C`: 30%, 40%, 50%, 60%, 70%.
- Rolling and price lookbacks: fixed at 60 sessions.
- Total attempted variants: 7 × 6 × 5 = 210.

The grid will not be expanded or refined after results are observed in this round.

## Selection objectives

1. **Robust consensus candidate (primary):** maximise the smaller of early
   (2002-2013) and late (2014-latest) Calmar; break ties by full-period Calmar,
   then lower completed-trade count.
2. **Early-selected walk-forward candidate:** maximise early Calmar and report
   late performance without reselection.
3. **Late-selected reverse-direction candidate:** maximise late Calmar and report
   early performance without reselection.
4. **Full-sample winner:** maximise full-period Calmar; descriptive only and
   never sufficient for promotion.

## Primary metric

Minimum historical-half Calmar for selection; full-period Calmar for the paired
baseline-versus-selected-candidate decision.

## Guardrails for the robust candidate

- exact baseline parity under the replacement harness;
- full-period Calmar must exceed baseline and maximum drawdown must not worsen;
- CAGR may not fall more than two percentage points below baseline;
- expectancy may not fall below baseline;
- both half-period Calmar deltas and 2007+ real-breadth Calmar delta must be
  non-negative;
- paired annual return difference must remain positive at 5x costs;
- candidate must not sit on any grid boundary;
- at least 75% of immediate one-step neighbours must retain at least 90% of the
  candidate's full-period Calmar and have positive Calmar in both halves.

## Expected helpful regimes

Mature advances with rising NDX and broad internal deterioration, while avoiding
the widespread whipsaw produced by breadth-only rolling exits.

## Expected failure regimes

Narrow but persistent mega-cap leadership, temporary breadth rotation, and
parameter cells that win only because they suppress nearly every replacement exit.

## Multiple testing

The DSR trial count is 4,816: 4,606 prior related trials plus all 210 cells in
this grid.  Historical halves are pseudo-OOS because the family and grid are now
being examined with data through the latest observation.

## Falsification rule

Reject the family for adoption if parity fails, the robust candidate misses any
guardrail, forward/reverse selections disagree materially, the winner lies on a
boundary, local neighbours are unstable, costs erase the result, or clean
post-2026-07-05 evidence is insufficient.  Still report the requested best cells
even when the family is rejected.
