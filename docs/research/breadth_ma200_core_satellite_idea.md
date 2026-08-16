# Pre-registered idea: breadth + MA200 core-satellite ensemble

Failure mode:
The frozen breadth strategy has strong absolute performance but only 17
completed round trips, creating a 40-point sample-size cap.  Forcing the same
strategy to exit more often raised the trade count but destroyed expectancy.
The sparse crash-timing component therefore needs an orthogonal decision
stream rather than more exits inside the same state machine.

Causal hypothesis:
A small, independently compounded MA200 trend sleeve can contribute many
separate trend cycles and participate during periods when the breadth strategy
is flat, while the 70% breadth sleeve retains the original high-expectancy
washout/divergence edge.  Combining two simple mechanisms may improve
consistency and reduce dependence on a few crisis calls.

Signal change:
No frozen signal changes.  Allocate 70% of initial capital to the canonical
breadth strategy and 30% to an independent NDX MA200 sleeve.  The MA200 sleeve
is invested when the NDX closes above its trailing 200-session average and is
in cash below it.  Every transition fills at the next session open.  Buckets
compound independently with no rebalancing.

Entry or exit only:
Portfolio-allocation hypothesis rather than a modification of the frozen
entry or exit rules.  The added sleeve is one symmetric trend-state rule;
the breadth component remains byte-for-byte unchanged.

Data available at decision time:
NDX closes and the trailing 200-session average through the signal close.
No future bar, breadth revision, or constituent data is required.

Primary metric:
Final Backtest Score under the installed `backtest-analyst` rubric.  The
challenger must reach at least 80 without a fatal cap.  Calmar and Sharpe are
the economic co-primary metrics.

Guardrails:
- exact canonical parity at a 0% MA200 sleeve;
- close signals and next-session-open fills on both buckets;
- commission and slippage on every component transaction;
- at least 30 independent exit-event clusters after combining component
  exits within 21 sessions;
- component daily-return correlation below 0.95;
- full-period Sharpe and Calmar both improve;
- maximum drawdown does not worsen and CAGR is no more than 2 percentage
  points below baseline;
- positive Sharpe and Calmar deltas in both 2002-2013 and 2014-present;
- 2007+ real-breadth direction agrees;
- positive paired mean-return difference at 5x costs;
- 20%/30%/40% initial sleeve sensitivity is smooth and directionally stable;
- clustered profit factor above 1.2 and positive clustered expectancy.

Single threshold and fixed sensitivity values:
- primary initial MA200 sleeve: 30%;
- sensitivity only: 20% and 40%;
- moving-average window: conventional 200 sessions, fixed;
- exit-event independence cluster: 21 sessions, fixed before results.

Expected helpful regimes:
Long trends following false breadth exits, recoveries that do not immediately
produce a washout, and quiet bull markets where the frozen strategy is flat.

Expected failure regimes:
Sideways MA200 whipsaws, simultaneous losses in both sleeves, and periods when
the lower-Sharpe MA200 system dilutes the breadth strategy without meaningful
diversification.

Prior related trials counted for DSR:
At least 4,596 related signal/vector/allocation trials, including the rejected
standalone MA200 and 12-month momentum families, allocation grids, TQQQ
core-satellite work, washout boost, and monthly regime-exit audit.  The exact
independent 70/30 breadth-plus-MA200 ensemble was not found, but it receives
the full multiplicity penalty.

Falsification rule:
Reject if parity/timing/cost accounting fails; fewer than 30 clustered events
remain; correlation is at least 0.95; final score is below 80; either Sharpe
or Calmar fails to improve; maximum drawdown worsens; CAGR falls more than 2
points; either historical half reverses; 2007+ reverses; 5x costs erase the
paired benefit; sensitivity is cliff-edge; or clustered expectancy/profit
factor fails.  Even a historical pass can only be tracked until meaningful
clean post-2026-07-05 evidence exists.
