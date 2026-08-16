# Twenty-session trajectory vector for MA200 trend re-entry

Failure mode:
The six-feature vector treats the current market as a static snapshot. Two
dates can have similar VIX, breadth, and drawdown levels while one is improving
and the other is deteriorating. The prior 60% six-feature recross filter
rejected all five baseline recross entries and reduced full-period Calmar.

Causal hypothesis:
Adding the twenty-session linear slopes of VIX, breadth, and SPX drawdown will
distinguish improving recoveries from deteriorating regimes and make the fixed
60% vector filter more selective in the correct direction.

Signal change:
Keep the canonical washout entry exactly unchanged. Permit an otherwise-valid
MA200-recross entry only when a nine-feature expanding-history vector buy
probability is at least 60%. The added features are:

- VIX twenty-session OLS slope in points per session;
- breadth twenty-session OLS slope in percentage points per session;
- SPX trailing-252-session drawdown twenty-session OLS slope in percentage
  points per session.

Entry or exit only:
Entry only. Every canonical sell rule remains unchanged.

Data available at decision time:
Each slope uses the signal close and the preceding nineteen completed sessions.
No future observation is used. Training labels become eligible only after the
complete 126-session outcome path is historical.

Primary metric:
Calmar ratio versus the frozen baseline. Secondary comparisons are Sharpe and
the already-tested six-feature recross challenger.

Guardrails:
- Exact baseline parity and all-true filter parity must pass.
- No lookahead: close signal, next-session-open fill.
- Washout entries and all sell rules remain unchanged.
- Maximum drawdown must not worsen.
- CAGR may not trail the frozen baseline by more than 2 percentage points.
- Completed trades and turnover may not increase.
- Challenger-minus-baseline CAGR must not reverse sign between 2002-2013 and
  2014-present historical splits.
- The challenger must retain a positive CAGR advantage at five times modeled
  costs.

Single threshold and fixed sensitivity values:
Primary probability threshold 60%; sensitivity thresholds 50% and 70%.
Trajectory window fixed at twenty sessions. No window, feature-subset, slope
normalisation, acceleration, or threshold search is permitted this round.

Expected helpful regimes:
Recoveries where breadth is rising, drawdown is shrinking, and VIX is falling,
even if the current static levels resemble prior weak markets.

Expected failure regimes:
Abrupt V-shaped reversals that occur before a twenty-session trend is visible,
noisy sideways regimes, and early observations with synthetic breadth.

Prior related trials counted for DSR:
At least 573 related vector configurations: 567 crash-exit grid variants,
three vector-only buy thresholds, and three static-vector recross thresholds.

Falsification rule:
Reject if primary Calmar does not exceed the frozen baseline, any guardrail
fails, the historical split effect reverses, or the 50%/60%/70% results show
that any apparent benefit depends on a single threshold. With insufficient
clean forward evidence, a historical pass remains research-only.
