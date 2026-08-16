# Pre-registered idea: breadth-only 60-to-80-session extension

Status: pre-registered before breadth-only challenger performance is run.

## Failure mode

The combined NDX-MA200 plus breadth-trajectory gate fixed recent-period Calmar
but remained at 79 points because MA200 filtered the 2008 and 2011 recoveries,
leaving early-period CAGR negative and incremental inference insignificant.
MA200 is a slow price trend measure at bear-market recoveries, while the
strategy's causal edge is explicitly broad participation.

## Causal hypothesis

At session 60 after a breadth washout, a positive breadth trajectory alone is
the more direct test of whether recovery participation is still expanding.
Removing the redundant MA200 requirement should retain early broad recoveries
without re-extending later narrow or deteriorating markets whose breadth is
already falling.

## Signal change

Keep the fixed 70% frozen-breadth / 30% NDX-above-MA200 ensemble and fixed 10%
washout-only TQQQ allocation.  At the session-60 close:

- extend TQQQ to the session-80 close when current breadth is higher than
  breadth 20 sessions earlier;
- otherwise rotate TQQQ into NDX at the following open;
- extended trades rotate at the open after the session-80 close;
- a frozen full exit due at the same open takes priority.

No NDX-versus-MA200 condition participates in the extension decision.

## Entry or exit only

Allocation/risk-management timing only.  Frozen entries, exits, cooldown,
sell priorities, boost size and independent MA200 sleeve are unchanged.

## Data available at decision time

Current and lagged breadth are both known at the session-60 close.  No data
from sessions 61-80 influence the decision.  Every transaction fills at the
following session open.

## Primary metric

Final Backtest Score.  The fixed 20-session breadth-only gate must reach at
least 80 without a cap and pass every guardrail; CAGR alone cannot pass.

## Guardrails

- exact frozen-baseline parity;
- forcing every gate false exactly reproduces the fixed 60-session engine;
  forcing every gate true exactly reproduces the fixed 80-session engine;
- the default combined MA200+breadth path remains unchanged after adding the
  reusable engine flag;
- at least 30 independent exit clusters and component-return correlation below
  0.95;
- all entries, exits and rotations are next-session-open fills with $1
  commission and 0.05% slippage on every leg;
- CAGR, Sharpe and Calmar exceed frozen baseline; maximum drawdown is below 30%
  and no worse than baseline by more than two points;
- paired annual return is positive, HAC p is below 0.05 and the 21-session
  block-bootstrap 95% interval excludes zero;
- CAGR, Sharpe and Calmar deltas are positive in 2002-2013, 2014-present,
  2007+ real-breadth and 2010-02-11+ actual-TQQQ periods;
- paired mean is positive in odd and even calendar years;
- paired annual return remains positive at 5x costs;
- profit factor exceeds 1.2 and expectancy is positive;
- 10/20/40-session breadth lookbacks all have positive paired mean, CAGR
  positive, Sharpe above 1.0 and Calmar above 0.5; adjacent metrics differ by
  less than 20% relatively; score spread is at most five points;
- under 3x proxy drag, paired return remains positive, maximum drawdown remains
  below 30%, and final score remains at least 80.

## Single threshold and fixed sensitivity values

- primary breadth trajectory lookback: 20 sessions;
- sensitivity only: 10 and 40 sessions;
- threshold is the non-tuned sign test `breadth[t] > breadth[t-L]`;
- boost ages remain fixed at 60/80 sessions, boost size at 10%, MA200 sleeve at
  30%; no other value will be tested or selected.

## Expected helpful regimes

Early bear-market recoveries where breadth expands well before NDX recovers its
slow MA200, plus broad post-panic rallies that remain healthy through month four.

## Expected failure regimes

Temporary breadth rebounds inside unfinished bear markets and broad recoveries
that reverse immediately after the day-60 observation.

## Prior related trials counted for DSR

At least 4,602 related trials are counted, including fixed-duration and combined
trajectory gates.  This simplification is not treated as independent discovery.

## Falsification rule

Reject if any guardrail fails.  Do not alter the gate or primary lookback after
viewing results.  A complete historical pass reaches the research-score goal
and supports forward tracking only; it does not modify the frozen baseline or
establish live tradeability without meaningful clean forward OOS evidence.
