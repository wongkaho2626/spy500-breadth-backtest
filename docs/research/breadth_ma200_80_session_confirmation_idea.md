# Pre-registered confirmation: fixed 80-session washout boost

Status: confirmatory audit registered after the 80-session rule scored 83 in a
previous sensitivity table.  The full-history score is therefore selected and
not independent evidence.  This round cannot claim blind discovery; its new
evidence is the previously unviewed 100-session neighbour, actual-TQQQ period,
calendar-parity subsets and stricter combined guardrails.

## Failure mode

The 60-session primary rule improved risk-adjusted performance and reduced
maximum drawdown below 30%, but its incremental block-bootstrap interval still
included zero.  The pre-registered 80-session sensitivity retained exposure to
the rebound slightly longer and crossed the 80-point threshold, but sensitivity
rows cannot be promoted after results are viewed without a separate selection
penalty and confirmation audit.

## Causal hypothesis

A four-month rather than three-month leveraged rebound window may capture the
later part of post-washout recovery while still rotating out of TQQQ long before
multi-year frozen breadth trades end.  If this is structural rather than a
single-row selection artifact, the fixed 80-session rule should remain robust
in adjacent horizons, both historical directions, calendar-parity subsets and
the post-inception actual-TQQQ sample.

## Signal change

Fix the previous architecture without alteration: 70% frozen breadth timing,
30% NDX-above-MA200 timing, and a 10-percentage-point TQQQ allocation only on
frozen washout entries.  On the 80th trading-session close after a leveraged
fill, schedule a next-session-open TQQQ sale and NDX purchase.  Hold NDX until
the unchanged frozen breadth exit.  MA200-recross entries never use TQQQ.

## Entry or exit only

Allocation/risk-management timing only.  Frozen entry dates, exit dates,
cooldown, sell priorities and the independent 30% MA200 sleeve are unchanged.

## Data available at decision time

The washout label is known at its signal close.  Boost age uses only elapsed
sessions.  Every order fills at the following session open.  Actual TQQQ is
used from 2010-02-11; the earlier proxy and its later-overlap calibration are
explicitly penalised with 1x and 3x drag results.

## Primary metric

Final Backtest Score for the fixed 80-session rule.  It must remain at least 80
without a hard cap and pass every guardrail.  The previously observed 83 is
treated as selected historical evidence, not a fresh p-value.

## Guardrails

- exact frozen-baseline parity and exact parity to the prior timed engine for
  the same fixed 80-session configuration;
- at least 30 independent exit clusters after grouping component exits within
  21 sessions, and component-return correlation below 0.95;
- all original entries/exits and rotations use close information and fill at
  the next open, with $1 commission and 0.05% slippage on every leg;
- CAGR, Sharpe and Calmar exceed the frozen baseline; maximum drawdown is below
  30% and no worse than the frozen baseline by more than two points;
- paired annual return is positive, HAC p is below 0.05 and the 21-session
  block-bootstrap 95% interval excludes zero;
- CAGR, Sharpe and Calmar deltas are positive in 2002-2013, 2014-present,
  2007+ real-breadth and 2010-02-11+ actual-TQQQ periods;
- challenger-minus-baseline mean daily return is positive in both odd calendar
  years and even calendar years;
- paired annual return remains positive at 5x transaction costs;
- profit factor exceeds 1.2 and expectancy is positive;
- the fixed 60/80/100-session family has positive paired mean at every point;
  each has CAGR positive, Sharpe above 1.0 and Calmar above 0.5; adjacent CAGR,
  Sharpe and Calmar values differ by less than 20% relatively; final-score
  spread is at most five points; the 80-session rule is not replaced by 100;
- under 3x pre-inception TQQQ drag, paired return stays positive, maximum
  drawdown stays below 30%, and final score remains at least 80.

## Single threshold and fixed sensitivity values

- primary maximum boost age: fixed 80 sessions, selected from the previous
  pre-registered sensitivity table;
- sensitivity: 60 sessions (previously viewed) and 100 sessions (new); no other
  duration will be run or selected;
- washout boost remains fixed at 10%; MA200 sleeve remains fixed at 30%.

## Expected helpful regimes

Post-panic rebounds that continue through months three and four before the
benefit of long-lived leverage decays.

## Expected failure regimes

Rapid rebounds completed before day 60, renewed declines during days 61-80,
and slow recoveries whose leverage premium arrives after day 80.

## Prior related trials counted for DSR

At least 4,600 related trials are counted, including explicit selection of the
80-session rule from the prior 40/60/80 table.  No claim of independent
discovery is permitted.

## Falsification rule

Reject if any guardrail fails.  Do not promote the 100-session neighbour even
if it performs better.  A complete historical pass reaches the user's research
score objective and may be labelled **Track as research challenger**, but it is
not eligible for frozen-baseline adoption until meaningful clean forward OOS
evidence exists and the user explicitly requests promotion.
