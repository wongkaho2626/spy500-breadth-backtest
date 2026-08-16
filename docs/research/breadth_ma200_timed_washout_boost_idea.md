# Pre-registered idea: time-limited washout TQQQ boost

Status: pre-registered before any timed-boost performance is run.

## Failure mode

The fixed 70% frozen-breadth / 30% NDX-MA200 ensemble with a 10% washout-only
TQQQ allocation raises CAGR and produces a statistically positive paired
increment, but keeping TQQQ until the frozen breadth exit raises maximum
drawdown to about 34% and slightly lowers Sharpe.  Frozen breadth trades can
last for years, far longer than the panic-rebound mechanism that justifies the
leveraged allocation.

## Causal hypothesis

The washout premium should be concentrated in the first quarter after a panic
entry.  Limiting TQQQ to that rebound window and then rotating the same sleeve
into unlevered NDX should retain much of the positive return increment while
removing long-lived leverage and volatility decay.

## Signal change

Start with the already fixed 70% frozen-breadth / 30% NDX-above-MA200
ensemble and its fixed 10-percentage-point washout-only TQQQ allocation.  On
the close of the 60th trading session after a leveraged washout fill, schedule
a next-session-open sale of TQQQ and use the net proceeds to buy NDX.  Hold that
NDX until the unchanged frozen breadth sell signal.  If the frozen sell occurs
first, exit normally.  MA200-recross breadth entries never use TQQQ.

## Entry or exit only

Position allocation/risk management only.  No frozen entry date, frozen exit
date, cooldown, sell priority or 30% MA200-sleeve rule changes.

## Data available at decision time

The washout label is known at its signal close.  Boost age is a simple count of
completed trading sessions since the fill.  The rotation decision is made only
after the age-session close and fills at the next open.  Actual TQQQ data are
used from 2010-02-11; earlier values use the existing 3x-NDX-minus-drag proxy.

## Primary metric

Final Backtest Score.  The primary 60-session challenger must score at least 80
without a hard cap and pass every guardrail.  CAGR alone cannot pass.

## Guardrails

- exact frozen-baseline parity for the 100% breadth control;
- exact parity to the prior unlimited-duration 10% washout-boost ensemble when
  the maximum boost age is disabled;
- at least 30 independent exit clusters after grouping component exits within
  21 sessions, with component-return correlation below 0.95;
- every original entry, exit and timed rotation uses close information and
  fills no earlier than the next-session open;
- $1 commission and 0.05% slippage apply to every buy, sell and rotation leg;
- CAGR, Sharpe and Calmar exceed the frozen baseline; maximum drawdown is no
  worse than the frozen baseline by more than two percentage points;
- paired annual return is positive and its 21-session block-bootstrap 95%
  interval excludes zero;
- CAGR, Sharpe and Calmar deltas are positive in 2002-2013, 2014-present and
  the 2007+ real-breadth period;
- paired annual return remains positive at 5x transaction costs;
- profit factor exceeds 1.2 and expectancy is positive;
- the 40/60/80-session family has positive paired mean at every point; every
  point keeps CAGR positive, Sharpe above 1.0 and Calmar above 0.5; adjacent
  CAGR, Sharpe and Calmar values differ by less than 20% on a relative basis;
  and final-score spread is at most five points;
- under 3x pre-inception TQQQ drag, paired return remains positive, maximum
  drawdown stays within the two-point allowance and final score remains at
  least 80.

## Single threshold and fixed sensitivity values

- primary maximum boost age: 60 trading sessions, a conventional quarter;
- sensitivity only: 40 and 80 trading sessions;
- washout TQQQ allocation remains fixed at the previously registered 10%;
- MA200 sleeve remains fixed at the previously registered 30%;
- no other holding period will be viewed or selected in this round.

## Expected helpful regimes

Fast post-panic recoveries where leveraged exposure is rewarded early but
unnecessary once the acute rebound has passed.

## Expected failure regimes

Slow recoveries that take more than one quarter, renewed selloffs inside the
first 60 sessions, and long bull runs where continued TQQQ exposure would have
outperformed NDX despite higher risk.

## Prior related trials counted for DSR

At least 4,599 related signal, vector, allocation and ensemble trials are
counted.  The 10% allocation and its unlimited-duration result are already
known; only the pre-registered duration rule is new.

## Falsification rule

Reject if any guardrail fails.  Do not change the 60-session primary horizon
after seeing results.  A full historical pass can only be tracked as a research
challenger until meaningful clean post-2026-07-05 observations accumulate; it
does not modify the frozen baseline.
