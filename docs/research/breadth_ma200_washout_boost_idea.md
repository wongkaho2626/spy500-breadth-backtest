# Pre-registered idea: MA200 ensemble with washout-only leveraged allocation

Status: pre-registered before the combined challenger performance is run.

## Failure mode

The fixed 70% frozen-breadth / 30% NDX-MA200 ensemble reduces drawdown and
creates 48 independent exit clusters, but its MA200 sleeve sits in cash during
part of strong recoveries.  It therefore loses about 1.5 percentage points of
annual return versus the frozen strategy.  Separately, canonical washout
entries have historically supported a positive TQQQ allocation effect, but the
standalone boost deepens drawdown and still has only 17 independent trades.

## Causal hypothesis

The 30% MA200 sleeve can provide enough defensive diversification to fund a
small leveraged allocation only during the frozen strategy's highest-conviction
breadth-washout entries.  The washout allocation should offset the trend
sleeve's cash drag while the trend sleeve limits the leveraged drawdown.  The
two mechanisms use different state variables and should retain more than 30
independent event clusters.

## Signal change

Keep the fixed 70% frozen breadth sleeve and 30% NDX-above-MA200 sleeve from the
previous audit.  Within the 70% breadth sleeve only, allocate 10 percentage
points of total initial capital to TQQQ when a completed frozen entry is a
breadth washout (`buy_trigger != MA200-recross`).  The remaining 60 points hold
NDX.  On a frozen MA200-recross entry, all 70 points hold NDX.  Both legs exit
on the unchanged frozen breadth sell signal.  There is no rebalancing within a
trade and no change to any entry or exit date.

## Entry or exit only

Entry allocation only.  Frozen washout, MA200-recross, bearish-divergence,
climax-top, trailing-stop, cooldown, exit priority and the independent 30%
MA200 sleeve remain unchanged.

## Data available at decision time

The frozen buy trigger is known at close on signal day.  Trades fill at the
next-session open.  Actual TQQQ data are used from 2010-02-11; earlier values
use the existing 3x-NDX-minus-calibrated-drag proxy.  Because overlap drag uses
later observations, 1x and punitive 3x pre-inception drag are reported and the
synthetic-data limitation remains explicit.

## Primary metric

Final Backtest Score.  The challenger must reach at least 80 without a hard cap
and pass every guardrail below.  CAGR alone cannot pass the hypothesis.

## Guardrails

- exact frozen-baseline parity for a 100% breadth control;
- exact parity to the prior fixed 70/30 ensemble when washout boost is zero;
- at least 30 exit clusters after grouping component exits within 21 sessions;
- close signal and next-session-open execution, with $1 commission and 0.05%
  slippage on every traded leg;
- CAGR, Sharpe and Calmar each exceed the frozen baseline, and maximum drawdown
  is no worse by more than 2 percentage points;
- paired annual return is positive and its 21-session block-bootstrap 95%
  interval excludes zero;
- CAGR, Sharpe and Calmar deltas are positive in 2002-2013, 2014-present and
  2007+ real-breadth periods;
- paired annual return remains positive at 5x transaction costs;
- profit factor is above 1.2 and expectancy is positive;
- all fixed neighbouring allocations are directionally consistent and their
  scores differ by no more than five points;
- under 3x pre-inception TQQQ drag, paired return remains positive, maximum
  drawdown stays inside the 2-point allowance and the final score is at least
  80.

## Single threshold and fixed sensitivity values

- primary washout-only TQQQ allocation: 10% of total initial capital;
- sensitivity only: 5% and 15%;
- trend sleeve remains fixed at the previously registered 30%;
- no value outside 5%/10%/15% will be viewed or selected in this round.

## Expected helpful regimes

Broad panic washouts followed by persistent rebounds, especially recoveries
where the MA200 sleeve is initially out of market.

## Expected failure regimes

Early washouts in unfinished bear markets, leveraged volatility decay, adverse
overnight gaps, and periods where the defensive sleeve is already fully
invested and provides little offsetting protection.

## Prior related trials counted for DSR

At least 4,598 related signal, vector, allocation and ensemble trials are
counted.  Both ingredients have been viewed separately, so this is a
pre-registered combination test rather than independent discovery.

## Falsification rule

Reject if any guardrail fails.  Do not change the 10% primary allocation after
seeing performance.  Even a complete historical pass is only eligible for
forward tracking because there is not yet a meaningful clean post-2026-07-05
OOS sample; it does not alter the frozen baseline.
