# Pre-registered idea: trajectory-gated 60-to-80-session washout boost

Status: pre-registered before adaptive-rule performance is run.

## Failure mode

The fixed 60-session boost has stable recent-period Calmar but insufficient
incremental statistical significance, while the fixed 80-session boost reaches
83 points yet worsens Calmar in the 2014+ and actual-TQQQ periods.  A fixed
duration cannot distinguish a recovery that is still broadening at day 60 from
one whose trend or participation has already weakened.

## Causal hypothesis

Leveraged washout exposure should be extended from 60 to 80 sessions only when
both the index trend and broad participation are still improving at the day-60
decision close.  Otherwise rotating to NDX at day 61 should preserve the safer
60-session profile.  This uses the trajectory concept requested earlier rather
than a calendar-era rule.

## Signal change

Keep the fixed 70% frozen-breadth / 30% NDX-above-MA200 ensemble and its fixed
10-percentage-point washout-only TQQQ allocation.  On the close of session 60
after a leveraged entry:

- extend TQQQ to the session-80 close only when NDX close is above its MA200
  **and** current breadth is higher than breadth 20 sessions earlier;
- otherwise schedule TQQQ sale and NDX purchase at the next-session open;
- an extended trade rotates at the next open after the session-80 close;
- a frozen breadth exit due at the same open takes priority.

The replacement NDX remains until the unchanged frozen breadth exit.

## Entry or exit only

Allocation/risk-management timing only.  No frozen entry, frozen exit,
cooldown, sell priority, boost size or MA200-sleeve rule changes.

## Data available at decision time

At the day-60 close, NDX, its MA200, current breadth, and breadth 20 completed
sessions earlier are all known.  The extension decision cannot use days 61-80.
Every resulting transaction fills at the following session open.

## Primary metric

Final Backtest Score.  The 20-session trajectory-gated challenger must score at
least 80 without a cap and pass every guardrail.  CAGR alone cannot pass.

## Guardrails

- exact frozen-baseline parity;
- forcing every gate false exactly reproduces the fixed 60-session engine;
  forcing every gate true exactly reproduces the fixed 80-session engine;
- at least 30 independent exit clusters and component-return correlation below
  0.95;
- all original entries/exits and adaptive rotations use close information and
  next-session-open fills, with $1 commission and 0.05% slippage on every leg;
- CAGR, Sharpe and Calmar exceed the frozen baseline; maximum drawdown is below
  30% and no worse than baseline by more than two points;
- paired annual return is positive, HAC p is below 0.05 and the 21-session
  block-bootstrap 95% interval excludes zero;
- CAGR, Sharpe and Calmar deltas are positive in 2002-2013, 2014-present,
  2007+ real-breadth and 2010-02-11+ actual-TQQQ periods;
- paired mean return is positive in both odd and even calendar years;
- paired annual return remains positive at 5x transaction costs;
- profit factor exceeds 1.2 and expectancy is positive;
- 10/20/40-session trajectory lookbacks all have positive paired mean, CAGR
  positive, Sharpe above 1.0 and Calmar above 0.5; adjacent core metrics differ
  by less than 20% relatively; score spread is at most five points;
- under 3x pre-inception TQQQ drag, paired return remains positive, maximum
  drawdown stays below 30%, and final score remains at least 80.

## Single threshold and fixed sensitivity values

- primary breadth trajectory lookback: 20 sessions, one trading month;
- sensitivity only: 10 and 40 sessions;
- improvement threshold is the non-tuned sign test `breadth[t] > breadth[t-L]`;
- short/long boost ages remain fixed at the previously studied 60/80 sessions;
- boost size remains 10% and MA200 sleeve remains 30%.

## Expected helpful regimes

Recoveries whose price trend and breadth participation remain healthy after
three months, where days 61-80 retain positive convexity.

## Expected failure regimes

Narrow index rallies with weakening breadth, false MA200 recoveries, and cases
where breadth pauses temporarily before a continued advance.

## Prior related trials counted for DSR

At least 4,601 related trials are counted, including the selected fixed-duration
results and earlier trajectory-vector studies.  This adaptive rule is not
treated as independent discovery.

## Falsification rule

Reject if any guardrail fails.  Do not change the primary lookback or extension
logic after viewing results.  A complete historical pass reaches the research
score objective and supports forward tracking only; frozen-baseline adoption
still requires meaningful clean forward OOS evidence and explicit user intent.
