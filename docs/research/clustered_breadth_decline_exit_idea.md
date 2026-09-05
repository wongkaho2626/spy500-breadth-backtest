# Clustered breadth-decline exit — pre-registered idea card

## Failure mode

The frozen QQQ strategy can remain invested while market breadth is deteriorating
rapidly but has not yet satisfied the slower 60-session bearish-divergence rule.
This may delay exits near the start of persistent declines.

## Causal hypothesis

Repeated large one-session contractions in the percentage of S&P 500 members
above their 200-day moving average indicate broadening internal weakness.  Requiring
the ten-session breadth trajectory to slope downward should reject isolated breadth
shocks and identify deterioration that is more likely to persist.

## Signal change

Add one lower-priority exit to the frozen strategy.  At close on session `t`, exit
when both conditions are true:

1. at least four of the last ten breadth observations, including `t`, have a
   one-session relative change strictly below -2%; and
2. the ordinary-least-squares slope of the ten breadth levels against session
   number is strictly negative.

Existing bearish-divergence, climax-top, and trailing-stop exits retain their
priority.  The new close signal fills at the next session open and uses the frozen
15-calendar-day cooldown and trend/washout re-entry logic.

## Entry or exit only

Exit only.

## Data available at decision time

Breadth levels through the signal-session close only.  No future breadth, price, or
fill information is used.  The earliest fill is the next available session open.

## Primary metric

Full-period Calmar ratio from 2002-01-02 through the latest common observation.

## Guardrails

- exact disabled-signal parity with `qqq_backtest.py` for equity, completed trades,
  and open-trade state;
- maximum drawdown must not worsen;
- CAGR may not fall by more than one percentage point versus the frozen baseline;
- cost-adjusted trade expectancy may not fall below baseline;
- completed-trade count may not rise by more than 25%;
- early (2002-2013) and late (2014-latest) Calmar deltas must both be non-negative;
- the 2007+ real-breadth Calmar delta must be non-negative;
- the paired annual return difference must remain positive at 5x costs;
- nearby event-count thresholds must not show a cliff edge: all three must beat
  baseline Calmar, and adjacent Calmar ratios may not differ by 25% or more.

## Single threshold and fixed sensitivity values

- Fixed lookback: 10 sessions.
- Fixed large-decline threshold: relative breadth change < -2%.
- Fixed slope rule: ten-session OLS slope < 0 percentage points per session.
- Primary event-count threshold: at least 4 sessions.
- Pre-registered sensitivity: at least 3, 4, or 5 sessions.

Only the event-count threshold varies.  No full-history optimisation will be run.

## Expected helpful regimes

Persistent deterioration after a mature advance, including declines where breadth
weakens quickly before the frozen 60-session divergence becomes active.

## Expected failure regimes

Fast V-shaped pullbacks, noisy breadth near an otherwise healthy uptrend, and
periods where breadth contracts while cap-weighted NDX leadership remains strong.

## Prior related trials counted for DSR

4,603.  This conservatively includes the repository's broad QQQ signal searches,
vector/slope research, breadth-regime exits, and prior challenger rounds even
though no exact match for this rule was found.

## Falsification rule

Reject if baseline parity fails, the primary Calmar ratio does not improve, any
guardrail fails, either historical half reverses the Calmar improvement, all nearby
thresholds do not improve Calmar, or transaction costs erase the paired return
benefit.  A historical pass can only be tracked as a research challenger because
post-2026-07-05 clean forward evidence is still too short.
