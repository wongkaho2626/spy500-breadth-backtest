# Rolling-max-only exit — pre-registered idea card

## Failure mode

Climax-top and trailing-stop exits may interrupt otherwise profitable holds or
create state-dependent re-entry delays.  The previously selected rolling breadth
rule may be sufficient as the sole exit.

## Causal hypothesis

The robust grid winner's price-confirmed rolling breadth deterioration signal
captures the intended mature-advance failure mode directly.  Removing auxiliary
climax and trailing exits may reduce premature sells and improve compounding.

## Signal change

Use only this sell rule:

`NDX 60-session return >=2%`

and

`max(breadth[t-59:t]) - breadth[t] >=30 points`

and

`breadth[t] <70%`.

Disable both:

- climax top: price >=5% above its 10-day MA plus MACD bearish cross within 10 days;
- trailing stop: price 25% below the high since entry.

Signal close, next-session-open fill, entries, cooldown, commission, and slippage
remain unchanged.

## Entry or exit only

Exit logic only.  No entry or position-sizing changes.

## Data available at decision time

NDX and breadth closes through the signal session.  No future information.

## Primary metric

Full-period Calmar ratio versus both the frozen baseline and the same 2%/30/70%
rolling rule with climax/trailing exits still enabled.

## Guardrails

- exact canonical parity when the replacement and auxiliary-disable switches are off;
- Calmar must exceed frozen baseline and the auxiliary-exits-on comparator;
- maximum drawdown must not worsen versus frozen baseline;
- CAGR may not fall more than two percentage points below frozen baseline;
- expectancy may not fall below frozen baseline;
- early, late, and 2007+ real-breadth Calmar deltas must be non-negative;
- paired annual return difference must remain positive at 5x costs;
- 25/30/35-point sensitivity must improve Calmar throughout without a 25% cliff.

## Fixed threshold and sensitivity

- Price rise: fixed at 2% over 60 sessions.
- Rolling breadth window: fixed at 60 sessions.
- Breadth cap: fixed at 70%.
- Primary breadth drawdown: 30 points.
- Sensitivity only: 25, 30, 35 points.

## Expected helpful regimes

Long bull trends where climax exits are premature and the rolling deterioration
rule eventually provides a cleaner exit.

## Expected failure regimes

Persistent bear markets where NDX 60-session return becomes negative before the
rolling rule can trigger.  Without the 25% trailing stop, such positions may
remain invested through a deep decline.  Removing climax exits may also prevent
the special MA200 re-entry path reserved for prior climax exits.

## Prior related trials counted for DSR

4,819: 4,816 trials through the fixed grid plus the three pre-registered
drawdown sensitivities in this round.

## Falsification rule

Reject if parity/timing fails, primary Calmar does not beat both comparators,
drawdown worsens, either historical half reverses, sensitivity is unstable, or
5x costs erase the benefit.  Historical success remains research-only because
the 2%/30/70% rule was selected using already-seen data.
