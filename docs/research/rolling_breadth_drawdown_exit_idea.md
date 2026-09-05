# Rolling 60-session breadth-drawdown exit — pre-registered idea card

## Failure mode

The frozen divergence exit can remain inactive after breadth has collapsed from
a recent high because it compares with exactly 60 sessions earlier and also
requires a positive NDX 60-session return.  A recent breadth peak may occur
inside that window rather than at its first observation.

## Causal hypothesis

The decline from the highest breadth observation in the latest 60 sessions is a
more direct measure of recent participation damage.  A drawdown of at least 20
percentage points while current breadth is below 60% may identify persistent
internal deterioration earlier than the frozen divergence rule.

## Signal change

Replace only the frozen bearish-divergence exit with:

`rolling maximum breadth over sessions t-59 through t - breadth[t] >= 20 points`

and

`breadth[t] < 60%`.

The NDX price-rise vote and exact-session-60 breadth comparison are absent.  The
signal is evaluated at the close and fills at the next session open.  Existing
climax-top and 25% trailing-stop exits retain their priority and logic.

## Entry or exit only

Exit only.  Washout entry, MA200 trend re-entry, vote gate, 15-calendar-day
cooldown, commissions, slippage, and all entry thresholds remain unchanged.

## Data available at decision time

Only breadth closes from the current and preceding 59 strategy sessions are
used.  No future breadth, price, or fill data enters the signal.

## Primary metric

Full-period Calmar ratio from 2002-01-02 through the latest common observation.

## Guardrails

- exact baseline parity when the replacement signal is disabled;
- maximum drawdown must not worsen;
- CAGR may not fall more than two percentage points below baseline;
- cost-adjusted trade expectancy may not fall below baseline;
- completed-trade count may not rise by more than 50% unless expectancy improves;
- early (2002-2013), late (2014-latest), and 2007+ real-breadth Calmar deltas
  must all be non-negative;
- paired annual return difference must remain positive at 5x costs;
- the pre-registered 15/20/25-point sensitivity family must improve Calmar at
  every value, with no adjacent Calmar change of 25% or more.

## Single threshold and fixed sensitivity values

- Fixed rolling window: 60 sessions, including the signal session.
- Fixed breadth cap: current breadth < 60%.
- Primary breadth drawdown: at least 20 percentage points.
- Sensitivity only: at least 15, 20, or 25 percentage points.
- No price-return condition in any challenger variant.

No full-history optimisation will be run.

## Expected helpful regimes

Persistent breadth breakdowns after a recent participation high, including
cases where breadth rebounds or the 60-session anchor no longer represents the
peak while the market remains internally damaged.

## Expected failure regimes

Ordinary bull-market breadth rotation, sharp V-shaped washouts, and periods of
narrow cap-weighted NDX leadership that recover without a prolonged price fall.

## Prior related trials counted for DSR

4,605.  This includes the prior exact-60-session breadth-only challenger, the
clustered-breadth exit, repository-wide signal grids, divergence sweeps, vector
studies, and breadth-regime exits.

## Falsification rule

Reject if parity or timing fails, primary Calmar does not improve, any guardrail
fails, either historical half or the 2007+ period reverses the improvement,
sensitivity is cliff-edge, or 5x costs erase the paired benefit.  Historical
success without sufficient clean post-2026-07-05 evidence can only be tracked as
a research challenger and cannot modify the frozen baseline.
