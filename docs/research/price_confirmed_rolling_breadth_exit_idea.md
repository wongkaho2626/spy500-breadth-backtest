# Price-confirmed rolling breadth-drawdown exit — pre-registered idea card

## Failure mode

The frozen divergence exit measures breadth damage from exactly 60 sessions
earlier.  If breadth peaked later inside the window, that anchor can understate
the current deterioration even when NDX has risen enough to satisfy the existing
price-confirmation vote.

## Causal hypothesis

Keeping the NDX 60-session return >=3% vote should filter the excessive sells
seen in the breadth-only rolling-max challenger, while measuring breadth damage
from the most relevant recent participation high may identify mature, narrowing
advances more accurately than the fixed 60-session anchor.

## Signal change

Replace only the breadth component of the frozen bearish-divergence exit:

`breadth[t-60] - breadth[t] >= 20 points`

with:

`max(breadth[t-59:t]) - breadth[t] >= 20 points`.

The complete challenger exit is therefore:

`NDX 60-session return >=3% AND rolling breadth drawdown >=20 points AND breadth <60%`.

Signals use the close and fill at the next session open.  Climax-top and 25%
trailing-stop exits retain their existing priority and logic.

## Entry or exit only

Exit only.  Entries, MA200 trend re-entry, vote gate, cooldown, commissions,
slippage, and all other frozen thresholds remain unchanged.

## Data available at decision time

NDX and breadth closes through the signal session only: NDX close at `t` and
`t-60`, plus breadth observations `t-59` through `t`.  No future data is used.

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

- Fixed NDX confirmation: 60-session return >=3%.
- Fixed rolling breadth window: 60 sessions, including signal session.
- Fixed breadth cap: current breadth <60%.
- Primary rolling breadth drawdown: at least 20 percentage points.
- Sensitivity only: at least 15, 20, or 25 percentage points.

No full-history optimisation will be run.

## Expected helpful regimes

Mature advances where NDX remains up over 60 sessions but participation has
fallen sharply from a breadth peak that occurred inside the lookback window.

## Expected failure regimes

Temporary breadth rotation during a continuing cap-weighted bull trend, or a
rolling peak that keeps the exit active after the deterioration has begun to heal.

## Prior related trials counted for DSR

4,606.  This includes the frozen divergence sweeps, exact-60 breadth-only exit,
rolling breadth-only exit, clustered-breadth exit, signal grids, and vector work.

## Falsification rule

Reject if parity or timing fails, primary Calmar does not improve, any guardrail
fails, either historical half or 2007+ reverses the improvement, sensitivity is
cliff-edge, or 5x costs erase the paired benefit.  Historical success without
sufficient clean evidence after 2026-07-05 can only be tracked as research and
cannot modify the frozen baseline.
