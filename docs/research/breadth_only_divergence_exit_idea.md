# Breadth-only 60-session decline exit — pre-registered idea card

## Failure mode

The frozen bearish-divergence exit requires the NDX to have risen at least 3%
over 60 sessions.  That price condition can block an exit after market breadth
has already fallen sharply during a flat or declining NDX market, potentially
delaying risk reduction.

## Causal hypothesis

A fall of at least 20 percentage points in S&P 500 percentage-above-200-day
breadth over 60 sessions, while current breadth is below 60%, is sufficient
evidence of broad internal deterioration.  Removing the NDX price-rise vote may
exit persistent declines earlier without changing entries or the other exits.

## Signal change

Replace only the frozen bearish-divergence condition:

`NDX 60d return >= 3% AND breadth 60d fall >= 20 points AND breadth < 60%`

with:

`breadth 60d fall >= 20 points AND breadth < 60%`.

The signal is evaluated at session `t` close and fills at the next session open.
Climax-top and 25% trailing-stop exits retain their existing priority and logic.

## Entry or exit only

Exit only.  Washout entry, MA200 trend re-entry, vote gate, 15-calendar-day
cooldown, costs, and all entry thresholds remain unchanged.

## Data available at decision time

The current breadth close and breadth close exactly 60 strategy sessions earlier.
No future breadth, price, or fill data is used.

## Primary metric

Full-period Calmar ratio from 2002-01-02 through the latest common observation.

## Guardrails

- exact baseline parity when the price-rise removal is disabled;
- maximum drawdown must not worsen;
- CAGR may not fall more than two percentage points below baseline;
- cost-adjusted trade expectancy may not fall below baseline;
- completed-trade count may not rise by more than 50% unless expectancy improves;
- early (2002-2013), late (2014-latest), and 2007+ real-breadth Calmar deltas
  must be non-negative;
- paired annual return difference must remain positive at 5x costs;
- the pre-registered 15/20/25-point sensitivity family must improve Calmar at
  every value, with no adjacent Calmar change of 25% or more.

## Single threshold and fixed sensitivity values

- Fixed lookback: 60 sessions.
- Fixed breadth cap: current breadth < 60%.
- Primary breadth-fall threshold: at least 20 percentage points.
- Sensitivity only: at least 15, 20, or 25 percentage points.
- The NDX price-rise condition is absent in every challenger variant.

No full-history optimisation will be run.

## Expected helpful regimes

Persistent bear-market onset or broad internal breakdown where NDX is flat or
already declining, so the frozen price-rise vote is false despite severe breadth
deterioration.

## Expected failure regimes

Short breadth washouts close to a market bottom, sharp V-shaped recoveries, and
leadership-narrowing bull markets where cap-weighted NDX remains resilient.

## Prior related trials counted for DSR

4,604.  This conservatively includes repository-wide signal grids, divergence
parameter sweeps, breadth-regime exits, vector studies, and the immediately prior
clustered-breadth challenger.  The rule is not treated as statistically independent.

## Falsification rule

Reject if parity or timing fails, primary Calmar does not improve, any guardrail
fails, the improvement reverses across historical halves or the 2007+ period,
sensitivity is cliff-edge, or 5x costs erase the paired benefit.  Historical
success without sufficient observations after 2026-07-05 can only be tracked as
a research challenger and cannot modify the frozen baseline.
