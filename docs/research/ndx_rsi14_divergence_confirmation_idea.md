# Idea card — NDX RSI(14) confirmation for bearish-divergence exits

Pre-registered on 2026-09-02 before running challenger performance.

## Failure mode

The frozen strategy sometimes exits on a 60-session price/breadth divergence while
NASDAQ-100 momentum is still healthy, creating a premature exit followed by a later
re-entry or missed upside. The baseline trade review shows multiple divergence exits
followed by positive 21- or 63-session index returns; this experiment tests whether a
neutral-momentum confirmation can distinguish those exits without changing entries.

## Causal hypothesis

A breadth divergence is more likely to mark a durable risk-off transition when NDX
RSI(14) has already fallen below its neutral 50 midpoint. If RSI remains above 50,
index-level momentum is still positive enough that the breadth divergence is more
likely to be an early warning than an immediately tradeable exit.

## Signal change

Replace only the frozen bearish-divergence exit condition with:

`canonical bearish divergence AND NDX Wilder RSI(14) <= threshold`.

The primary threshold is 50. The climax-top and 25% trailing-stop exits are unchanged.
All entry, cooldown, fill, cost, and accounting rules remain frozen.

## Entry or exit only

Exit only.

## Data available at decision time

Daily NASDAQ-100 close from `NASDAQ100.csv`, S&P 500 breadth, and the existing frozen
features, all observed at session close. RSI uses only closes through signal day `t`.
Any resulting order fills no earlier than the next available session open (`t+1`).

## Primary metric

Full-period Calmar ratio (CAGR divided by absolute maximum drawdown), paired against
the frozen baseline over identical dates.

## Guardrails

- Exact disabled-harness baseline parity.
- No lookahead; next-session-open fills and the frozen cost model.
- Challenger maximum drawdown must not worsen.
- CAGR may not fall by more than 2 percentage points.
- Cost-adjusted trade expectancy may not worsen.
- Completed-trade count may not increase.
- Calmar improvement must be non-negative in both historical halves and in 2007+
  real-breadth data.
- Paired annualized mean return must remain positive at 5x costs.
- Sensitivity must not be cliff-edge.

## Single threshold and fixed sensitivity values

- Primary RSI threshold: 50.
- Sensitivity: 45 and 55.
- RSI horizon and smoothing: fixed at 14 sessions with Wilder smoothing
  (`ewm(alpha=1/14, adjust=False)`).

The thresholds are conventional momentum-regime levels, not selected from a grid.

## Expected helpful regimes

Narrowing but still-rising bull markets where breadth fires early while NDX momentum
remains above neutral, followed by renewed upside.

## Expected failure regimes

Fast breaks where RSI crosses 50 only after a large gap, or slow bear markets where
delaying the breadth exit materially increases drawdown before the trailing stop acts.

## Prior related trials counted for DSR

4,820 total trials: the repository's prior 4,819-trial research count plus this one
pre-registered RSI-confirmation hypothesis. Earlier SPX-RSI entry/exit grids, the
NDX RSI<35 mid-trend entry, and RSI components in the bearish-composite search are
treated as related prior exploration rather than independent confirmation.

## Falsification rule

Reject if baseline parity fails, primary Calmar does not improve, maximum drawdown or
expectancy worsens, either historical half reverses the Calmar improvement, the 45/50/55
sensitivity is cliff-edge, or 5x costs erase the paired return benefit. Because data
through 2026-07-02 informed prior strategy research, even a historical pass can only be
tracked as a research challenger until meaningful untouched forward evidence accrues.
