# Idea card — NDX RSI(7/14) golden-cross re-entry

Pre-registered on 2026-09-02 before running challenger performance.

## Failure mode

After a frozen-strategy exit, the strategy can remain flat for months while waiting
for another breadth washout or a qualified MA200 recross. The baseline audit shows
that most such flat periods contain an RSI(7/14) golden cross before the next canonical
entry, so a momentum-recovery path may reduce missed upside.

## Causal hypothesis

When short NDX momentum recovers above long momentum after an exit, downside pressure
has eased and a new advance may be starting. Allowing a golden cross to re-enter after
the frozen cooldown may capture recovery earlier without changing the strategy's exits.

## Signal change

While flat and after the existing 15-calendar-day cooldown, add a third entry path when
NDX Wilder RSI(7) crosses from at-or-below Wilder RSI(14) to above it at the daily close:

`RSI7[t] > RSI14[t] AND RSI7[t-1] <= RSI14[t-1]`.

The frozen washout and MA200-recross entries remain available and retain priority.
All bearish-divergence, climax-top, and trailing-stop exits remain unchanged.

## Entry or exit only

Entry only. The prior death-cross exit is not included in this round.

## Data available at decision time

NASDAQ-100 closes from `NASDAQ100.csv` through session `t`. Complete history is used
only to warm up RSI. A golden cross observed at close `t` may fill no earlier than the
next available session open (`t+1`).

## Primary metric

Full-period Calmar ratio, paired against the frozen baseline over identical dates.

## Guardrails

- Exact disabled-harness baseline parity.
- No lookahead; next-session-open fills.
- Frozen $1 commission and 0.05% slippage per side.
- CAGR may not fall by more than 2 percentage points.
- Maximum drawdown and cost-adjusted expectancy may not worsen.
- Turnover may not rise by more than 50% unless expectancy improves.
- Calmar improvement must be non-negative in both historical halves and 2007+
  real-breadth data.
- Paired annualized mean return must remain positive at 5x costs.
- Sensitivity must not be cliff-edge.

## Single parameter and fixed sensitivity values

- Long RSI: fixed at 14 sessions.
- Primary short RSI: 7 sessions.
- Short-RSI sensitivity: 5, 7, and 9 sessions.
- Wilder smoothing: `ewm(alpha=1/window, adjust=False)`.

These horizons match the prior pre-registered crossover round and are not selected from
the new performance results.

## Expected helpful regimes

Recoveries after premature divergence or climax exits where momentum turns up before
the next breadth washout or qualified MA200 recross.

## Expected failure regimes

Persistent bear markets and choppy ranges where repeated golden crosses re-enter too
early, front-run later washout entries, or increase losses and turnover.

## Prior related trials counted for DSR

4,822 total trials: the repository's prior 4,821-trial count plus this one
pre-registered golden-cross re-entry hypothesis. All earlier RSI threshold, divergence,
composite, and death-cross studies remain part of the multiple-testing penalty.

## Falsification rule

Reject if parity fails, Calmar does not improve, either historical half reverses the
Calmar improvement, maximum drawdown or expectancy worsens, turnover breaches its
guardrail, 5x costs erase the paired benefit, or 5/7/9 sensitivity is cliff-edge. A
historical pass can only be tracked until meaningful untouched post-freeze evidence exists.
