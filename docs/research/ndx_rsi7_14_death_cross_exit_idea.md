# Idea card — NDX RSI(7/14) death-cross exit

Pre-registered on 2026-09-02 before running challenger performance.

## Failure mode

The frozen strategy can remain invested while short-term NDX momentum rolls over,
because its divergence exit needs a specific 60-session price/breadth pattern and its
25% trailing stop is deliberately slow. A causal momentum cross may identify some
persistent declines sooner.

## Causal hypothesis

When short RSI falls below long RSI, recent downside pressure has become stronger than
the slower momentum regime. Adding that death cross as an exit may reduce drawdown and
improve risk-adjusted returns without altering the high-quality washout and trend entries.

## Signal change

Add one lower-priority exit when NDX Wilder RSI(7) crosses from at-or-above Wilder
RSI(14) to below it on the daily close:

`RSI7[t] < RSI14[t] AND RSI7[t-1] >= RSI14[t-1]`.

The frozen bearish-divergence, climax-top, and trailing-stop exits retain priority.

## Entry or exit only

Exit only. Golden crosses are calculated for auditability but are not used as entries
in this round.

## Data available at decision time

NASDAQ-100 closes from `NASDAQ100.csv` through session `t`. The complete history is
used only to warm up the recursive RSI calculations. A death cross observed at the
close of `t` may fill no earlier than the next available session open (`t+1`).

## Primary metric

Full-period Calmar ratio, paired against the frozen baseline over identical dates.

## Guardrails

- Exact disabled-harness baseline parity.
- No lookahead and next-session-open fills.
- Frozen $1 commission and 0.05% slippage per side.
- CAGR may not fall by more than 2 percentage points.
- Maximum drawdown and cost-adjusted expectancy may not worsen.
- Turnover may not rise by more than 50% unless expectancy improves.
- Calmar improvement must be non-negative in both historical halves and 2007+
  real-breadth data.
- Paired annualized mean return must remain positive at 5x costs.
- Sensitivity must not be cliff-edge.

## Single parameter and fixed sensitivity values

- Long RSI: fixed at 14 sessions, following the user's specified horizon.
- Primary short RSI: 7 sessions, one-half of the long horizon.
- Short-RSI sensitivity: 5, 7, and 9 sessions.
- Wilder smoothing: `ewm(alpha=1/window, adjust=False)`.

These are symmetric, conventional horizon choices and are not selected by backtest
performance.

## Expected helpful regimes

Gradual momentum deterioration before the frozen breadth divergence or trailing stop,
especially persistent bear-market declines.

## Expected failure regimes

Healthy bull-market consolidations with frequent momentum whipsaws, where death-cross
exits trigger cooldowns and cause missed upside before a valid re-entry appears.

## Prior related trials counted for DSR

4,821 total trials: the repository's prior 4,820-trial count plus this one
pre-registered crossover hypothesis. Earlier RSI threshold, divergence, and composite
searches are counted as related exploration.

## Falsification rule

Reject if parity fails, Calmar does not improve, either historical half reverses the
Calmar improvement, drawdown or expectancy worsens, turnover breaches its guardrail,
5x costs erase the paired benefit, or the 5/7/9 sensitivity is cliff-edge. A historical
pass can only be tracked until meaningful untouched post-freeze evidence exists.
