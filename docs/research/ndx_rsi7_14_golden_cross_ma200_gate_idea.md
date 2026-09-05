# Idea card — NDX RSI(7/14) golden-cross re-entry with MA200 gate

Pre-registered on 2026-09-02 before running gated-challenger performance.

## Failure mode

The ungated RSI(7/14) golden-cross re-entry buys short-term rebounds during negative
long-term trends. Its 19 executed entries increased exposure to 91% and worsened CAGR,
Calmar, and maximum drawdown. Trade attribution found that the seven crosses below
MA200 had much lower average trade returns than crosses above MA200, including failed
2008 and 2022 bear-market rebounds.

## Causal hypothesis

An RSI golden cross measures short-term momentum recovery, while price above a long
moving average establishes that the long-term regime is already positive. Requiring
both should reject many bear-market rallies without discarding useful bull-market
re-entry signals.

## Signal change

While flat, after a prior strategy exit and the frozen 15-calendar-day cooldown, add
an entry only when both are true at the daily close:

1. `RSI7[t] > RSI14[t] AND RSI7[t-1] <= RSI14[t-1]`;
2. `NDX close[t] > NDX moving average[t]`.

Primary moving-average window: 200 sessions. Existing washout and MA200-recoss entries
retain priority. All frozen exits remain unchanged. The prior death-cross exit is not used.

## Entry or exit only

Entry only.

## Data available at decision time

NASDAQ-100 closes from `NASDAQ100.csv` through session `t`; S&P breadth and VIX are
unchanged canonical inputs. A qualified close signal fills at the next available
session open (`t+1`).

## Primary metric

Full-period Calmar ratio versus the frozen baseline over identical dates.

## Guardrails

- Exact disabled-harness baseline parity.
- No lookahead and next-session-open fills.
- Frozen $1 commission and 0.05% slippage per side.
- Maximum drawdown and cost-adjusted expectancy may not worsen.
- CAGR may not fall by more than 2 percentage points.
- Turnover may not rise by more than 50% unless expectancy improves.
- Calmar improvement must be non-negative in both historical halves and in 2007+
  real-breadth data.
- Paired annualized mean return must remain positive at 5x costs.
- MA-window sensitivity must not be cliff-edge.

## Single parameter and fixed sensitivity values

- RSI periods are frozen at 7 and 14 from the prior crossover round.
- Primary trend window: 200 sessions.
- Sensitivity windows: 150, 200, and 250 sessions.

These are conventional long-trend horizons, not selected from gated performance.

## Expected helpful regimes

Bull-market pullbacks and post-exit recoveries where short momentum turns up while the
index remains above its established long-term trend.

## Expected failure regimes

Late-cycle rallies still above MA200, fast bear markets before the moving average turns
down, and recoveries that begin below MA200 and are therefore entered too late or missed.

## Prior related trials counted for DSR

4,823 total trials: the repository's prior 4,822-trial count plus this one
pre-registered MA-gated golden-cross hypothesis.

## Falsification rule

Reject if parity fails, primary Calmar does not improve, either historical half reverses
the Calmar improvement, drawdown or expectancy worsens, turnover breaches its guardrail,
5x costs erase the paired benefit, or 150/200/250-session sensitivity is cliff-edge. A
historical pass can only be tracked until meaningful untouched post-freeze evidence exists.
