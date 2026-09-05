# Pre-registered grid — NDX RSI golden-cross plus MA regime gate

Registered on 2026-09-02 before running any grid performance.

## Failure mode

The fixed RSI(7/14)+MA200 re-entry reduced some false golden crosses but still
underperformed the frozen QQQ strategy. The user explicitly requested a grid search to
determine whether the failure is specific to those conventional horizons or applies to
the broader RSI/MA family.

## Rule family

While flat, after at least one prior strategy exit and the frozen 15-calendar-day
cooldown, add a lower-priority entry when:

`RSI_short[t] > RSI_long[t]`

and

`RSI_short[t-1] <= RSI_long[t-1]`

and

`NDX close[t] > MA_window[t]`.

Signals use `NASDAQ100.csv` closes through day `t` and fill at day `t+1` open.
Canonical washout and MA200-recross entries retain priority. All exits remain frozen.

## Grid fixed before results

- Short RSI windows: `3, 5, 7, 9, 11`.
- Long RSI windows: `10, 14, 21, 28`.
- Require short window < long window.
- MA windows: `100, 150, 200, 250, 300`.
- Total combinations: 19 valid RSI pairs × 5 MA windows = **95**.
- RSI smoothing: Wilder-style `ewm(alpha=1/window, adjust=False)`.

No grid expansion or threshold refinement is permitted after results are seen.

## Selection objective and tie-breaks

Primary objective: highest Calmar ratio. Ties are resolved by higher Sharpe, then
higher CAGR, then lexicographically smaller `(short, long, MA)` for determinism.

## Historical split protocol

1. Primary time direction: rank all 95 configurations on 2002–2013, freeze the winner,
   and evaluate it on 2014–2026.
2. Reverse robustness direction: rank on 2014–2026, freeze that winner, and evaluate it
   on 2002–2013.
3. Also report full-history rankings, labelled **in-sample descriptive only**.
4. Re-run the primary time-direction winner over the full history for trade, cost,
   statistical, score, regime, and artifact reporting.

All pre-2026-07-02 observations have already been seen by the project, so even the
historical test halves are pseudo-OOS rather than genuine forward OOS.

## Primary metric

Calmar improvement of the 2002–2013-selected combination on the 2014–2026 test half.

## Guardrails

- Exact disabled-harness parity with `qqq_backtest.py`.
- Primary test-half Calmar must exceed the matching frozen baseline.
- Reverse-direction test-half Calmar must also exceed its matching baseline.
- Full-history Calmar and maximum drawdown must not worsen.
- Cost-adjusted expectancy must not worsen.
- Turnover may not rise by more than 50% unless expectancy improves.
- 2007+ real-breadth Calmar delta must be non-negative.
- Paired annualized mean return must remain positive at 5x costs.
- The selected point must not be an isolated cliff relative to adjacent grid points.

## Prior trials counted for DSR

The prior repository count is 4,823. This grid adds 95 attempted configurations, so
the multiple-testing count is fixed at **4,918**.

## Falsification rule

Reject the family if baseline parity fails, the primary test-half Calmar does not
improve, reverse-direction test Calmar does not improve, full-history Calmar/drawdown
or expectancy worsens, cost stress reverses the edge, or the selected combination is
cliff-edge. A historical pass may only be tracked; adoption requires meaningful clean
post-freeze evidence and an explicit user promotion request.
