# Validation plan: washout-only 10% TQQQ boost

Status: retrospective validation plan, not a blind pre-registration.  The
repository README already reports that the historical CAGR improved, so that
fact is treated as previously seen and cannot be used by itself to pass this
round.

Failure mode:
The frozen strategy treats every entry as 100% unlevered NASDAQ exposure even
though completed washout entries have materially stronger historical trade
returns than MA200-recross entries.  This may leave return on the table in the
highest-conviction entry regime.

Causal hypothesis:
At a genuine breadth washout, a small and fixed leveraged sleeve can exploit
the rebound without changing the timing rules.  Keeping the sleeve off during
MA200 recrosses should avoid levering the weaker entry regime.

Signal change:
No entry or exit date changes.  For the 70% QQQ / 30% annual NDX top-1
portfolio, carve 10 percentage points from QQQ into TQQQ only for trades whose
entry is a canonical breadth washout.  The conditional sleeve holds QQQ on
MA200-recross trades.  No daily rebalancing occurs within a trade.

Entry or exit only:
Entry allocation only.  Washout, MA200-recross, bearish-divergence,
climax-top, trailing-stop, cooldown, and signal priority are unchanged.

Data available at decision time:
The canonical buy trigger known at the signal close; next-session opens for
execution; annual top-holding membership known for that calendar year; TQQQ
prices after inception and a fixed pre-inception 3x-NDX-minus-drag proxy whose
drag is calibrated on the complete actual overlap.

Primary metric:
Full Backtest Score under the installed `backtest-analyst` rubric.  The
challenger must improve the raw score and reach at least 80 without a fatal
cap to satisfy the user's objective.  CAGR is secondary.

Guardrails:
- exact baseline parity when the conditional sleeve weight is zero;
- close signal and next-session-open fill, with slippage and commission on
  every traded leg;
- no worse full-period maximum drawdown by more than 5 percentage points;
- positive CAGR and Sharpe deltas in both 2002-2013 and 2014-present;
- positive cost-adjusted expectancy and no change in signal turnover;
- benefit remains positive at 5x transaction costs;
- the 2007+ real-breadth result is directionally consistent;
- synthetic TQQQ drag stress of 1x and 3x does not reverse the result.

Single threshold and fixed sensitivity values:
- primary conditional TQQQ sleeve: 10%;
- sensitivity only: 5% and 15%;
- no parameter search or selection from the sensitivity rows.

Expected helpful regimes:
Broad panic washouts followed by rapid rebounds, especially 2002, 2009, 2020,
and 2022-2023.

Expected failure regimes:
Early washouts in unfinished bear markets, leveraged volatility decay, large
overnight gaps, and washout entries immediately followed by another decline.

Prior related trials counted for DSR:
At least 4,593 related signal/vector trials from the current research log,
plus the 231-allocation portfolio grid and the already disclosed washout-boost
study.  The exact 10% challenger is therefore previously selected, not an
independent discovery.

Falsification rule:
Reject as an 80-score strategy if the final score is below 80, any hard cap
applies, the raw score does not improve, either historical half has negative
CAGR or Sharpe delta, maximum drawdown worsens by more than 5 points, the 5x
cost result reverses, 3x proxy drag reverses the benefit, or parity/timing
fails.  A historical pass without sufficient clean forward OOS can only be
tracked as a research challenger, never promoted into the frozen baseline.
