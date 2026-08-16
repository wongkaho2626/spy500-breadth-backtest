# Pre-registered idea: monthly MA200 × breadth regime exit

Failure mode:
The frozen strategy can remain invested for several month-ends after the NDX
has broken below its 200-day moving average and fewer than half of S&P 500
members remain above their own 200-day averages.  The 25% trailing stop is
deliberately slow, while the divergence and climax exits do not directly test
whether both trend and participation are already weak.  This contributes to
large drawdowns and only 17 completed round trips.

Causal hypothesis:
A month-end-only exit when both NDX trend and broad participation are broken
will leave persistent bear regimes earlier without reacting to every daily
whipsaw.  The frozen washout and MA200-recross entry paths should then provide
causal re-entry after panic or trend recovery, potentially improving drawdown
and increasing the number of genuinely separate market-timing episodes.

Signal change:
While IN, add one exit rule evaluated on the final trading close of each
calendar month: NDX close below its trailing 200-session moving average AND
S&P 500 percentage-above-200-day breadth below 50%.  The signal fills at the
next session open.  Existing bearish-divergence, climax-top, and 25% trailing
stop exits retain their priority.

Entry or exit only:
Exit only.  Washout, MA200-recross, cooldown, and all entry thresholds remain
unchanged.

Data available at decision time:
The month-end NDX close, its 200-session moving average, and breadth through
that same close.  Calendar month-end status is known at the close.  No future
bar or future constituent information is used.

Primary metric:
Final Backtest Score under the installed `backtest-analyst` rubric, with
Calmar as the primary economic metric.  Reaching at least 80 without a fatal
cap is required to satisfy the active objective.

Guardrails:
- exact frozen-baseline parity when the extra exit is disabled;
- close signal and next-session-open fill;
- CAGR no more than 2 percentage points below baseline;
- full-period Calmar improves and maximum drawdown does not worsen;
- positive cost-adjusted expectancy and profit factor above 1.2;
- no negative Calmar effect in either 2002-2013 or 2014-present;
- no more than double the baseline round trips unless expectancy also rises;
- positive paired mean-return difference at 5x costs;
- 2007+ real-breadth result directionally agrees;
- 40%/50%/60% breadth sensitivity is not cliff-edge.

Single threshold and fixed sensitivity values:
- primary breadth threshold: 50%;
- sensitivity only: 40% and 60%;
- NDX MA: 200 sessions, conventional and fixed;
- evaluation frequency: final trading close of each calendar month, fixed.

Expected helpful regimes:
Persistent bear markets and broad risk-off phases such as 2008, 2011, 2015,
2018, 2020, and 2022, where both price trend and participation deteriorate.

Expected failure regimes:
Brief month-end breaks below MA200 followed by rapid recovery, repeated
sideways recrosses, and strong NDX leadership while S&P breadth remains weak.

Prior related trials counted for DSR:
At least 4,595 related signal/vector/allocation trials, including the rejected
MA200 trend-following family, the monthly drawdown-throttle grid, the 2,880-row
signal grid, and the washout-boost score audit.  This exact month-end
dual-confirmation add-on was not found in the repository search, but it is not
statistically independent of those families.

Falsification rule:
Reject if parity or timing fails; final score is below 80; Calmar fails to
improve; maximum drawdown worsens; CAGR falls by more than 2 points; either
historical half has a negative Calmar delta; expectancy or profit factor
fails; 5x costs reverse the paired return benefit; the 2007+ direction
reverses; or performance is isolated to only the 50% threshold.  Historical
success without meaningful clean post-freeze evidence can only be tracked as
a research challenger and cannot modify the frozen baseline.
