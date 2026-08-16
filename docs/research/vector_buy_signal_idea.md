# Pre-registered idea: causal vector buy signal

Failure mode:
The frozen strategy's five MA200-recross entries earned about 14.9% on average
versus about 38.1% for twelve washout entries, while the January 2008 washout
continued into a broken trend and lost 10.8%. The current entry rules identify
specific chart conditions rather than estimating the forward reward/risk of
the complete market state.

Causal hypothesis:
A nearest-neighbour probability based on the six existing market-state factors
can identify entry dates with a favorable six-month NDX reward/risk profile
more consistently than the frozen washout and MA200-recross rules.

Signal change:
Disable both frozen buy paths. While out of the market and past the unchanged
15-calendar-day cooldown, buy only when the vector probability of a successful
126-session NDX outcome is at least 60%.

A successful historical buy state is defined before testing as:
- NDX close 126 sessions later is at least 10% above the signal close; and
- the minimum NDX close during those 126 sessions is not more than 15% below
  the signal close.

Vector factors:
1. S&P 500 close-to-close daily return;
2. NASDAQ-100 60-session return;
3. percentage of S&P 500 constituents above their 200-day moving average;
4. 60-session breadth decline;
5. S&P 500 drawdown from its trailing 252-session high;
6. VIX close.

Model:
Robust-scale with the median and IQR using only labels whose complete
126-session window has resolved before the prediction close. Select the nearest
historical state from each calendar month, retain the closest 15 months, and
use inverse-distance weights. This is the same causal analogue mechanism used
in the prior vector research, but the target and changed side are entry-only.

Entry or exit only:
Entry only. Frozen bearish-divergence, climax-top, and 25% trailing-stop exits
remain unchanged.

Data available at decision time:
SPX, NDX, breadth, and VIX closes through the signal close. Historical labels
are admitted only after all 126 future sessions are already in the past.
Signals fill at the next session open.

Primary metric:
Improve both Sharpe and Calmar relative to the frozen baseline.

Guardrails:
- use the canonical engine for exact baseline parity;
- no lookahead and next-session-open fills;
- CAGR no more than two percentage points below baseline;
- maximum drawdown must not deteriorate;
- cost-adjusted trade expectancy remains positive;
- no more than one additional round trip per year;
- Sharpe and Calmar changes are directionally non-negative in both historical
  halves;
- result survives 5x costs.

Single threshold and fixed sensitivity values:
- primary probability threshold: 60%;
- sensitivity only: 50% and 70%;
- label horizon: 126 sessions, fixed;
- reward threshold: +10%, fixed;
- maximum adverse excursion threshold: -15%, fixed;
- neighbours: 15 monthly-independent analogues, fixed.

Expected helpful regimes:
Post-washout recoveries, healthy trend resumptions, and market states with
similar breadth/volatility combinations to historically favorable six-month
advances.

Expected failure regimes:
First-of-kind crashes, sudden regime changes, prolonged sideways markets, and
the early sample before sufficient resolved analogue months exist.

Prior related trials counted for DSR:
The documented baseline and walk-forward grids, eleven challenger families,
the bearish-composite search, the rejected vector-exit and factor-subset
searches, and this three-threshold sensitivity set. The buy target is new, but
the features and nearest-neighbour mechanism have already been viewed.

Falsification rule:
Reject if the 60% signal fails to improve both Sharpe and Calmar, violates any
guardrail, reverses direction across historical halves, is cliff-edge across
50%/60%/70%, or loses its benefit at 5x costs. Track rather than adopt if all
historical tests pass because there is not yet meaningful clean forward OOS
evidence after the 2026-07-05 freeze.
