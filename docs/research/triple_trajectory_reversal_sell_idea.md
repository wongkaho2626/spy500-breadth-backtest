# Triple trajectory reversal sell signal

Failure mode:
The peak event study found that S&P 500 peaks preceding twenty-percent
drawdowns generally still had positive NDX momentum, falling VIX slopes, and
improving SPX drawdown slopes. Static deterioration therefore arrives too late
or is absent at the exact peak. A sell rule should detect the first coordinated
turn away from that strong state.

Causal hypothesis:
After a strong near-high regime, the combination of NDX momentum decelerating,
VIX slope turning upward, and SPX drawdown slope turning downward contains more
useful exit information than the frozen bearish-divergence rule.

Signal change:
Replace only bearish-divergence with one triple-transition exit. Keep both
canonical buy paths, climax-top, and 25% trailing stop unchanged.

Definitions:
- NDX momentum is the trailing sixty-session NDX return.
- NDX deceleration event: its twenty-session OLS slope crosses from
  non-negative to negative.
- VIX reversal event: the twenty-session VIX OLS slope crosses from
  non-positive to positive.
- Drawdown reversal event: the twenty-session slope of SPX drawdown from its
  trailing 252-session high crosses from non-negative to negative.
- Primary sell: all three events occurred within the most recent ten sessions,
  current NDX sixty-session return remains positive, and current SPX drawdown
  is no worse than minus two percent.

Entry or exit only:
Exit only.

Data available at decision time:
NDX, SPX, and VIX closes through the signal close. Every slope uses the signal
close and preceding nineteen sessions. The fill is the next session open.

Primary metric:
Exit through this transition rule before the first S&P 500 twenty-percent
breach in the 2008 and 2020 episodes when the strategy was exposed.

Guardrails:
- Exact baseline parity.
- Canonical washout and MA200-recross inputs remain unchanged.
- No lookahead; next-session-open execution.
- Climax-top and 25% trailing-stop exits remain unchanged.
- Challenger CAGR may not trail baseline by more than two percentage points.
- Maximum drawdown must not worsen.
- Completed round trips may not increase by more than one per year.
- Cost-adjusted expectancy remains positive.
- Challenger-minus-baseline CAGR does not reverse between 2002-2013 and
  2014-present.
- A positive primary effect survives five times modeled costs.

Single parameter and fixed sensitivity:
Primary transition confirmation window ten sessions; sensitivity windows five
and twenty sessions. Slope window twenty sessions, near-high gate minus two
percent, and positive NDX sixty-session return gate are fixed. No threshold,
gate, feature, or window search is permitted.

Expected helpful regimes:
Rounded or rolling tops where risk appetite first changes direction while
price remains close to its high.

Expected failure regimes:
Sudden shocks, transitions that occur only after price has already fallen more
than two percent, and noisy crossovers whose three events cluster by chance.

Prior related trials counted for DSR:
All prior baseline, bearish-composite, vector-exit, recross-filter, trajectory,
and factor-grid work. This is a highly selected hypothesis and receives a
severe multiple-testing penalty.

Falsification rule:
Reject if the primary rule fails to protect either 2008 or 2020 through a
triple-transition exit, any guardrail fails, historical split direction
reverses, or the 5/10/20-session results show cliff-edge dependence. Even a
historical pass remains research-only without meaningful clean forward OOS.
