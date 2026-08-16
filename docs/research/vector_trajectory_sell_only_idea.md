# Nine-feature trajectory vector as a sell-only signal

Failure mode:
The prior six-feature crash vector could not reliably replace the frozen
bearish-divergence exit. It either produced no primary-threshold exits or
generated false exits at lower thresholds. A static market snapshot may fail
to distinguish improving from deteriorating states with similar levels.

Causal hypothesis:
Adding the twenty-session slopes of VIX, breadth, and SPX trailing-252-session
drawdown to the six crash-state features will improve identification of a
future S&P 500 decline of at least twenty percent within 126 sessions.

Signal change:
Use a nine-feature expanding-history nearest-neighbour crash probability to
replace only the frozen bearish-divergence exit. Trigger the primary sell
signal when probability is at least 50%. Preserve both canonical buy paths,
cooldown, climax-top exit, and 25% trailing-stop exit exactly.

Entry or exit only:
Exit only. No buy signal is filtered, replaced, or added.

Data available at decision time:
The original six close-based features plus causal twenty-session OLS slopes of
VIX, breadth, and SPX drawdown. Each slope uses the signal close and preceding
nineteen sessions. A crash label becomes eligible only after its complete
126-session future window is historical.

Primary metric:
Exit by the next open before the first close at a twenty-percent S&P 500
drawdown breach for every evaluable crash episode in which the strategy was
exposed between the prior peak and breach.

Guardrails:
- Exact canonical baseline parity.
- No lookahead and next-session-open execution.
- Canonical washout and MA200-recross buy logic remain unchanged.
- Climax-top and 25% trailing-stop exits remain unchanged.
- Maximum drawdown must not deteriorate.
- CAGR may not trail baseline by more than two percentage points.
- Cost-adjusted expectancy must remain positive.
- Completed round trips may not increase by more than one per year.
- Performance effect must not reverse between 2002-2013 and 2014-present.
- Any primary benefit must remain at five times modeled costs.

Single threshold and fixed sensitivity values:
Primary crash probability 50%; sensitivity 40% and 60%. Twenty-session slope
window, 126-session crash horizon, fifteen monthly-independent neighbours, and
all nine features are fixed. No threshold, window, feature-subset, or
acceleration search is permitted.

Expected helpful regimes:
Persistent deterioration where breadth and drawdown worsen and volatility
rises before a bear-market breach.

Expected failure regimes:
Sudden first-of-kind shocks, early history without enough resolved crash
labels, fast V-shaped declines, and regimes whose slope direction changes
before the next-open fill.

Prior related trials counted for DSR:
All prior baseline and bearish-composite searches, the 567 crash-vector factor
grid variants, three static crash-vector thresholds, three static recross
thresholds, and three trajectory recross thresholds. This challenger is highly
related to prior research and receives a severe multiple-testing penalty.

Falsification rule:
Reject if the 50% primary rule misses either the 2008 or 2020 evaluable
peak-to-breach episode while exposed, if any guardrail fails, if historical
split direction reverses, or if the 40%/50%/60% results are cliff-edge. Even a
historical pass remains research-only until meaningful clean forward evidence
exists.
