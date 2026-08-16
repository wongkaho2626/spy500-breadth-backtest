# Pre-registered idea: vector SPX crash-risk exit

Failure mode:
The frozen QQQ strategy was invested through the 2008 S&P 500 bear-market
breach and did not exit until its 25% trailing stop. The current bearish-
divergence rule worked before the 2020 breach, but it is not explicitly
designed to warn of an S&P 500 decline of at least 20%.

Causal hypothesis:
Market states resembling historically resolved pre-crash states contain useful
information about whether the S&P 500 will fall at least 20% from the current
close within the next 126 trading sessions.

Signal change:
Replace only the frozen bearish-divergence exit with a causal nearest-neighbour
crash-risk exit. Retain the frozen climax-top and 25% trailing-stop exits.
At each close, form a six-element vector:

1. S&P 500 close-to-close daily return;
2. NASDAQ-100 60-session return;
3. S&P 500 percentage-above-200-day-MA breadth;
4. 60-session breadth decline;
5. S&P 500 drawdown from its trailing 252-session high;
6. VIX close.

Robust-scale the vector using only labels whose 126-session forward windows
have fully resolved before the decision date. Select the nearest historical
state from each calendar month, take the closest 15 months, and compute an
inverse-distance-weighted probability that the S&P 500 subsequently fell at
least 20% from that historical close within 126 sessions.

Entry or exit only:
Exit only.

Data available at decision time:
Daily SPX, NASDAQ-100, breadth, and VIX closes through the signal close.
Execution is the next session's open with the frozen commission and slippage.
Forward crash labels are used only after their complete 126-session window is
historical relative to the prediction date.

Primary metric:
Exit by the next open before the first close at a 20% S&P 500 decline from the
preceding all-time peak, for every evaluable crash episode in which the
strategy was exposed during the peak-to-breach interval.

Guardrails:
- no lookahead;
- challenger CAGR no more than 2 percentage points below baseline;
- challenger cost-adjusted expectancy remains positive;
- no more than one additional completed round trip per year;
- maximum drawdown must not deteriorate;
- retain identical buy logic, cooldown, next-open execution, costs, climax
  exit, and trailing stop.

Single threshold and fixed sensitivity values:
- primary crash-risk probability threshold: 50%;
- sensitivity only: 40% and 60%;
- forecast horizon: 126 sessions, fixed;
- neighbours: 15 monthly-independent analogues, fixed.

Expected helpful regimes:
Persistent deterioration like 2008 and slower bear markets where breadth,
trend, drawdown, and volatility jointly resemble previously resolved
pre-crash states.

Expected failure regimes:
First-of-kind shocks, especially sudden crashes with no comparable historical
state; early history before enough resolved labels exist; and fast V-shaped
declines where a next-open warning arrives too late.

Prior related trials counted for DSR:
At least the roughly 1,000 baseline parameter trials documented in the
repository, the approximately 1,500 bearish-composite configurations per
historical half, eleven documented challenger families, and this threshold
sensitivity set. This challenger is not statistically independent of those
trials.

Falsification rule:
Reject if the primary 50% rule misses either the 2008 or 2020 evaluable
peak-to-breach episode while invested, if its direction reverses across the
two historical halves, if 40%/50%/60% sensitivity is cliff-edge, if costs erase
the benefit, or if any guardrail fails. With fewer than 30 independent crash
episodes or completed trades, the rule cannot be promoted beyond research
tracking even if it passes historically.
