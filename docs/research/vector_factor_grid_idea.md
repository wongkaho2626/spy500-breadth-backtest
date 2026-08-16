# Pre-registered idea: vector-factor subset grid

Research round:
This is a new user-authorized round after the fixed six-factor vector exit was
rejected. The search is explicitly allowed to add or remove any of the six
existing factors, but it does not change the buy logic or mine new indicators.

Failure mode:
The six-factor, 50% nearest-neighbour exit never fired before the 2008 or 2020
S&P 500 20% breaches. A noisy or regime-specific factor may be diluting useful
similarity information.

Causal hypothesis:
At least one smaller subset of the existing six-dimensional market vector
separates pre-crash states from safe states more consistently than the complete
vector.

Candidate factors:

1. S&P 500 close-to-close daily return;
2. NASDAQ-100 60-session return;
3. percentage of S&P 500 constituents above their 200-day moving average;
4. 60-session breadth decline;
5. S&P 500 drawdown from its trailing 252-session high;
6. VIX close.

Search space:
- all 63 non-empty factor subsets;
- crash-risk probability thresholds 10%, 15%, 20%, 25%, 30%, 35%, 40%,
  45%, and 50%;
- 567 total configurations;
- 126-session label horizon, 20% future SPX decline, 15 neighbours, one nearest
  state per historical calendar month, robust median/IQR scaling: all fixed.

Execution and unchanged rules:
- signal at close, fill next session open;
- frozen commission and slippage;
- replace bearish divergence only;
- retain washout/trend-reentry buy logic, 15-day cooldown, climax-top exit, and
  25% trailing stop;
- keep `qqq_backtest.py` unchanged.

Historical selection design:
- early half: 2002-01-02 through 2013-12-31;
- late half: 2014-01-02 through 2026-07-29;
- select on early and report late, then select on late and report early;
- a configuration passes only if its crash-capture direction and guardrails
  hold in the opposite half;
- all observations through 2026-07-02 remain previously seen, so both
  directions are pseudo-out-of-sample, not clean forward evidence.

Primary event metric:
A true vector exit is a vector-triggered next-open sale whose signal date is
followed by a 20% SPX fall from that signal close within 126 sessions. For each
evaluable 20% SPX episode in a half, success additionally requires the strategy
to be out at the first 20% breach. Maximize true vector exits and protected
episodes while minimizing false vector exits.

Selection order:
1. require maximum drawdown no worse than baseline in the selection half;
2. require CAGR no more than two percentage points below baseline;
3. require no more than one additional round trip per year;
4. maximize protected evaluable crash episodes;
5. maximize vector-exit precision;
6. maximize Sharpe;
7. prefer fewer factors, then the higher threshold.

Guardrails and falsification:
- reject configurations with lookahead or baseline-parity failure;
- reject if the selected configuration misses the opposite-half evaluable
  crash that the baseline avoided;
- reject if performance improvement reverses across halves;
- reject if neighbouring thresholds materially reverse the conclusion;
- reject if 5x costs erase the benefit;
- do not adopt from historical grid results alone.

Multiplicity:
Count all 567 configurations, the prior fixed-vector thresholds, roughly 1,000
documented baseline trials, approximately 1,500 bearish-composite
configurations per half, and the eleven documented challenger families. Any
historical winner is highly selection-biased and can be tracked only until
meaningful forward evidence accumulates.
