# Pre-registered idea: breadth + leveraged MA200 trend satellite

Failure mode:
The 70% breadth / 30% unlevered MA200 ensemble reached 76 points and improved
Sharpe, Calmar, drawdown, monthly consistency, and independent event count.
It failed because the lower-return unlevered trend sleeve reduced paired
annual return by about 1.5 percentage points.  The useful independent trend
decisions need more capital efficiency without surrendering the risk gains.

Causal hypothesis:
A smaller 15% TQQQ sleeve using the same conventional NDX MA200 state should
deliver roughly the directional beta of a larger unlevered trend sleeve while
leaving 85% in the high-expectancy frozen breadth strategy.  The trend gate
keeps the leveraged sleeve out during sustained below-MA200 declines, so the
incremental return may be positive without restoring the full buy-and-hold
TQQQ drawdown.

Signal change:
No frozen signal changes.  Allocate 85% of initial capital to the canonical
breadth strategy and 15% to an independently compounded TQQQ sleeve.  The
TQQQ sleeve is invested when NDX closes above its trailing 200-session moving
average and is in cash below it.  Both buckets fill on the next session open,
pay their own costs, and are never rebalanced.

Entry or exit only:
Portfolio-allocation hypothesis.  The added sleeve uses one symmetric MA200
trend-state rule; the frozen component is unchanged.

Data available at decision time:
NDX close and its trailing 200-session average through the signal close;
actual TQQQ close/open from 2010-02-11 onward; before inception, a causal
3×-NDX daily-return proxy less an overlap-calibrated constant drag, with
synthetic opens based on the NDX overnight gap.

Primary metric:
Final Backtest Score under the installed `backtest-analyst` rubric.  At least
80 with no fatal cap is mandatory.  Positive paired return, Sharpe, and Calmar
are economic co-primary guardrails.

Guardrails:
- exact canonical parity with a 0% TQQQ sleeve;
- close signals and next-session-open execution;
- commission and slippage on every component transaction;
- at least 30 exit-event clusters after combining exits within 21 sessions;
- component daily-return correlation below 0.95;
- full-period CAGR, Sharpe, and Calmar all improve;
- maximum drawdown does not worsen by more than 2 percentage points;
- paired annual mean is positive and its 21-session block-bootstrap 95%
  interval excludes zero;
- positive Sharpe and Calmar deltas in both 2002-2013 and 2014-present;
- positive Sharpe and Calmar deltas in the 2007+ real-breadth era;
- positive paired return at 5x transaction costs;
- 10%/15%/20% initial sleeve sensitivity is directionally stable and no
  neighbouring score changes by more than 5 points;
- clustered profit factor above 1.2 and expectancy positive;
- tripling the pre-inception synthetic TQQQ drag preserves positive paired
  return, does not worsen maximum drawdown beyond the 2-point guardrail, and
  retains a final score of at least 80.

Single threshold and fixed sensitivity values:
- primary initial TQQQ trend sleeve: 15%;
- sensitivity only: 10% and 20%;
- NDX moving average: conventional 200 sessions, fixed;
- independent-event cluster: 21 sessions, fixed;
- synthetic-drag stress: calibrated 1× and punitive 3×, fixed.

Expected helpful regimes:
Long technology bull trends, recoveries that begin before the next breadth
washout, and periods when the frozen breadth strategy is flat above MA200.

Expected failure regimes:
MA200 whipsaws amplified by leverage, volatility decay, pre-2010 proxy error,
and crashes that gap through MA200 before the next-open exit.

Prior related trials counted for DSR:
At least 4,597 related signal/vector/allocation trials, including standalone
MA200 and TQQQ systems, leverage and volatility-management studies, allocation
grids, washout boost, monthly regime exit, and the 76-point unlevered
core-satellite ensemble.  The exact 85/15 conditional TQQQ ensemble was not
found, but it receives the full multiplicity penalty.

Falsification rule:
Reject if any parity, timing, cost, independent-event, correlation, score,
CAGR/Sharpe/Calmar, drawdown, paired-inference, historical-half, real-breadth,
5x-cost, sensitivity, trade-quality, or 3×-drag guardrail fails.  Passing
historical tests can only justify forward tracking; the frozen baseline cannot
be modified before meaningful clean post-2026-07-05 evidence and a user
adoption decision.
