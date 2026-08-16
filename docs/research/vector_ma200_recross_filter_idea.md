# Vector filter for MA200 trend re-entry

Failure mode:
The baseline MA200-recross entries have weaker historical expectancy than
washout entries, while replacing every buy rule with the vector discarded the
stronger washout path and materially reduced performance.

Causal hypothesis:
A causal nearest-neighbour estimate of a favourable six-month NDX outcome can
veto weak MA200 trend re-entries without disturbing high-conviction washout
entries. The vector is used only as a confirmation filter, not as a standalone
entry signal.

Signal change:
Keep the canonical washout entry exactly unchanged. Permit an otherwise-valid
MA200-recross entry only when the close-of-day vector buy probability is at
least 60%. Keep all canonical exit rules unchanged.

Entry or exit only:
Entry only.

Data available at decision time:
SPX close-to-close daily return, NDX trailing 60-session return, S&P 500
percentage above the 200-day moving average, its trailing 60-session decline,
SPX drawdown from its trailing 252-session high, and VIX. Training labels enter
the neighbour pool only after the full 126-session outcome path has resolved.

Primary metric:
Calmar ratio, with Sharpe ratio as the secondary risk-adjusted metric.

Guardrails:
- Baseline parity must pass exactly.
- No lookahead: close signal, next-session-open fill.
- Washout entries and all sell rules remain unchanged.
- Maximum drawdown must not worsen.
- CAGR may not trail the baseline by more than 2 percentage points.
- Completed trades and turnover may not increase.
- Challenger-minus-baseline performance must not reverse sign between
  2002-2013 and 2014-present historical splits.
- The result must remain directionally intact at five times modeled costs.

Single threshold and fixed sensitivity values:
Primary probability threshold 60%; sensitivity thresholds 50% and 70%.
No feature-subset search and no additional threshold search.

Expected helpful regimes:
Failed rebounds and weak trend resumptions after defensive exits, especially
when cross-market breadth and volatility resemble historically poor entries.

Expected failure regimes:
Fast V-shaped recoveries, secular bull markets whose current vector has few
close historical analogues, and early history that relies on synthetic breadth.

Prior related trials counted for DSR:
The prior six-factor vector crash-exit grid contained 567 configurations, and
the prior vector-only buy challenger tested three thresholds. These trials are
not independent; the challenger therefore receives a severe multiple-testing
penalty.

Falsification rule:
Reject if baseline parity fails, primary Calmar does not improve, any guardrail
fails, the historical split effect reverses, or the 50%/60%/70% sensitivity
results show a cliff-edge. With insufficient clean post-freeze observations,
even a historical pass remains a research challenger rather than eligible for
adoption.
