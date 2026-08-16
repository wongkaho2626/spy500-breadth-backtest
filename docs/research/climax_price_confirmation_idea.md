# Pre-registered idea: price-confirmed climax exit

Failure mode:
The frozen climax-top rule fully exits QQQ when a 5% extension above the
10-day moving average and a bearish MACD cross occur within ten sessions.
Six of seven historical climax exits were followed by a higher NASDAQ-100
close within the next 126 sessions; several led to short exit/re-entry
whipsaws. The rule has momentum confirmation but no minimum pullback from a
recent high.

Causal hypothesis:
Requiring a modest close-to-close pullback from the trailing 10-session high
on the MACD-cross day will filter shallow momentum pauses while retaining
climax exits that show actual price weakness.

Signal change:
Keep the frozen extension and 10-session vote window, but count a bearish MACD
cross toward the climax exit only when the NASDAQ-100 close is at least 3%
below its trailing 10-session closing high.

Entry or exit only:
Exit only. Bearish-divergence and 25% trailing-stop exits remain unchanged.

Data available at decision time:
NASDAQ-100 closes, the trailing 10-session closing high, the existing
extension flag, and the existing MACD histogram cross through the signal
close. A signal at close fills at the next session open.

Primary metric:
Improve the full-period Calmar ratio and the paired challenger-minus-baseline
daily return without reducing CAGR.

Guardrails:
- baseline parity at a 0% confirmation threshold;
- no lookahead and next-session-open fills;
- maximum drawdown no more than two percentage points worse;
- positive cost-adjusted expectancy;
- no material increase in turnover;
- directionally non-negative CAGR and Calmar changes in both 2002–2013 and
  2014–2026 historical halves;
- result survives 5x costs.

Single threshold and fixed sensitivity values:
- primary pullback confirmation: 3%;
- sensitivity only: 2% and 4%;
- trailing high: 10 sessions, fixed to the existing extension horizon.

Expected helpful regimes:
Strong bull trends and post-washout rebounds where a MACD pause occurs near
the high but price has not actually broken down.

Expected failure regimes:
Fast crashes where waiting for a 3% pullback gives back gains, or slow rounded
tops where the ten-session climax vote expires before confirmation.

Prior related trials counted for DSR:
The approximately 1,000 documented baseline trials, the bearish-composite
search, eleven documented challenger families, the rejected vector searches,
the existing trailing-stop sweep, and the prior TQQQ partial-climax study.
The specific full-QQQ price-confirmation rule has not been found in the
repository, but it is not statistically independent of those trials.

Falsification rule:
Reject if the 3% rule lowers Calmar, lowers CAGR, reverses direction across
historical halves, worsens maximum drawdown by more than two points, has
cliff-edge 2%/3%/4% sensitivity, loses its benefit at 5x costs, or fails any
timing/parity audit. Track rather than adopt if historical tests pass but clean
forward evidence remains insufficient.
