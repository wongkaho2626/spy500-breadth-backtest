# Backtest Verification Report — Main-Force Washout Skill

Evidence timestamp: completed daily OHLCV through **2026-08-14**; US equities, daily timeframe; local Yahoo-derived archive and community point-in-time membership.

## Backtest Score: 20 / 100 — Reject

The rule is causal and the available-history result is reproducible, but missing delisted histories and ticker-reuse contamination impose the rubric's 20-point survivorship-bias cap.

| Component | Score | Max |
|---|---:|---:|
| A. Statistical Validity & Significance | 7 | 30 |
| B. Risk-Adjusted Performance | 0 | 25 |
| C. Robustness & Out-of-Sample | 14 | 25 |
| D. Trade Quality & Consistency | 3 | 20 |
| **Raw total** | **24** | **100** |
| Caps applied | Missing-history/survivorship bias → 20 | |
| **Final score** | **20** | **100** |

## Executive Summary

Across 660 usable point-in-time ticker histories, the completed-washout rule produced 846 non-overlapping trades. The equal-weight active-position portfolio returned -0.22% annualised with Sharpe 0.0979216311502172 and maximum drawdown -71.16%.
The median next-open event return was -0.43% at 20 sessions and 2.14% at 60 sessions. These figures describe the available subset, not an institutional point-in-time universe. Uninvested cash earns 0% in this test.

## Washout Signal Verification

1. **Signal 1 (MA Support): Strong.** Entry candidates had at least two distinct 20/30-SMA defenses inside 15 sessions, buying response, lighter pullback volume, and no recent high-volume 30-SMA break.
2. **Signal 2 (Volume Contraction): Strong.** Setup volume was below 50% of the prior-20-session median with a doji/narrow-range candle within the 1% MA zone and a stable close.
3. **Signal 3 (Order Book Test): Unknown.** The archive has no timestamped depth or time-and-sales data.
4. **Time & Market Risk Check:** Consolidation is measured as sessions since the latest rolling 63-session high; >21 is an opportunity-cost warning and >63 elevated risk. SPY must be above its 30-SMA or no worse than -5% over 20 sessions. Point-in-time sector mappings are unavailable.
5. **True vs. Fake Washout Assessment:** A later close above the prior 20-session high on >1.2× median volume is mandatory. High-volume/persistent 30-SMA failure invalidates the setup and exits next open.
6. **Final Verdict: Washout Completed (medium confidence)** only for rows in `signals.csv`; invalidated by a decisive/persistent 30-SMA loss. Confidence cannot be high without order-flow and complete histories.
7. **Suggested Action:** Treat a signal as educational confirmation only; avoid new exposure if the 30-SMA breaks on expanding volume. This is not financial advice.

## 1. Performance Metrics

| Metric | Strategy | SPY benchmark / threshold | Status |
|---|---:|---:|---|
| CAGR | -0.22% | 10.49% | Lower |
| Annual volatility | 21.57% | 19.20% | — |
| Sharpe | 0.0979216311502172 | >1 good | Below good |
| Sortino | 0.13790721376260784 | >1.5 good | Below good |
| Max drawdown | -71.16% | <20% good | High |
| Ulcer / lake ratio | 46.41% / 98.48% | Lake >50% red flag | — |
| VaR / CVaR 95% | 2.03% / 3.35% | — | — |
| Trades / win rate / PF | 846 / 34.63% / 0.9492272413911733 | PF>1.5 good | — |

## 2. Statistical Significance

Effective-N t-stat = 0.4685472714037495, p = 0.639410980481846; PSR vs zero = 70.59%. Jarque–Bera p = 0.0 and Ljung–Box(20) p = 6.699941884712476e-08. ADF is unavailable because `statsmodels` is not installed; no price-level regression is used. DSR was not required for the fixed skill-derived baseline; its score mirrors the t-stat tier. Per-ticker p-values are Benjamini–Hochberg adjusted in `per_ticker.csv`.

## 3. Bias Assessment

| Bias | Verdict | Evidence |
|---|---|---|
| Lookahead | Absent | Prior-session rolling baselines; close signals fill next open. |
| Survivorship / missing histories | **Present** | 381/1,206 archive symbols unavailable; final score capped at 20. |
| Ticker reuse | **Present, mitigated** | 157 downloaded mappings have no membership overlap and are excluded. |
| Overfitting / data snooping | Mitigated | Baseline thresholds fixed before results; sensitivity is diagnostic only. |
| Transaction costs | Partly verifiable | 10 bps/side baseline and 2×/5×/10× stress; spread/impact unavailable. |
| Liquidity | Cannot verify | No trade size, ADV participation, or historical spreads. |
| Frequency mismatch | Absent | Daily close decision, next-session adjusted-open fill. |
| Sector/regime | Cannot fully verify | SPY regime used; point-in-time sector mapping unavailable. |

## 4. Robustness

Fixed-rule 70/30 split: IS Sharpe 0.08001256930341068, OOS Sharpe 0.15305633793632084, efficiency 1.9129036758702929. The block-bootstrap 95% Sharpe interval is [-0.23772135066454692, 0.431365014136771]; sensitivity classification is **failed**. Cost-stress trade expectancy is recorded at 1×/2×/5×/10× in `summary.json`.

## 5. Red Flags 🚩

1. Missing delisted/acquired histories are not random and can materially bias both signal frequency and returns.
2. Community membership data and reused Yahoo symbols are not institutional-grade point-in-time data.
3. Signal 3 and sector-relative confirmation cannot be tested from these daily files.
4. Equal-weight portfolio turnover costs are approximate; capacity and market impact are unknown.
5. The SPY benchmark is close-to-close while strategy fills are adjusted open-to-open.

## 6. Improvement Recommendations 💡

1. Acquire CRSP/Compustat-quality delisted histories and permanent identifiers; rerun without changing rules.
2. Add timestamped order-book/tape archives to validate persistence, absorption, and follow-through.
3. Add point-in-time sector classifications and sector ETF total-return benchmarks.
4. Freeze these parameters prospectively and collect a genuinely unseen live sample.
5. Model spread and nonlinear market impact by historical ADV and signal-time volatility.

## 7. Verdict

**20/100 — Reject.** The available subset can test whether the OHLCV translation is causal and directionally useful, but it cannot establish a tradeable edge while the missing-history bias remains.
