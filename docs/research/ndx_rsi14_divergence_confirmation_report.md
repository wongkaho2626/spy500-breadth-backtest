# Backtest Verification Report — NDX RSI(14) divergence confirmation

## Verdict: Reject

## Executive Summary

The RSI-confirmed challenger scores **40 / 100 (Weak)** versus baseline **40 / 100 (Weak)**. The decision follows the pre-registered Calmar objective and guardrails; historical splits are pseudo-OOS, not untouched forward evidence.

## Backtest Scores

| Component | Baseline | Challenger | Max |
|---|---:|---:|---:|
| A. Statistical validity | 23 | 23 | 30 |
| B. Risk-adjusted performance | 11 | 11 | 25 |
| C. Robustness / OOS | 25 | 18 | 25 |
| D. Trade quality / consistency | 16 | 16 | 20 |
| **Raw total** | **75** | **68** | **100** |
| Hard cap | 40 | 40 | |
| **Final score** | **40** | **40** | **100** |

## Performance and risk

| Metric | Baseline | Challenger | Delta |
|---|---:|---:|---:|
| CAGR | 20.27% | 19.86% | -0.41% |
| Volatility | 18.20% | 18.83% | +0.63% |
| Sharpe | 1.107 | 1.058 | -0.049 |
| Sortino | 0.942 | 0.929 | -0.013 |
| Calmar | 0.630 | 0.617 | -0.013 |
| Maximum drawdown | -32.18% | -32.18% | +0.00% |
| Ulcer Index | 5.80% | 6.21% | +0.40% |
| Time underwater | 85.24% | 85.36% | +0.13% |
| Win rate | 94.12% | 93.33% | -0.78% |
| Payoff ratio | 3.14 | 4.02 | +0.89 |
| Profit factor | 50.21 | 56.33 | +6.12 |
| Expectancy | 31.29% | 39.87% | +8.58% |
| Exposure | 73.19% | 78.53% | +5.34% |
| Turnover changes/year | 1.42 | 1.26 | -0.16 |
| Completed trades | 17 | 15 | -2 |

### Tail, drawdown, and benchmark diagnostics

| Metric | Baseline | Challenger |
|---|---:|---:|
| Omega (0% threshold) | 1.266 | 1.241 |
| Daily VaR 95% | -1.77% | -1.87% |
| Daily CVaR 95% | -2.73% | -2.80% |
| Daily VaR 99% | -3.30% | -3.35% |
| Daily CVaR 99% | -4.31% | -4.33% |
| Average drawdown episode | -2.49% | -2.55% |
| Average recovery | 15.9 sessions | 16.1 sessions |
| Maximum recovery | 353 sessions | 353 sessions |
| Pain ratio | 3.493 | 3.200 |
| Annual alpha vs NDX | 11.33% | 10.47% |
| Beta vs NDX | 0.609 | 0.653 |
| Correlation vs NDX | 0.781 | 0.808 |
| Information ratio | 0.389 | 0.396 |

## Statistical significance

- Effective observations: 5459 / 5586.
- t-stat: 5.153 / 4.981; PSR: 1.0000 / 1.0000.
- DSR after 4,820 trials: 0.9323 / 0.9061.
- Skewness: 0.313 / 0.264; excess kurtosis: 9.854 / 8.486.
- Jarque-Bera p: 0 / 0; Ljung-Box p: 3.08e-13 / 4.85e-11.
- Return ADF p: 1.13e-27 / 2.27e-28 (stationary when below 0.05).
- Paired annual mean: -0.23%; HAC t=-0.265, p=0.791.
- Block-bootstrap 95% interval: [-0.018627788811569207, 0.014204186184775163].

## Robustness

- Historical-half efficiency: 0.763 / 0.786 (pseudo-OOS).
- Sensitivity Calmar: rsi_le_45=0.614, rsi_le_50=0.617, rsi_le_55=0.621.
- Odd/even paired means: -0.29% / -0.17%.
- Trade bootstrap simulations: 5,000.
- Challenger trade-bootstrap terminal-return percentiles: p05=575.5%, p50=4636.9%, p95=55515.3%.
- Challenger trade-bootstrap max-drawdown percentiles: p05=-20.4%, p50=-10.8%, p95=0.0%.
- 5x/10x-cost paired means: -0.19% / -0.15%.
- Clean forward slice: 42 daily observations; not meaningful under the frozen plan's 3-year / 5-trade checkpoint.

### Regime attribution

| Regime | Delta CAGR | Delta Sharpe | Delta max DD | Delta Calmar |
|---|---:|---:|---:|---:|
| synthetic_breadth_2002_2006 | +0.18% | -0.018 | +0.00% | +0.010 |
| gfc_and_recovery_2007_2013 | -0.73% | -0.035 | +0.00% | -0.023 |
| pre_pandemic_2014_2019 | -0.00% | -0.000 | -0.00% | -0.000 |
| pandemic_and_inflation_2020_2022 | -5.50% | -0.254 | +0.00% | -0.253 |
| recent_2023_present | +2.87% | +0.026 | +0.00% | +0.211 |

## Bias assessment

| Bias | Status | Evidence |
|---|---|---|
| Lookahead bias | Absent | RSI ends at signal close; every changed exit fills next session open |
| Look-forward in features | Absent | RSI uses current and prior NDX closes only; no future label or shift(-1) enters the signal |
| Survivorship | Cannot fully verify | Aggregate NDX price index and breadth series; constituent history is not modeled |
| Data snooping / overfitting | Present, material | 4,820 prior/current trials penalized in DSR |
| Transaction costs | Included | $1 plus 0.05% per side, stressed at 2x/5x/10x |
| Liquidity | Low concern, not fully verified | Liquid index proxy; no position-size-versus-ADV model |
| Data frequency mismatch | Absent | Daily close signal and following daily open fill |
| Synthetic data | Present before 2007 | Real-breadth period is reported separately |
| Regime overfit | Tested, not eliminated | Historical halves, named regimes, odd/even years, and sensitivity |
| Clean forward OOS | Insufficient | Post-freeze sample is short and contains too few completed trades |

## Guardrails

- baseline_parity: PASS
- primary_calmar_improved: FAIL
- max_drawdown_not_worse: PASS
- cagr_within_two_points: PASS
- expectancy_not_worse: PASS
- turnover_not_increased: PASS
- historical_halves_calmar_nonnegative: FAIL
- real_breadth_calmar_nonnegative: FAIL
- five_x_paired_return_positive: FAIL
- sensitivity_not_cliff_edge: FAIL

## Current signal

As of 2026-09-02, NDX RSI(14) is 47.22. Canonical divergence is inactive; the combined challenger signal is inactive.

## Red Flags

1. Thousands of prior repository trials make historical improvements vulnerable to selection bias.
2. The RSI filter can delay a valid exit until after a sharp gap or until the trailing stop fires.
3. Pre-2007 breadth is synthetic, and post-freeze forward history is too short for adoption.

## Improvement Recommendations

1. Follow the pre-registered verdict and leave the frozen baseline unchanged.
2. If tracked, record the fixed RSI<=50 signal without retuning until the forward checkpoint.

## Decision

The verdict follows the primary Calmar objective and every pre-registered guardrail.

Research evidence only; not investment advice.
