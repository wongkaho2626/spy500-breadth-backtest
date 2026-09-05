# Backtest Verification Report — rolling-only exit

## Verdict: Reject

## Executive Summary

The 2% / 30-point / 70% rolling rule is the sole sell signal; climax-top and trailing-stop exits are disabled.

## Performance comparison

| Metric | Frozen baseline | Rolling + auxiliary exits | Rolling only |
|---|---:|---:|---:|
| CAGR | 20.26% | 18.98% | 16.28% |
| Sharpe | 1.107 | 1.007 | 0.814 |
| Sortino | 0.941 | 0.893 | 0.751 |
| Calmar | 0.630 | 0.590 | 0.328 |
| Maximum drawdown | -32.18% | -32.18% | -49.56% |
| Ulcer Index | 5.80% | 6.53% | 11.40% |
| Exposure | 73.19% | 81.53% | 89.54% |
| Completed trades | 17 | 14 | 5 |
| Expectancy | 31.29% | 43.89% | 104.67% |

## Backtest Score

Baseline raw/final: 75 / 40; rolling-only raw/final: 63 / 40.

## Statistical significance

- Versus baseline paired annual mean: -2.75%; HAC p=0.190; bootstrap 95% interval [-0.06855445938996678, 0.012848265906492842].
- Versus rolling auxiliary-on paired annual mean: -1.85%; HAC p=0.280.

## Robustness

- Sensitivity Calmar: drawdown_25=0.276, drawdown_30=0.328, drawdown_35=0.328.
- 5x/10x cost paired means: -2.54% / -2.27%.
- Historical-half efficiency: 0.763 / 0.783.

## Bias assessment

| Bias | Status | Evidence |
|---|---|---|
| Lookahead | Absent | Close features; next-open fill |
| Data snooping | Present | Rolling parameters were selected from the prior grid |
| Costs | Included | 1x/2x/5x/10x stress |
| Synthetic breadth | Present before 2007 | 2007+ reported separately |
| Clean forward OOS | Insufficient | Parameters and auxiliary removal are historically evaluated |

## Guardrails

- baseline_parity: PASS
- calmar_beats_baseline: FAIL
- calmar_beats_auxiliary_on: FAIL
- max_drawdown_not_worse: FAIL
- cagr_within_two_points: FAIL
- expectancy_not_worse: PASS
- historical_halves_calmar_nonnegative: FAIL
- real_breadth_calmar_nonnegative: FAIL
- five_x_paired_return_positive: FAIL
- sensitivity_not_cliff_edge: FAIL

## Current signal

As of 2026-09-01, signal is inactive; NDX return +0.41%, breadth drawdown 10.15 points, breadth 62.60%.

## Red Flags

1. Without a trailing stop, a persistent bear market can remain unprotected once the price-rise vote turns negative.
2. Removing climax exits also removes the special re-entry treatment following a climax exit.

## Verdict

The decision follows the pre-registered Calmar and drawdown guardrails.

Research evidence only; not investment advice.
