# Backtest Verification Report — Triple trajectory reversal sell

## Verdict: Reject

以NDX momentum slope由正轉負、VIX slope由負轉正、SPX drawdown slope由正
轉負組成的10日sell rule，未能在2008或2020首次跌穿20%前提供有效保護。
它產生12次transition exits，只有一次之後126日SPX跌20%，precision僅8.3%。
CAGR由20.18%跌至13.44%，Calmar由0.627跌至0.418。

## Backtest Scores

| Strategy | Raw score | Hard cap | Final score | Band |
|---|---:|---:|---:|---|
| Frozen baseline | 69 | 40 | 40 | Weak |
| Triple trajectory reversal | 44 | 40 | 40 | Weak |

兩者均少於30次獨立completed trades，觸發40分hard cap。Challenger raw score
明顯較低，主要由大量false exits、低月度一致性、低DSR及沒有危機保護造成。

## Signal definition

Primary sell需要：

1. NDX trailing-60-session return的20日OLS slope由非負轉為負；
2. VIX 20日OLS slope由非正轉為正；
3. SPX trailing-252-session drawdown的20日OLS slope由非負轉為負；
4. 三個cross在最近10 sessions內發生；
5. 當時NDX 60日回報仍為正；
6. SPX距252日高位不超過2%。

Close產生訊號、下一session open成交。它只取代bearish-divergence；washout、
MA200-recross、climax-top及25% trailing stop保持不變。

## Integrity

- Baseline parity：equity最大絕對差異0，trade signatures及open position一致。
- `breadth`、`vix_vote`、`ma200_vote`、`vote_gate`及`ma200_recross`逐欄一致。
- 未來NDX、SPX或VIX突變不會改變cutoff當日或之前的features及crosses。
- 4項單元測試全部通過。

## Performance

| Metric | Baseline | 10-day transition | Difference |
|---|---:|---:|---:|
| CAGR | 20.18% | 13.44% | -6.74 pp |
| Annual volatility | 18.20% | 16.83% | -1.37 pp |
| Sharpe | 1.103 | 0.835 | -0.268 |
| Sortino | 0.936 | 0.530 | -0.406 |
| Calmar | 0.627 | 0.418 | -0.209 |
| Maximum drawdown | -32.18% | -32.19% | -0.001 pp |
| Ulcer Index | 5.81% | 5.72% | -0.09 pp |
| Time underwater | 85.18% | 86.10% | +0.92 pp |
| Exposure | 73.09% | 40.42% | -32.67 pp |
| Completed trades | 17 | 21 | +4 |
| Win rate | 94.12% | 85.71% | -8.40 pp |
| Profit factor | 50.21 | 28.55 | -21.66 |
| Positive months | 53.06% | 32.65% | -20.41 pp |

較低volatility及略低Ulcer Index來自長時間留在現金，不足以補償大幅複利
回報損失。

## Exit quality

10日規則有12次transition exits：

| Signal date | Exit date | Future SPX minimum, 126d | Followed by -20% |
|---|---|---:|---|
| 2004-12-08 | 2004-12-09 | -3.83% | No |
| 2006-11-08 | 2006-11-09 | -0.84% | No |
| 2010-12-01 | 2010-12-02 | +1.28% | No |
| 2012-04-05 | 2012-04-09 | -8.59% | No |
| 2013-02-27 | 2013-02-28 | -0.09% | No |
| 2015-01-22 | 2015-01-23 | -3.31% | No |
| 2016-06-21 | 2016-06-22 | -4.23% | No |
| 2019-12-03 | 2019-12-04 | -27.67% | Yes |
| 2021-03-01 | 2021-03-02 | -3.42% | No |
| 2023-07-07 | 2023-07-10 | -6.40% | No |
| 2024-02-09 | 2024-02-12 | -1.46% | No |
| 2025-08-04 | 2025-08-05 | -0.49% | No |

Precision為1/12，即8.3%。2019-12-03係唯一true label，但它距2020 peak尚早；
策略在2020-03-02重新買入，因此2020-03-12首次跌穿20%時仍有持倉。換言之，
即使label為true，交易狀態路徑仍沒有保護2020危機。

2008亦沒有transition exit，最終仍依賴trailing stop。

## Sensitivity

| Confirmation window | CAGR | Sharpe | Calmar | MDD | Transition exits | Precision |
|---:|---:|---:|---:|---:|---:|---:|
| 5 sessions | 14.49% | 0.860 | 0.309 | -46.93% | 10 | 10.0% |
| 10 sessions | 13.44% | 0.835 | 0.418 | -32.19% | 12 | 8.3% |
| 20 sessions | 12.18% | 0.781 | 0.378 | -32.19% | 12 | 0.0% |

三個窗口全部明顯落後baseline。5日窗口雖然在2020-01-28發出true exit，但
其後同樣於2020-03-02重新入場，而且2008 MDD惡化至-46.93%。

## Historical splits

| Period | Baseline CAGR | 10-day CAGR | Difference |
|---|---:|---:|---:|
| 2002-2013 | 17.54% | 12.81% | -4.73 pp |
| 2014-2026 | 22.84% | 14.05% | -8.78 pp |
| 2007+ real-breadth era | 22.59% | 14.12% | -8.48 pp |

方向沒有反轉，因為challenger在所有period都較差。

## Statistical comparison

- Paired annualised mean-return difference：-6.02 pp。
- Newey-West HAC t-stat：-3.173；two-sided p = 0.0015。
- 21-session block-bootstrap 95% interval：[-9.64 pp, -2.41 pp]。
- Challenger DSR probability在約4,588個related trials後約47.69%，baseline
  約93.05%。
- Ljung-Box lag 10 p約1.38e-43，日回報存在強serial correlation。
- Jarque-Bera拒絕常態分布。

Challenger相對baseline的負面差異具有統計證據，而不是單純噪音。

## Cost stress

| Cost multiplier | Baseline CAGR | Challenger CAGR | Difference |
|---:|---:|---:|---:|
| 1x | 20.18% | 13.44% | -6.74 pp |
| 2x | 20.08% | 13.33% | -6.75 pp |
| 5x | 19.80% | 13.00% | -6.79 pp |
| 10x | 19.32% | 12.46% | -6.87 pp |

成本會令高turnover challenger進一步惡化。

## Current status

截至2026-07-30：

- NDX 60日回報：+1.64%；
- NDX momentum slope：-1.106/day；
- VIX slope：約0；
- SPX drawdown：-2.28%；
- SPX drawdown slope：-0.105/day；
- 最近NDX deceleration cross：16 sessions前；
- 最近VIX slope up cross：47 sessions前；
- 最近drawdown slope down cross：3 sessions前；
- near-high gate未通過，三個cross亦未在10日內聚集；
- Sell signal：未觸發。

## Decision

2008及2020保護、CAGR、MDD、五倍成本benefit等預註冊guardrails均未全部通過。
結論為 **Reject**。不應取代canonical bearish-divergence，亦不應在本輪事後
調整2% gate、slope window或cross組合。

研究結果只屬回測證據，不是投資建議。
