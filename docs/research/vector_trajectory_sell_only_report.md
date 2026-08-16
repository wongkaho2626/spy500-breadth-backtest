# Backtest Verification Report — Nine-feature trajectory sell-only signal

## Verdict: Reject

9維Trajectory已經只套用於賣出，canonical washout及MA200-recross買入完全保留。
50%主門檻沒有產生任何 `vector-crash` exit，亦沒有在2008或2020 SPX跌穿
20%前離場。40%和60%敏感度結果同樣沒有持倉期間的Vector exit。刪除原有
bearish-divergence後，CAGR由20.18%跌至16.97%，最大回撤由-32.18%惡化至
-46.93%。

## Backtest Scores

| Strategy | Raw score | Hard cap | Final score | Band |
|---|---:|---:|---:|---|
| Frozen baseline | 69 | 40 | 40 | Weak |
| 9D trajectory sell-only | 53 | 40 | 40 | Weak |

兩者均少於30次獨立completed trades而觸發40分cap。Challenger raw score較低，
原因包括危機保護失敗、MDD惡化、沒有有效Vector exit、DSR下降及缺乏clean
forward OOS。

## Implementation and integrity

- 保留canonical washout及MA200-recross兩條買入路徑。
- 只以9維Trajectory crash probability取代bearish-divergence exit。
- 保留climax-top及25% trailing-stop。
- Crash target：未來126 sessions SPX由當日close下跌至少20%。
- Primary threshold：50%；固定敏感度40%及60%。
- Close signal，下一session open成交。
- 五個買入輸入欄位逐欄完全相同：
  `breadth`、`vix_vote`、`ma200_vote`、`vote_gate`、`ma200_recross`。
- Baseline replacement parity：equity最大絕對差異0，trades及open position
  完全一致。
- 未來Vector及label突變不會改變cutoff當日或之前的crash probability。

## Performance

| Metric | Frozen baseline | 9D trajectory sell-only | Difference |
|---|---:|---:|---:|
| CAGR | 20.18% | 16.97% | -3.20 pp |
| Annual volatility | 18.20% | 19.81% | +1.60 pp |
| Sharpe | 1.103 | 0.892 | -0.211 |
| Sortino | 0.936 | 0.811 | -0.125 |
| Calmar | 0.627 | 0.362 | -0.265 |
| Maximum drawdown | -32.18% | -46.93% | -14.74 pp |
| Ulcer Index | 5.81% | 8.99% | +3.18 pp |
| Time underwater | 85.18% | 85.66% | +0.49 pp |
| Exposure | 73.09% | 88.92% | +15.83 pp |
| Completed trades | 17 | 11 | -6 |
| Win rate | 94.12% | 90.91% | -3.21 pp |
| Profit factor | 50.21 | 34.08 | -16.13 |
| Mean closed-trade return | 31.29% | 68.90% | +37.61 pp |

平均trade return上升係因為策略持倉更長及交易更少，不能抵銷更大回撤及較低
複利回報。

## Signal diagnostics

Trajectory crash probability全歷史：

- Maximum：41.39%；
- 99th percentile：33.79%；
- 95th percentile：25.30%；
- 50%或60% signal days：0；
- 40% signal days：1，日期為2022-12-08，但策略當時已經OUT。

| Crisis window | Peak trajectory probability | Date | Vector exit before breach |
|---|---:|---|---|
| 2007-10-09至2008-07-09 | 32.84% | 2008-03-20 | No |
| 2020-02-19至2020-03-12 | 21.11% | 2020-03-12 | No |
| 2022-01-03至2022-06-13 | 35.14% | 2022-05-19 | No |

2008最終由trailing stop在2008-03-11平倉，但其後策略重新入場並在SPX首次
跌穿20%時仍有持倉。2020則沒有Vector exit，亦失去baseline在2020-02-26的
bearish-divergence保護。

9維50%結果與舊6維50% static crash Vector完全相同：兩者均沒有Vector exit。
軌跡特徵沒有改變主要決策。

## Exit attribution

| Exit reason | Baseline | Trajectory sell-only |
|---|---:|---:|
| Bearish divergence | 9 | Removed |
| Vector crash | 0 | 0 |
| Climax top | 7 | 8 |
| Trailing stop | 1 | 3 |

Trajectory版本實際退化成「只靠climax及trailing stop」的策略。

## Statistical comparison

- Paired annualised mean-return difference versus baseline：-2.40 pp。
- Newey-West HAC t-stat：-1.332；two-sided p = 0.183。
- 21-session block-bootstrap 95% interval：[-6.36 pp, +0.92 pp]。
- Challenger PSR versus zero：約99.999%。
- 以約4,587個相關prior trials計算，challenger DSR probability約69.90%，
  低於baseline約93.05%。
- Ljung-Box lag 10 p = 2.16e-7，日回報有serial correlation。
- Jarque-Bera拒絕常態分布。

沒有證據顯示challenger優於baseline，而且經濟效果明顯負面。

## Robustness

### Threshold sensitivity

| Threshold | Vector exits while invested | CAGR | Sharpe | Calmar | MDD |
|---:|---:|---:|---:|---:|---:|
| 40% | 0 | 16.97% | 0.892 | 0.362 | -46.93% |
| 50% | 0 | 16.97% | 0.892 | 0.362 | -46.93% |
| 60% | 0 | 16.97% | 0.892 | 0.362 | -46.93% |

三個門檻並非cliff-edge，而係全部無效。按照預註冊規則，不會事後測試30%或
其他更低門檻。

### Historical splits

| Period | Baseline CAGR | Challenger CAGR | Difference |
|---|---:|---:|---:|
| 2002-2013 | 17.54% | 14.36% | -3.18 pp |
| 2014-2026 | 22.84% | 19.60% | -3.23 pp |
| 2007+ real-breadth era | 22.59% | 18.42% | -4.17 pp |

方向沒有反轉，但challenger在每段都較差。

### Cost stress

| Cost multiplier | Baseline CAGR | Challenger CAGR | Difference |
|---:|---:|---:|---:|
| 1x | 20.18% | 16.97% | -3.20 pp |
| 2x | 20.08% | 16.91% | -3.17 pp |
| 5x | 19.80% | 16.72% | -3.08 pp |
| 10x | 19.32% | 16.40% | -2.92 pp |

交易成本不是失敗原因；即使10x成本，差距仍然巨大。

## Bias assessment

| Bias | Status | Evidence |
|---|---|---|
| Lookahead | Absent | 20日斜率向後望；126日label完全resolved後才訓練 |
| Buy-rule contamination | Absent | 五個買入欄位逐欄相同 |
| Survivorship | Cannot fully verify | Aggregate index及breadth無完整constituent history |
| Data snooping | Present, severe | 數千個相關baseline、composite及Vector trials |
| Transaction costs | Absent | 1x、2x、5x、10x已測 |
| Frequency mismatch | Absent | Close signal、next-session-open fill |
| Synthetic breadth | Present before 2007 | 已報告2007+結果 |
| Clean forward OOS | Insufficient | 沒有足夠post-freeze trades或resolved crash labels |

## Decision

50%主門檻同時錯過2008及2020，CAGR guardrail、MDD guardrail、五倍成本benefit
亦全部失敗。結論為 **Reject**，不應取代canonical bearish-divergence。

截至2026-07-30，static及trajectory crash probability均為0%，沒有Vector
sell signal。

研究結果只屬回測證據，不是投資建議。
