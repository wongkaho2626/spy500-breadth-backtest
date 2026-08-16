# Backtest Verification Report — Vector filter for MA200 re-entry

## Verdict: Reject

保留 washout、只用 Vector 過濾 MA200-recross 的 60% challenger 未能改善預先
指定的主要指標。Calmar 由 0.627 降至 0.598，CAGR 由 20.18% 降至
19.26%，最大回撤亦沒有改善。Sharpe 雖然由 1.103 輕微升至 1.118，但
challenger 的相對表現在 2002-2013 與 2014-2026 方向相反，未通過預先註冊
的穩定性條件。

## Backtest Scores

### Frozen baseline: 40 / 100 — Weak

| Component | Score | Max |
|---|---:|---:|
| A. Statistical validity and significance | 27 | 30 |
| B. Risk-adjusted performance | 11 | 25 |
| C. Robustness and out-of-sample | 15 | 25 |
| D. Trade quality and consistency | 16 | 20 |
| **Raw total** | **69** | **100** |
| Cap | Fewer than 30 independent completed trades | 40 |
| **Final score** | **40** | **100** |

### Vector recross filter: 40 / 100 — Weak

| Component | Score | Max |
|---|---:|---:|
| A. Statistical validity and significance | 27 | 30 |
| B. Risk-adjusted performance | 11 | 25 |
| C. Robustness and out-of-sample | 8 | 25 |
| D. Trade quality and consistency | 13 | 20 |
| **Raw total** | **59** | **100** |
| Cap | Fewer than 30 independent completed trades | 40 |
| **Final score** | **40** | **100** |

兩者都被少於 30 次獨立平倉交易的 hard cap 限制。相同 final score 不代表
兩者證據一樣好；challenger 的 raw score 較低，主要因為 historical split
方向反轉、所有敏感度門檻都落後基準，以及月度回報一致性下降。

## Idea card

- Failure mode：MA200-recross 的歷史期望值低於 washout，而 Vector-only
  challenger 錯誤地刪除了較強的 washout 路徑。
- Hypothesis：保留原有 washout，只讓 Vector 成功機率至少 60% 的
  MA200-recross 入場。
- Entry change only：所有 bearish-divergence、climax-top、25% trailing-stop
  賣出規則保持不變。
- Primary metric：Calmar；secondary metric：Sharpe。
- Primary threshold：60%；固定敏感度：50% 和 70%。
- Falsification：Calmar 無改善、MDD 變差、CAGR 落後超過 2 個百分點、
  前後期方向反轉、成本或敏感度失效，任何一項均不可通過。

完整預註冊內容見 `vector_ma200_recross_filter_idea.md`。

## Experiment design and parity

- 期間：2002-01-02 至 2026-07-30，共 6,180 bars。
- Vector：6 個原有因子、15 個每月獨立 nearest neighbours、expanding history。
- Outcome：未來 126 sessions 回報至少 +10%，期間跌幅不低於 -15%。
- Label 只有完整 126-session path 成為歷史後才可進入訓練集。
- Close 產生訊號，下一個交易日 open 成交。
- Commission 及 slippage 與 canonical baseline 完全相同。
- Baseline parity：equity 最大絕對差異 0；trade signatures 及 open trade
  完全一致。
- Gate 全部設為 True 時，challenger 與 canonical baseline 完全一致。

## Performance metrics

| Metric | Baseline | Vector filter 60% | Difference |
|---|---:|---:|---:|
| CAGR | 20.18% | 19.26% | -0.91 pp |
| Annual volatility | 18.20% | 17.10% | -1.10 pp |
| Sharpe | 1.103 | 1.118 | +0.015 |
| Sortino | 0.936 | 0.870 | -0.066 |
| Calmar | 0.627 | 0.598 | -0.028 |
| Maximum drawdown | -32.18% | -32.19% | -0.003 pp |
| Ulcer Index | 5.81% | 6.19% | +0.38 pp |
| Time underwater | 85.18% | 85.87% | +0.70 pp |
| Daily VaR 95% | -1.78% | -1.63% | +0.15 pp |
| Daily CVaR 95% | -2.74% | -2.63% | +0.11 pp |
| Exposure | 73.09% | 61.21% | -11.88 pp |
| Completed trades | 17 | 14 | -3 |
| Win rate | 94.12% | 92.86% | -1.26 pp |
| Profit factor | 50.21 | 48.61 | -1.60 |
| Mean closed-trade return | 31.29% | 36.75% | +5.46 pp |
| Positive months | 53.06% | 45.92% | -7.14 pp |

較高的平均交易回報主要來自刪除多次較短的正回報交易，不能抵銷長時間留在
現金造成的複利損失。MDD 數值差別經濟上極小，但按照預先註冊的「不可變差」
條件仍屬未通過。

## Entry attribution

Baseline 有 13 次 washout 及 5 次 MA200-recross 入場（包括現有 open
position 共 18 次）。60% challenger 有 14 次 washout 及 1 次
MA200-recross，共 15 次。

60% gate 否決了 baseline 原本五次 recross：

| Signal date | Baseline entry | Vector probability | Baseline trade return |
|---|---|---:|---:|
| 2003-01-09 | 2003-01-10 | unavailable | +1.36% |
| 2004-03-24 | 2004-03-25 | 42.59% | +5.38% |
| 2005-03-30 | 2005-03-31 | 4.61% | +33.40% |
| 2018-10-12 | 2018-10-15 | 32.29% | +24.47% |
| 2023-01-26 | 2023-01-27 | 40.01% | +10.10% |

由於狀態路徑改變，challenger 其後可在不同日期重新入場：例如
2018-12-24 的 washout 和 2023-02-01 的合格 recross。Washout 的規則本身
沒有改變；實際 washout 次數增加，是因為策略在那些日期仍然處於 OUT。
所有 challenger recross 均通過 60% gate，測試沒有發現違規入場。

## Statistical significance

- Paired annualised mean daily-return difference：-0.96 pp。
- Newey-West HAC t-stat：-0.835；two-sided p = 0.404。
- 21-session block-bootstrap 95% interval：[-3.20 pp, +1.27 pp]。
- Mean-return t-stat：baseline 5.13；challenger 5.12。
- PSR versus zero：兩者均約 100%。
- 以 570 個相關 prior trials 計算的 DSR probability：baseline 98.00%；
  challenger 98.00%。這只支持兩者絕對回報高於零，不支持 challenger
  優於 baseline。
- Ljung-Box lag 10 p-values 分別為 2.57e-13 和 3.22e-16，日回報存在顯著
  serial correlation；因此不能將 6,179 個 daily returns 當作完全獨立證據。
- Jarque-Bera p 值兩者均接近零，日回報不是常態分布。
- ADF 未執行，因本地沒有 `statsmodels`；本研究分析的是策略日回報而非
  非平穩價格水平，這不影響今次 entry-filter 的主要否證結果。

## Robustness

### Threshold sensitivity

| Threshold | CAGR | Sharpe | Calmar | MDD | Closed trades |
|---:|---:|---:|---:|---:|---:|
| Baseline | 20.18% | 1.103 | 0.627 | -32.18% | 17 |
| 50% | 18.57% | 1.075 | 0.577 | -32.19% | 15 |
| 60% | 19.26% | 1.118 | 0.598 | -32.19% | 14 |
| 70% | 18.83% | 1.104 | 0.585 | -32.19% | 13 |

三個固定門檻的 CAGR 及 Calmar 全部低於 baseline；結果並非只在單一門檻
失敗。

### Historical splits

| Period | Baseline CAGR | Challenger CAGR | Difference | Sharpe difference |
|---|---:|---:|---:|---:|
| 2002-2013 | 17.54% | 14.13% | -3.42 pp | -0.093 |
| 2014-2026 | 22.84% | 24.47% | +1.63 pp | +0.105 |
| 2007+ real-breadth era | 22.59% | 22.86% | +0.27 pp | +0.035 |

改善只集中在後期，早期明顯變差，違反預註冊的方向一致要求。這些仍然只是
pseudo-OOS robustness splits，並非真正未見過的 forward OOS。

### Cost stress

| Cost multiplier | Baseline CAGR | Challenger CAGR | Difference |
|---:|---:|---:|---:|
| 1x | 20.18% | 19.26% | -0.91 pp |
| 2x | 20.08% | 19.18% | -0.90 pp |
| 5x | 19.80% | 18.95% | -0.85 pp |
| 10x | 19.32% | 18.55% | -0.77 pp |

較低 turnover 令相對落後在高成本時略為縮窄，但即使 10x 成本仍未反勝。

### Trade bootstrap

5,000 次 completed-trade bootstrap 中，baseline terminal-return 中位數為
51.17 倍，challenger 為 42.83 倍；5th percentile 分別為 9.11 倍和
7.65 倍。兩者 trade-level MDD 5th percentile 均約 -20.45%。樣本只有
17 和 14 次 completed trades，區間不可視為精確估計。

## Bias assessment

| Bias | Status | Evidence |
|---|---|---|
| Lookahead | Absent | 126-session label 完全 resolved 後才可訓練；下一日 open fill |
| Look-forward features | Absent | 所有 Vector 特徵只用 signal close 或更早資料 |
| Survivorship | Cannot fully verify | 使用 aggregate index 及 breadth，無完整 constituent history |
| Data snooping | Present, material | 相關 Vector 研究約 570 個 prior variants |
| Transaction costs | Absent | 1x、2x、5x、10x 已測 |
| Liquidity | Low concern / incomplete | QQQ 模擬規模小，但未按 ADV 建模 |
| Frequency mismatch | Absent | Daily close signal 對 next-session open fill |
| Synthetic breadth | Present before 2007 | 已獨立報告 2007+ 結果 |
| Clean forward OOS | Insufficient | Freeze 後只有 19 bars，沒有足夠 completed trades |

## Red flags

1. Primary Calmar、CAGR、Ulcer Index 和 time underwater 均變差。
2. 早期與後期的相對效果反轉。
3. Vector 否決的五次 baseline recross 全部最終是正回報，包括
   2005-2007 的 +33.40%。
4. 只有 14 次 challenger closed trades，獨立樣本極少。
5. 相關 Vector 試驗數量龐大，任何小幅歷史改善都需要嚴重 multiplicity
   discount。

## Decision

按照 idea card，primary Calmar 無改善、MDD 技術上變差、historical split
方向反轉，以及 5x 成本下仍落後，故必須 **Reject**。不應將這個 60%
Vector filter 加入 frozen baseline，亦不應再用相同歷史調整門檻尋找勝者。

截至 2026-07-30，Vector buy probability 為 44.88%，當日亦沒有原始
MA200-recross；所以當日不會產生 filtered recross buy。

研究結果只屬回測證據，不是投資建議。
