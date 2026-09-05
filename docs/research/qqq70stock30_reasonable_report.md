# QQQ 70 / Point-in-Time Top-1 30：合理統計重算

資料截至：2026-09-02；審核只使用本機快取資料。

## Backtest Score: 40 / 100 — Weak

已移除年度持倉的直接前視偏差，並只用 2007+ S5TH 時段計算風險；但只有 13 宗完成交易、沒有真正 OOS，因此分數受 40 分上限約束。

| Component | Score | Max |
|---|---:|---:|
| A. Statistical validity | 23 | 30 |
| B. Risk-adjusted performance | 15 | 25 |
| C. Robustness / OOS | 4 | 25 |
| D. Trade quality / consistency | 18 | 20 |
| Raw | **60** | **100** |
| Final after caps | **40** | **100** |

## 20 年 DCA 歷史描述（非預測）

| Metric | Corrected result |
|---|---:|
| Total contributions | $4,800,000 |
| Median terminal | $75,174,911 |
| 5th–95th percentile | $37,546,642 – $147,529,403 |
| Minimum–maximum | $29,488,281 – $177,351,011 |
| Median implied annual return | 19.97% |
| NDX buy-and-hold median | $28,104,895 |
| Structural effective n | 0.53 |
| Lag-1 correlation | 0.9993 |

The overlapping-window percent above $80m is deliberately not reported as a probability.

## 2007+ Clean-source risk profile（無 DCA）

| Metric | Strategy | NDX buy-and-hold |
|---|---:|---:|
| CAGR | 24.92% | 15.34% |
| Annual volatility | 20.41% | 22.67% |
| Sharpe | 1.195 | 0.744 |
| Sortino | 1.771 | 1.059 |
| Maximum drawdown | -35.54% | -53.71% |
| Completed trades | 13 | — |
| Win rate | 84.62% | — |
| HAC alpha p-value | 4.123e-05 | — |
| DSR after 4,819 trials | 0.945 | — |

## Block bootstrap（仍屬 in-sample model）

20-year terminal 5/50/95 percentiles: $44,668,906 / $158,910,933 / $546,862,994.
這是假設 2007+ 回報生成機制不變的區塊重抽樣，不可當實際成功概率。

## Planning scenarios

| Annual return | 20y nominal terminal | 3% inflation real value |
|---:|---:|---:|
| 8.00% | $13,613,350 | $7,537,382 |
| 10.00% | $17,982,500 | $9,956,474 |
| 12.00% | $23,856,782 | $13,208,922 |
| 15.00% | $36,655,254 | $20,295,125 |
| 18.00% | $56,518,629 | $31,292,994 |
| 20.00% | $75,475,200 | $41,788,788 |

## Verdict

Point-in-time correction lowers the historical 20-year median, but the estimate remains dominated by one market history and an in-sample signal search. Use the scenario table for planning; do not use the overlapping-window frequency or bootstrap frequency as a promised probability.
