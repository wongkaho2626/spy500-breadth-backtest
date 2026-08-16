# Backtest Verification Report — Twenty-session trajectory vector

## Verdict: Reject

加入 VIX、breadth 及 SPX drawdown 的二十日斜率，確實令 Vector 相對上一輪
六維 static 版本稍為改善，但仍未能超越 frozen baseline。按預註冊規則，
主要 Calmar 未改善、最大回撤技術上稍差、前後期效果反轉，以及五倍成本下
仍落後，因此不可採用。

## Backtest Scores

| Strategy | Raw score | Hard cap | Final score | Band |
|---|---:|---:|---:|---|
| Frozen baseline | 69 | 40 | 40 | Weak |
| Six-feature static filter | 59 | 40 | 40 | Weak |
| Nine-feature trajectory filter | 59 | 40 | 40 | Weak |

三者都少於三十次獨立 completed trades，因此觸發 40 分 hard cap。Trajectory
的細微改善不足以提升 robustness component，因為 historical split 仍然反轉，
亦沒有足夠 clean forward OOS。

## Hypothesis and implementation

原有六個 Vector 特徵、126-session outcome、十五個 monthly-independent
neighbours、robust scaling、60%主門檻及50%/70%敏感度全部保持不變。

只加入三個特徵：

- `vix_slope_20`：VIX 過去二十 sessions 的 OLS slope；
- `breadth_slope_20`：breadth 過去二十 sessions 的 OLS slope；
- `spx_drawdown_252_slope_20`：SPX 252日回撤的二十日 OLS slope。

每個斜率只使用 signal close 及之前十九個 sessions。Washout entry、所有賣出
規則、交易成本、cooldown、close signal 和 next-session-open fill 均沒有改動。

## Parity and causality

- Canonical baseline parity：equity 最大絕對差異 0，trades 及 open position
  完全一致。
- Recross gate 全部設為 True：與 baseline 完全一致。
- 把 cutoff 後的 VIX、breadth 及 SPX 改成極端值，cutoff 當日及之前的
  trajectory Vector 完全不變。
- 原有六個 static features 在 trajectory Vector 內逐值相同。
- 所有實際 trajectory recross entries 都通過當日門檻。

## Performance metrics

| Metric | Frozen baseline | Static 6D | Trajectory 9D |
|---|---:|---:|---:|
| CAGR | 20.18% | 19.26% | 19.44% |
| Annual volatility | 18.20% | 17.10% | 17.13% |
| Sharpe | 1.103 | 1.118 | 1.125 |
| Sortino | 0.936 | 0.870 | 0.878 |
| Calmar | 0.627 | 0.598 | 0.604 |
| Maximum drawdown | -32.18% | -32.19% | -32.19% |
| Ulcer Index | 5.81% | 6.19% | 5.65% |
| Time underwater | 85.18% | 85.87% | 85.79% |
| Exposure | 73.09% | 61.21% | 61.50% |
| Completed trades | 17 | 14 | 15 |
| Win rate | 94.12% | 92.86% | 93.33% |
| Profit factor | 50.21 | 48.61 | 48.96 |
| Mean closed-trade return | 31.29% | 36.75% | 34.55% |
| Positive months | 53.06% | 45.92% | 46.60% |

Trajectory 對 static Vector 有一致但很小的改善：CAGR +0.18 pp、Sharpe
+0.007、Calmar +0.006，Ulcer Index 改善 0.54 pp。不過相對 frozen baseline，
CAGR 仍低 0.73 pp、Calmar 低 0.023，Sortino 亦較差。

## Statistical comparison

### Trajectory versus frozen baseline

- Paired annualised mean-return difference：-0.80 pp。
- Newey-West HAC t-stat：-0.705；two-sided p = 0.481。
- 21-session block-bootstrap 95% interval：[-3.05 pp, +1.42 pp]。

### Trajectory versus static Vector

- Paired annualised mean-return difference：+0.16 pp。
- Newey-West HAC t-stat：0.914；two-sided p = 0.361。
- 21-session block-bootstrap 95% interval：[-0.06 pp, +0.54 pp]。

因此，軌跡特徵相對 static Vector 的改善未達統計顯著。以至少 573 個相關
prior trials 計算，trajectory DSR probability 約 98.17%；這支持策略絕對
Sharpe 高於 selection-adjusted benchmark，但不證明 trajectory 優於 frozen
baseline。

Ljung-Box lag 10 p = 4.58e-16，顯示日回報有 serial correlation；不能把所有
daily bars 當作獨立交易證據。Jarque-Bera 亦拒絕常態假設。

## Entry attribution

60% trajectory 版本有十四次 washout 及兩次 MA200-recross entries，包括
open position 共十六次。Static 版本只有一次 recross。

Trajectory 重新加入了 2004-05-14 的 recross，該交易至 2004-06-14 回報
+3.79%；這解釋了它相對 static Vector 的大部分改善。但它仍然錯過 baseline
在 2005-03-31 的 recross，而該交易回報 +33.40%。

在五個 baseline recross signal dates，trajectory probability 分別為：

| Signal date | Static probability | Trajectory probability | 60% pass |
|---|---:|---:|---|
| 2003-01-09 | unavailable | unavailable | No |
| 2004-03-24 | 42.59% | 44.05% | No |
| 2005-03-30 | 4.61% | 11.52% | No |
| 2018-10-12 | 32.29% | 49.67% | No |
| 2023-01-26 | 40.01% | 45.54% | No |

斜率普遍提高了這些 recross 的相似案例分數，但未能把最有價值的 2005
recross提升至合格水平。Nearest-neighbour 模型亦沒有強制「breadth上升、
VIX下降就一定更有利」的單調關係；新增維度只會改變歷史距離及鄰居組合。

## Robustness

### Threshold sensitivity

| Threshold | CAGR | Sharpe | Calmar | MDD | Closed trades |
|---:|---:|---:|---:|---:|---:|
| Baseline | 20.18% | 1.103 | 0.627 | -32.18% | 17 |
| 50% trajectory | 19.44% | 1.125 | 0.604 | -32.19% | 15 |
| 60% trajectory | 19.44% | 1.125 | 0.604 | -32.19% | 15 |
| 70% trajectory | 18.83% | 1.104 | 0.585 | -32.19% | 13 |

50%與60%結果平滑而接近，但兩者仍落後 baseline；70%刪除所有recross後進一步
惡化。

### Historical splits

| Period | Baseline CAGR | Trajectory CAGR | Difference | Sharpe difference |
|---|---:|---:|---:|---:|
| 2002-2013 | 17.54% | 14.48% | -3.06 pp | -0.077 |
| 2014-2026 | 22.84% | 24.47% | +1.63 pp | +0.105 |
| 2007+ real-breadth era | 22.59% | 22.86% | +0.27 pp | +0.035 |

早期仍然明顯落後、後期則領先，未解決 regime instability。

### Cost stress

| Cost multiplier | Baseline CAGR | Trajectory CAGR | Difference |
|---:|---:|---:|---:|
| 1x | 20.18% | 19.44% | -0.73 pp |
| 2x | 20.08% | 19.36% | -0.72 pp |
| 5x | 19.80% | 19.11% | -0.69 pp |
| 10x | 19.32% | 18.69% | -0.64 pp |

成本增加會稍為縮窄差距，但 challenger 在所有成本水平仍然落後。

### Trade bootstrap

5,000次 completed-trade bootstrap 的 terminal-return 中位數為：baseline
51.17倍、static 42.83倍、trajectory 45.07倍。Trajectory 比 static 改善，
但仍低於 baseline。只有十五次 trajectory closed trades，結果不精確。

## Bias assessment

| Bias | Status | Evidence |
|---|---|---|
| Lookahead | Absent | Trajectory只使用當日及之前十九日；label完整resolved後才訓練 |
| Survivorship | Cannot fully verify | 使用aggregate index及breadth |
| Data snooping | Present, material | 至少573個相關Vector variants |
| Transaction costs | Absent | 1x、2x、5x、10x已測 |
| Frequency mismatch | Absent | Close signal、下一session open fill |
| Synthetic breadth | Present before 2007 | 已獨立報告2007+結果 |
| Clean forward OOS | Insufficient | Freeze後未有足夠completed trades |

## Decision

Trajectory features 令 Vector 本身變得稍好，證明「方向資訊比單純snapshot有
額外內容」並非完全錯誤；但改善幅度不足，而且沒有解決最主要的錯誤：
Vector仍然否決多次有利的 baseline recross。

按預註冊 falsification rule，結果為 **Reject**。不應加入 frozen baseline，
亦不應在本輪再搜尋斜率窗口、加速度或門檻。

截至 2026-07-30：

- Static probability：44.88%；
- Trajectory probability：45.33%；
- Breadth slope：每日 +0.175 percentage points，正在改善；
- SPX drawdown slope：每日 -0.105 percentage points，回撤正在惡化；
- VIX slope 約 0；
- 當日沒有原始 MA200-recross，因此沒有買入訊號。

研究結果只屬回測證據，不是投資建議。
