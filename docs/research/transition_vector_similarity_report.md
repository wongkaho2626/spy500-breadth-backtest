# 三重轉向 Vector 相似度分析

## 結論

「NDX momentum 開始減速、VIX slope 由負轉正、SPX drawdown slope 由正轉負」確實可以轉成 vector，但三個 crossing 本身只會得到固定的 binary vector `[1, 1, 1]`，每次訊號都完全一樣，無法計算有意義的相似度。因此本分析使用 crossing 後的連續 slope 值：

`[NDX 60日回報的20日斜率, VIX的20日斜率, SPX 252日回撤的20日斜率]`

結果顯示，這種形態在歷史上很常見，很多沒有隨後出現 20% 跌市的訊號都非常相似。嚴格只用當時已知歷史的 causal 比較，3 維相似度沒有預測能力；它不適合單獨作為 sell filter。

## 方法

- 資料截至 2026-07-30。
- 原始訊號期間為 2003-10-02 至 2026-01-28，共 61 個獨立訊號群組。
- 正例定義：訊號後 126 個交易日內，SPX 最低回報小於或等於 -20%。
- 歷史 peak reference：4 個 SPX 大跌事件（2000、2007、2020、2022）最高點前後 2% 區域，共 86 個交易日 vector。
- 每一維以普通市場日的 median / IQR robust scaling，再計算 Euclidean distance。
- Similarity percentile 越高，代表比一般市場日更接近 peak-zone vector。
- Full-sample 是描述性 hindsight 比較；causal 版本只容許使用訊號日之前已經跌穿 -20% 的歷史事件，並只用訊號日前資料做 scaling。

## 3 維 trajectory vector 結果

| 比較 | 真訊號 | 假訊號 | AUC |
|---|---:|---:|---:|
| Full-sample median similarity | 77.57% | 63.18% | 0.764 |
| Causal median similarity | 57.45% | 60.10% | 0.379 |

AUC 0.5 約等於隨機排序。Causal AUC 0.379 表示這批資料中，較似舊 peak 的三維 trajectory 甚至沒有較高的跌市命中率。Full-sample 的較好數字主要來自未來資料：例如 2020-02-04 的 similarity 由 full-sample 98.33% 跌至 causal 25.76%，因為當日不可能預先使用尚未形成的 2020 peak-zone 作 reference。

## 有幾類同其他 vector

61 個原始訊號群組只有 3 個正例，而且全部屬於同一個 2020 跌市，並非 3 個獨立 crash。三個正例與最近的其他訊號如下：

| 訊號日 | 3D slope vector（NDX / VIX / drawdown） | 最近其他訊號 | 距離 | 最近訊號結果 |
|---|---|---|---:|---|
| 2019-12-03 | -0.0538 / +0.0495 / -0.0489 | 2017-01-03 | 0.0847 | 假訊號 |
| 2020-01-28 | -0.0372 / +0.0688 / -0.0451 | 2019-12-03 | 0.0917 | 同一個 2020 正例 |
| 2020-02-04 | -0.1606 / +0.2908 / -0.1213 | 2011-03-03 | 0.1733 | 假訊號 |

換言之，2019-12-03 和 2020-02-04 都可找到非常接近、但沒有隨後 20% 跌市的歷史 vector。相似形態本身不能分辨普通回調與真正熊市。

## 加入水平值的 6 維版本

若把 NDX 60日回報、VIX 水平及 SPX 回撤水平一併加入，形成 6 維 vector：

`[NDX60回報, NDX slope, VIX, VIX slope, SPX回撤, 回撤 slope]`

| 比較 | 真訊號 median | 假訊號 median | AUC |
|---|---:|---:|---:|
| Full-sample | 87.23% | 75.73% | 0.810 |
| Causal | 83.39% | 65.71% | 0.592 |

6 維 causal AUC 只略高於 0.5，而且仍只有一個獨立正例，不能視為已驗證的 prediction。水平值可能有少量額外資訊，但證據不足以訂出可靠 threshold。

## 實際執行 exit

10 日 confirmation 規則實際產生 12 次 exit，只有 2019-12-03 一次在 126 日內遇到 -20% SPX 跌幅，precision 為 8.33%。

| 3D comparison | 唯一真 exit | 11 個假 exit median |
|---|---:|---:|
| Full-sample similarity | 65.82% | 52.31% |
| Causal similarity | 57.45% | 49.52% |

兩組重疊很大，而且只有一個真 exit。現有證據不支持用這個相似度作 sell trigger 或 threshold。

## 最新 vector

2026-07-30 的 3D trajectory vector 為：

`[-1.1060, 約 0.0000, -0.1052]`

- 對 peak-zone 的 full-sample similarity：8.27%。
- 最近的歷史訊號：2003-11-24，標準化距離 0.8361；該次是假的 20% 跌市訊號。
- 6 維 peak-zone similarity：26.68%。

這代表最新狀態不像歷史 peak-zone，亦不構成高可信度 bearish-divergence prediction。

## 限制與判斷

- 4 個 peak episode、但只有 1 個被三重轉向命中的獨立 crash，樣本遠不足以訓練或驗證 prediction。
- Peak-zone 是事後標記；full-sample 相似度只可作描述，不能作 live performance 證據。
- 126 日 label 會把較遲才跌穿 -20% 的事件列為假訊號，例如某些早期警告可能超出 horizon。
- 86 個 peak-zone 日高度相關，不能當成 86 個獨立 crash。
- 本次只做診斷，沒有更改 frozen buy/sell baseline，也沒有挑選新的最佳 threshold。

最合理的下一步不是再調 distance cutoff，而是增加獨立事件資料（更長的 NDX/VIX proxy 歷史或跨市場 panel），並用 walk-forward、每次只留一個 crash episode 做驗證。
