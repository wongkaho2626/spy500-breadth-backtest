# S&P 500 20% drawdown peak Vector analysis

## Executive conclusion

1990-01-02至2026-07-30共有四個非重疊、由running peak跌至少20%的episode：
2000、2007、2020及2022。真正最高點與其附近±2% peak zone的主要共通點，
不是VIX急升或breadth崩潰，而是市場仍然呈現risk-on強勢：

1. NDX過去60日回報明顯高於普通市場日；
2. SPX仍在或非常接近252日高位；
3. SPX drawdown的20日斜率仍然向上，即回撤正在收窄或價格仍在推高；
4. 有breadth資料的三次peak，成分股高於200天線比例均偏高；
5. VIX水平、breadth 60日下跌幅度及breadth斜率沒有跨episode一致的惡化訊號。

這是一個描述性event study，不是已驗證的sell signal。只有四個獨立episode，
完整9維Vector更只有三個，不能由此推斷可靠預測機率。

## Definition

- Episode：從running all-time peak開始、首次close跌穿-20%，至其後重返前高；
  episode互不重疊。
- Peak zone：由最高點前126 sessions開始至首次-20% breach為止，所有close
  距episode最高close不超過2%的日期。
- 每個peak-zone日期都保留完整Vector。
- 共通點分析先對每個episode的peak-zone日期取median，再將episode等權，
  避免較長的頂部平台取得較大權重。
- Breadth由2002開始，因此2000 episode只有非breadth的6維Vector。

## Drawdown episodes

| Episode | Peak | First -20% breach | Trough | Recovery | Peak-to-trough | Peak-zone dates |
|---:|---|---|---|---|---:|---:|
| 1 | 2000-03-24 | 2001-03-12 | 2002-10-09 | 2007-05-30 | -49.15% | 26 |
| 2 | 2007-10-09 | 2008-07-09 | 2009-03-09 | 2013-03-28 | -56.78% | 27 |
| 3 | 2020-02-19 | 2020-03-12 | 2020-03-23 | 2020-08-18 | -33.93% | 16 |
| 4 | 2022-01-03 | 2022-06-13 | 2022-10-12 | 2024-01-19 | -25.43% | 17 |

合共有86個peak-zone交易日。2000年的zone由2000-03-22至2000-09-07；
2007年由2007-06-01至2007-10-31；2020年由2020-01-17至2020-02-21；
2022年由2021-11-08至2022-01-12。

## Exact peak vectors

| Peak | SPX daily | NDX 60d | Breadth | Breadth fall 60d | SPX DD252 | VIX |
|---|---:|---:|---:|---:|---:|---:|
| 2000-03-24 | +0.01% | +27.15% | N/A | N/A | 0.00% | 23.31 |
| 2007-10-09 | +0.81% | +7.06% | 67.14% | +7.71 pts | 0.00% | 16.12 |
| 2020-02-19 | +0.47% | +17.32% | 75.04% | -4.27 pts | 0.00% | 14.38 |
| 2022-01-03 | +0.64% | +10.77% | 73.46% | -6.60 pts | 0.00% | 16.60 |

| Peak | VIX slope 20 | Breadth slope 20 | SPX DD252 slope 20 |
|---|---:|---:|---:|
| 2000-03-24 | -0.010/day | N/A | +0.431/day |
| 2007-10-09 | -0.435/day | +0.857/day | +0.250/day |
| 2020-02-19 | -0.057/day | -0.033/day | +0.084/day |
| 2022-01-03 | -0.334/day | +0.301/day | +0.048/day |

Exact peak的直接共通點：

- 四次SPX當日回報全部為正；
- 四次NDX 60日回報全部為正，介乎+7.06%至+27.15%；
- 四次SPX均處於252日高位；
- 四次SPX drawdown slope全部為正；
- 四次VIX slope全部為負；
- 有breadth的三次peak均有67%以上成分股高於200天線。

換句話說，最高點當日通常仍然是「價格上升、波動下降、參與面廣」的市場，
而不是已經明顯轉熊。

## Peak-zone median commonalities

以下比較每個episode的peak-zone median與非peak-zone普通市場日median。

| Feature | Peak median | Ordinary median | Episode consistency | Robust effect |
|---|---:|---:|---:|---:|
| NDX return 60d | +11.56% | +4.49% | 4/4 above | +0.58 IQR |
| SPX drawdown 252d | -0.23% | -2.45% | 4/4 above | +0.35 IQR |
| SPX DD slope 20d | +0.077/day | +0.001/day | 3/4 above | +0.39 IQR |

呢三個係使用預先設定規則後的主要peak-zone共通特徵。

Breadth zone median為71.28%，普通日為64.95%，三個有資料episode全部偏高；
不過effect只有+0.24 IQR，略低於本分析的0.25 commonality cutoff，因此列為
次要而非主要共通點。

## Exact-peak effect versus ordinary days

| Feature | Exact-peak median | Ordinary median | Consistency | Robust effect |
|---|---:|---:|---:|---:|
| SPX DD slope 20d | +0.167/day | +0.001/day | 4/4 above | +0.85 IQR |
| NDX return 60d | +14.05% | +4.49% | 4/4 above | +0.78 IQR |
| VIX slope 20d | -0.195/day | -0.013/day | 3/4 below ordinary median; 4/4 negative | -0.80 IQR |
| SPX drawdown 252d | 0.00% | -2.45% | 4/4 above | +0.39 IQR |
| SPX daily return | +0.55% | +0.06% | 3/4 above ordinary median; 4/4 positive | +0.49 IQR |
| Breadth | 73.46% | 64.95% | 3/3 above | +0.33 IQR |

## What was not common

- VIX level：peak-zone median 17.53，普通日17.68，幾乎沒有差異。
- Breadth fall 60d：2007惡化，但2020及2022反而改善，方向不一致。
- Breadth slope 20d：2007及2022上升，2020輕微下降，方向不一致。
- VIX slope across the whole peak zone：只有3/4低於普通日median；它在exact
  peak較一致，但不是整個頂部區域都一致。
- 單日SPX回報：exact peak全部正數，但peak-zone effect較小，不能單獨使用。

## Interpretation

呢四次跌市的峰頂較似「強勢市場最後階段」，而唔係「已經開始崩壞」：

- Momentum仍強；
- SPX仍貼近新高；
- Breadth通常仍高；
- VIX通常正常甚至正在下降。

所以使用VIX高、breadth急跌、drawdown惡化的Vector，在真正最高點附近天然較難
發出sell signal。這亦與9維Trajectory crash predictor在2008及2020未能達到
門檻的結果一致。

較合理的後續研究方向不是將「高VIX／差breadth」當頂部必要條件，而是測試
「強勢頂部之後首次狀態轉折」：例如NDX momentum由高位減速、VIX slope由負轉
正、drawdown slope由正轉負。不過這需要新一輪預註冊及回測，不能由今次四個
episode直接視為有效規則。

## Limitations

- 只有四個獨立20% drawdown episodes。
- 完整breadth Vector只有2007、2020、2022三個episode。
- Peak-zone內每日觀察高度相關，不能當86個獨立樣本。
- Exact peaks由未來完整episode識別，只可用於描述及建立假設，不能直接作為
  實時訊號；實時市場無法知道今日是否最終最高點。
- 沒有交易規則、equity curve或獨立OOS，因此Backtest Score不適用。

研究結果只屬歷史描述，不是投資建議。
