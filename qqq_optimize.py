"""Grid search for NASDAQ 100 breadth+CAPE strategy parameters."""
import itertools, numpy as np, pandas as pd
from pathlib import Path

DATA_DIR      = Path(__file__).parent
BREADTH_START = pd.Timestamp('2007-01-02')

def _parse_price(s): return s.astype(str).str.replace(',','').astype(float)

def load_base() -> pd.DataFrame:
    ndx  = pd.read_csv(DATA_DIR / 'NASDAQ100.csv')
    ndx['Date'] = pd.to_datetime(ndx['date']); ndx.set_index('Date', inplace=True)
    ndx  = ndx.rename(columns={'close':'price'}); ndx['price'] = ndx['price'].astype(float)
    b200 = pd.read_csv(DATA_DIR / 'S&P 500 Stocks Above 200-Day Average Historical Data.csv')
    b200['Date'] = pd.to_datetime(b200['Date'], format='%m/%d/%Y'); b200.set_index('Date', inplace=True); b200['Price'] = _parse_price(b200['Price'])
    b50  = pd.read_csv(DATA_DIR / 'S&P 500 Stocks Above 50-Day Average Historical Data.csv')
    b50['Date']  = pd.to_datetime(b50['Date'],  format='%m/%d/%Y'); b50.set_index('Date',  inplace=True); b50['Price']  = _parse_price(b50['Price'])
    cape = pd.read_csv(DATA_DIR / 'ShillerPE.csv')
    cape['Date'] = pd.to_datetime(cape['date']); cape.set_index('Date', inplace=True); cape = cape.rename(columns={'close':'cape'})
    merged = ndx[['price']].join(b200[['Price']].rename(columns={'Price':'breadth'}), how='left')
    merged = merged.join(b50[['Price']].rename(columns={'Price':'b50'}), how='left')
    merged = merged.join(cape[['cape']], how='left'); merged['cape'] = merged['cape'].ffill()
    merged.sort_index(inplace=True)
    return merged

def add_signals(df: pd.DataFrame, div_window: int, div_price_rise: float,
                div_breadth_fall: float, cap_expensive: float, cape_expensive: float) -> pd.DataFrame:
    d = df.copy()
    div_cap = d['cape'].apply(lambda c: cap_expensive if c >= cape_expensive else 55.0)
    pp = d['price'].shift(div_window); bp = d['breadth'].shift(div_window)
    d['bearish_div'] = (
        ((d['price'] - pp) / pp * 100 >= div_price_rise) &
        ((bp - d['breadth']) >= div_breadth_fall) &
        (d['breadth'] < div_cap)
    ).fillna(False)
    return d

def run(df: pd.DataFrame, cape_buy_abs: float, cape_sell_abs: float,
        buy_thresh_hi: float, cape_buy_high: float) -> tuple[pd.Series, list]:
    pos = 'OUT'; eff_entry = 0.0; entry_date = None; trade_low = 0.0; port = 10_000.0
    trades = []; values = {}
    for row in df.itertuples():
        date    = row.Index
        price   = row.price
        cape    = row.cape
        breadth = row.breadth
        b50     = row.b50
        bdiv    = row.bearish_div
        has_b   = date >= BREADTH_START and not (breadth != breadth)  # fast isnan check
        in_p2   = date >= BREADTH_START
        if pos == 'OUT':
            if has_b:
                ab = buy_thresh_hi if cape > cape_buy_high else 18.0
                do_buy = breadth < ab and b50 < 25.0
            elif not in_p2:
                do_buy = cape < cape_buy_abs
            else:
                do_buy = False
            if do_buy:
                port -= 1; eff_entry = price * 1.0005; entry_date = date; trade_low = price; pos = 'IN'
        elif pos == 'IN':
            trade_low = price if price < trade_low else trade_low
            if has_b:       do_sell = bdiv
            elif not in_p2: do_sell = cape > cape_sell_abs
            else:           do_sell = False
            if do_sell:
                eff_exit = price * 0.9995; gr = (eff_exit - eff_entry) / eff_entry
                port *= (1 + gr); port -= 1
                trades.append({'r': gr * 100, 'days': (date - entry_date).days})
                pos = 'OUT'
        values[date] = port * (price * 0.9995 / eff_entry) if pos == 'IN' else port
    if pos == 'IN':
        lp = df['price'].iloc[-1]; gr = (lp * 0.9995 - eff_entry) / eff_entry
        trades.append({'r': gr * 100, 'days': (df.index[-1] - entry_date).days})
    return pd.Series(values), trades

def score(values: pd.Series, trades: list) -> dict:
    if len(values) < 2 or not trades:
        return {'sharpe': -99, 'cagr': -99, 'max_dd': -99, 'n': 0, 'wr': 0, 'minh': 0}
    dr    = values.pct_change().dropna()
    years = (values.index[-1] - values.index[0]).days / 365.25
    cagr  = (values.iloc[-1] / values.iloc[0]) ** (1 / years) - 1
    mdd   = ((values - values.cummax()) / values.cummax()).min()
    sh    = (dr.mean() / dr.std() * np.sqrt(252)) if dr.std() > 0 else 0
    wins  = sum(1 for t in trades if t['r'] > 0)
    minh  = min(t['days'] for t in trades)
    return {'sharpe': sh, 'cagr': cagr, 'max_dd': mdd, 'n': len(trades),
            'wr': wins / len(trades), 'minh': minh}

if __name__ == '__main__':
    base = load_base()

    # Focused grid — key levers for fixing NASDAQ whipsaws
    grid = {
        'div_window':      [100, 120, 150],
        'div_price_rise':  [1.0, 2.0, 3.0],
        'div_breadth_fall':[15.0, 20.0, 25.0],
        'cap_expensive':   [40.0, 45.0, 50.0],
        'cape_expensive':  [28.0, 30.0, 32.0],
        'cape_buy_abs':    [20.0, 22.0],
        'cape_sell_abs':   [28.0, 30.0],
        'buy_thresh_hi':   [10.0, 12.0],
        'cape_buy_high':   [28.0, 30.0],
    }
    combos = list(itertools.product(*grid.values()))
    keys   = list(grid.keys())
    print(f'Testing {len(combos)} combos...')

    cache   = {}
    results = []
    for combo in combos:
        p  = dict(zip(keys, combo))
        dk = (p['div_window'], p['div_price_rise'], p['div_breadth_fall'],
              p['cap_expensive'], p['cape_expensive'])
        if dk not in cache:
            cache[dk] = add_signals(base, *dk)
        vals, trades = run(cache[dk], p['cape_buy_abs'], p['cape_sell_abs'],
                           p['buy_thresh_hi'], p['cape_buy_high'])
        s = score(vals, trades)
        results.append({**p, **s})

    results.sort(key=lambda r: (round(r['sharpe'], 2), -r['max_dd']), reverse=True)

    print(f"{'='*120}")
    print('Top 20 by Sharpe then Max Drawdown  (min hold >= 5 days)')
    print(f"{'='*120}")
    print(f"{'Sh':>5}  {'CAGR':>6}  {'MDD':>7}  {'#T':>3}  {'WR':>4}  {'minH':>5}"
          f"  {'cap€':>5}  {'C€':>4}  {'Cba':>3}  {'Csa':>3}  {'bthi':>4}  {'cbhi':>4}"
          f"  {'dw':>4}  {'dpr':>4}  {'dbf':>4}")
    print('-' * 120)
    for r in results[:20]:
        print(f"{r['sharpe']:>5.2f}  {r['cagr']:>6.1%}  {r['max_dd']:>7.1%}"
              f"  {r['n']:>3}  {r['wr']:>4.0%}  {r['minh']:>5}"
              f"  {r['cap_expensive']:>5.0f}  {r['cape_expensive']:>4.0f}"
              f"  {r['cape_buy_abs']:>3.0f}  {r['cape_sell_abs']:>3.0f}"
              f"  {r['buy_thresh_hi']:>4.0f}  {r['cape_buy_high']:>4.0f}"
              f"  {r['div_window']:>4.0f}  {r['div_price_rise']:>4.1f}  {r['div_breadth_fall']:>4.0f}")
    print(f"{'='*120}")

    best = results[0]
    print(f"\nBest: Sharpe={best['sharpe']:.2f}  CAGR={best['cagr']:.1%}"
          f"  MaxDD={best['max_dd']:.1%}  #Trades={best['n']}  MinHold={best['minh']}d")
    for k in keys:
        print(f"  {k} = {best[k]}")
