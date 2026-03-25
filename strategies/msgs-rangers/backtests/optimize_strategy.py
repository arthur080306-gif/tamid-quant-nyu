"""
MSGS Strategy Optimizer
- Grid search over ML threshold and hold days
- Reports best Sharpe ratio combination
"""

import pandas as pd
import numpy as np
import yfinance as yf
import math
from datetime import timedelta
import warnings
warnings.filterwarnings("ignore")

ODDS_CSV = "betting_backtest/data/rangers_real_odds.csv"
POSITION_SIZE = 10000

# ── Load & prep data ─────────────────────────────────────────────────────────
df = pd.read_csv(ODDS_CSV)
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

def parse_ml(x):
    try:
        return int(float(str(x).replace("+", "")))
    except:
        return None

df["rangers_ml_num"] = df["rangers_ml"].apply(parse_ml)

# ── Fetch MSGS prices once ───────────────────────────────────────────────────
print("Fetching MSGS prices...")
start = (df["date"].min() - timedelta(days=5)).strftime("%Y-%m-%d")
end   = (df["date"].max() + timedelta(days=30)).strftime("%Y-%m-%d")
ticker = yf.Ticker("MSGS")
prices = ticker.history(start=start, end=end, interval="1d")
prices.index = pd.to_datetime(prices.index).tz_localize(None).normalize()
print(f"Got {len(prices)} trading days\n")

def next_trading_day(date, price_df):
    d = pd.Timestamp(date)
    for _ in range(14):
        if d in price_df.index:
            return d
        d += timedelta(days=1)
    return None

def get_nth_trading_day(start_date, n, price_df):
    d = pd.Timestamp(start_date)
    count = 0
    for _ in range(60):
        d += timedelta(days=1)
        if d in price_df.index:
            count += 1
            if count == n:
                return d
    return None

def run_backtest(ml_threshold, hold_days, short=True):
    signals = df[df["rangers_ml_num"] >= ml_threshold].copy()
    if len(signals) < 5:
        return None

    trades = []
    for _, row in signals.iterrows():
        entry_day = next_trading_day(row["date"] + timedelta(days=1), prices)
        if entry_day is None:
            continue
        exit_day = get_nth_trading_day(entry_day, hold_days, prices)
        if exit_day is None:
            continue

        ep = prices.loc[entry_day, "Open"]
        xp = prices.loc[exit_day, "Close"]

        if short:
            pnl_pct = (ep - xp) / ep
        else:
            pnl_pct = (xp - ep) / ep

        trades.append(pnl_pct)

    if len(trades) < 5:
        return None

    returns = np.array(trades)
    mean_r = np.mean(returns)
    std_r  = np.std(returns, ddof=1)
    if std_r == 0:
        return None

    trades_per_year = 252 / hold_days
    sharpe = (mean_r / std_r) * math.sqrt(trades_per_year)
    roi = np.sum(returns) * 100
    win_rate = np.mean(returns > 0) * 100

    return {
        "ml_threshold": ml_threshold,
        "hold_days":    hold_days,
        "direction":    "short" if short else "long",
        "n_trades":     len(trades),
        "win_rate":     round(win_rate, 1),
        "roi_pct":      round(roi, 2),
        "sharpe":       round(sharpe, 3),
        "mean_ret":     round(mean_r * 100, 3),
        "std_ret":      round(std_r * 100, 3),
    }

# ── Grid search ──────────────────────────────────────────────────────────────
ml_thresholds = [120, 140, 150, 160, 175, 200, 225, 250]
hold_days_list = [1, 2, 3, 4, 5, 7, 10]
directions = [True, False]  # True=short, False=long

results = []
total = len(ml_thresholds) * len(hold_days_list) * len(directions)
done = 0

print(f"Running {total} combinations...")
for ml in ml_thresholds:
    for hd in hold_days_list:
        for short in directions:
            r = run_backtest(ml, hd, short)
            if r:
                results.append(r)
            done += 1
            if done % 20 == 0:
                print(f"  {done}/{total} done...")

print(f"Done. {len(results)} valid results.\n")

results_df = pd.DataFrame(results).sort_values("sharpe", ascending=False)

print("="*75)
print("TOP 15 COMBINATIONS BY SHARPE RATIO")
print("="*75)
print(results_df.head(15).to_string(index=False))
print()

best = results_df.iloc[0]
print("="*75)
print("BEST COMBINATION:")
print(f"  Direction  : {best['direction'].upper()} MSGS")
print(f"  ML Threshold: Rangers >= +{int(best['ml_threshold'])}")
print(f"  Hold Days  : {int(best['hold_days'])} trading days")
print(f"  Trades     : {int(best['n_trades'])}")
print(f"  Win Rate   : {best['win_rate']}%")
print(f"  ROI        : {best['roi_pct']:+.2f}%")
print(f"  Sharpe     : {best['sharpe']:.3f}")
print("="*75)

results_df.to_csv("betting_backtest/data/grid_search_results.csv", index=False)
print("\nFull results saved → betting_backtest/data/grid_search_results.csv")
