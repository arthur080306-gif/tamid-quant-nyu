"""
MSGS Final Backtest Report
Best combination from grid search:
  LONG MSGS when Rangers ML >= +160, hold 2 trading days
"""

import pandas as pd
import numpy as np
import yfinance as yf
import math, json, os
from datetime import timedelta
import warnings
warnings.filterwarnings("ignore")

ODDS_CSV      = "betting_backtest/data/rangers_real_odds.csv"
TRADE_LOG     = "betting_backtest/data/final_trade_log.csv"
METRICS_OUT   = "betting_backtest/data/final_metrics.json"
POSITION_SIZE = 10000
ML_THRESHOLD  = 160
HOLD_DAYS     = 2
DIRECTION     = "long"

os.makedirs("betting_backtest/data", exist_ok=True)

# ── Load data ────────────────────────────────────────────────────────────────
df = pd.read_csv(ODDS_CSV)
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

def parse_ml(x):
    try:
        return int(float(str(x).replace("+", "")))
    except:
        return None

df["rangers_ml_num"] = df["rangers_ml"].apply(parse_ml)
signals = df[df["rangers_ml_num"] >= ML_THRESHOLD].copy()

# ── Fetch MSGS prices ────────────────────────────────────────────────────────
print("Fetching MSGS prices...")
start = (df["date"].min() - timedelta(days=5)).strftime("%Y-%m-%d")
end   = (df["date"].max() + timedelta(days=30)).strftime("%Y-%m-%d")
prices = yf.Ticker("MSGS").history(start=start, end=end, interval="1d")
prices.index = pd.to_datetime(prices.index).tz_localize(None).normalize()
print(f"Got {len(prices)} trading days of MSGS data\n")

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

# ── Build trades ─────────────────────────────────────────────────────────────
trades = []
for _, row in signals.iterrows():
    entry_day = next_trading_day(row["date"] + timedelta(days=1), prices)
    if entry_day is None:
        continue
    exit_day = get_nth_trading_day(entry_day, HOLD_DAYS, prices)
    if exit_day is None:
        continue

    ep = prices.loc[entry_day, "Open"]
    xp = prices.loc[exit_day, "Close"]
    pnl_pct    = (xp - ep) / ep        # long
    pnl_dollar = pnl_pct * POSITION_SIZE

    opponent = row["away_team"] if row["rangers_home"] else row["home_team"]

    trades.append({
        "game_date":    row["date"].strftime("%Y-%m-%d"),
        "season":       row["season"],
        "opponent":     opponent,
        "rangers_home": row["rangers_home"],
        "rangers_ml":   int(row["rangers_ml_num"]),
        "entry_date":   entry_day.strftime("%Y-%m-%d"),
        "exit_date":    exit_day.strftime("%Y-%m-%d"),
        "entry_price":  round(ep, 2),
        "exit_price":   round(xp, 2),
        "pnl_pct":      round(pnl_pct * 100, 3),
        "pnl_dollar":   round(pnl_dollar, 2),
        "win":          pnl_pct > 0,
        "cumulative_pnl": 0,  # fill below
    })

trade_df = pd.DataFrame(trades)
trade_df["cumulative_pnl"] = trade_df["pnl_dollar"].cumsum().round(2)
trade_df.to_csv(TRADE_LOG, index=False)

# ── Metrics ──────────────────────────────────────────────────────────────────
returns = trade_df["pnl_pct"].values / 100

total_pnl      = trade_df["pnl_dollar"].sum()
total_invested = POSITION_SIZE * len(trade_df)
roi_pct        = (total_pnl / total_invested) * 100
win_rate       = trade_df["win"].mean() * 100
avg_win        = trade_df.loc[trade_df["win"],  "pnl_dollar"].mean()
avg_loss       = trade_df.loc[~trade_df["win"], "pnl_dollar"].mean()
profit_factor  = abs(trade_df.loc[trade_df["win"],  "pnl_dollar"].sum() /
                     trade_df.loc[~trade_df["win"], "pnl_dollar"].sum()) if (~trade_df["win"]).any() else float("inf")

trades_per_year = 252 / HOLD_DAYS
mean_r = np.mean(returns)
std_r  = np.std(returns, ddof=1)
sharpe = (mean_r / std_r) * math.sqrt(trades_per_year)

equity = trade_df["pnl_dollar"].cumsum().values
peak   = np.maximum.accumulate(equity)
max_dd = (equity - peak).min()
max_dd_pct = (max_dd / POSITION_SIZE) * 100

# Per-season
season_stats = {}
for s, grp in trade_df.groupby("season"):
    wr   = grp["win"].mean() * 100
    pnl  = grp["pnl_dollar"].sum()
    roi  = (pnl / (POSITION_SIZE * len(grp))) * 100
    r    = grp["pnl_pct"].values / 100
    sh   = (np.mean(r) / np.std(r, ddof=1)) * math.sqrt(trades_per_year) if np.std(r, ddof=1) > 0 else 0
    season_stats[s] = {
        "trades":    len(grp),
        "win_rate":  round(wr, 1),
        "roi_pct":   round(roi, 2),
        "total_pnl": round(pnl, 2),
        "sharpe":    round(sh, 2),
    }

metrics = {
    "strategy":          f"LONG MSGS when Rangers ML >= +{ML_THRESHOLD}, hold {HOLD_DAYS} trading days",
    "ml_threshold":      ML_THRESHOLD,
    "hold_days":         HOLD_DAYS,
    "direction":         DIRECTION,
    "position_size_usd": POSITION_SIZE,
    "total_trades":      len(trade_df),
    "win_rate_pct":      round(win_rate, 1),
    "total_pnl_usd":     round(total_pnl, 2),
    "roi_pct":           round(roi_pct, 2),
    "annualised_sharpe": round(sharpe, 2),
    "profit_factor":     round(profit_factor, 2),
    "max_drawdown_usd":  round(max_dd, 2),
    "max_drawdown_pct":  round(max_dd_pct, 2),
    "avg_win_usd":       round(avg_win, 2),
    "avg_loss_usd":      round(avg_loss, 2),
    "per_season":        season_stats,
}

with open(METRICS_OUT, "w") as f:
    json.dump(metrics, f, indent=2)

# ── Print report ─────────────────────────────────────────────────────────────
print("=" * 60)
print("  MSGS CROSS-MARKET BACKTEST — FINAL REPORT")
print("=" * 60)
print(f"  Signal     : Rangers moneyline >= +{ML_THRESHOLD}")
print(f"  Action     : BUY MSGS at next-day open")
print(f"  Exit       : Sell at close of day +{HOLD_DAYS}")
print(f"  Position   : ${POSITION_SIZE:,} per trade")
print(f"  Period     : {trade_df['game_date'].min()} to {trade_df['game_date'].max()}")
print("-" * 60)
print(f"  Total Trades     : {len(trade_df)}")
print(f"  Win Rate         : {win_rate:.1f}%")
print(f"  Total PnL        : ${total_pnl:+,.2f}")
print(f"  ROI (cumulative) : {roi_pct:+.2f}%")
print(f"  Ann. Sharpe      : {sharpe:.2f}")
print(f"  Profit Factor    : {profit_factor:.2f}")
print(f"  Max Drawdown     : ${max_dd:,.2f} ({max_dd_pct:.1f}%)")
print(f"  Avg Win          : ${avg_win:+,.2f}")
print(f"  Avg Loss         : ${avg_loss:+,.2f}")
print("-" * 60)
print("  Per Season Breakdown:")
for s, st in sorted(season_stats.items()):
    print(f"    {s}: {st['trades']:2d} trades | {st['win_rate']:5.1f}% wins | "
          f"ROI {st['roi_pct']:+6.2f}% | PnL ${st['total_pnl']:+,.2f} | Sharpe {st['sharpe']:.2f}")
print("=" * 60)
print(f"\n  Files saved:")
print(f"    {TRADE_LOG}")
print(f"    {METRICS_OUT}")

# ── Equity curve (ASCII) ──────────────────────────────────────────────────────
print("\n  Equity Curve (cumulative PnL):")
eq = trade_df["cumulative_pnl"].values
mn, mx = eq.min(), eq.max()
width = 50
print(f"  ${mx:+,.0f} ┐")
for i in range(0, len(eq), max(1, len(eq)//20)):
    bar_len = int((eq[i] - mn) / (mx - mn) * width) if mx != mn else 25
    bar = "█" * bar_len
    print(f"           │{bar}")
print(f"  ${mn:+,.0f} └{'─'*width}")
print(f"           {'start':^10}{'→':^{width-10}}{'end':^10}")

# ── Full trade table ──────────────────────────────────────────────────────────
print("\n  Full trade log:")
print(trade_df[["game_date","season","opponent","rangers_ml",
                "entry_price","exit_price","pnl_pct","pnl_dollar","cumulative_pnl"]].to_string(index=False))
