"""
MSGS Backtest Engine v2
- Works with real scraped data from rangers_real_odds.csv
- Strategy: when Rangers moneyline moves significantly (sharp money signal),
  short MSGS stock for 3 trading days
- Uses moneyline as proxy for line movement signal
- Fetches real MSGS prices via yfinance
"""

import os
import json
import math
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf

# ── CONFIG ──────────────────────────────────────────────────────────────────
ODDS_CSV    = "betting_backtest/data/rangers_real_odds.csv"
TRADE_LOG   = "betting_backtest/data/trade_log.csv"
METRICS_OUT = "betting_backtest/data/backtest_metrics.json"

# Signal: Rangers are a significant underdog (sharp money moved line)
# Moneyline +200 or higher = Rangers are big underdogs = signal to short MSGS
ML_THRESHOLD    = 150   # Rangers ML >= +150 triggers signal (big underdog)
HOLD_DAYS       = 3     # trading days to hold short
POSITION_SIZE   = 10000 # $ per trade (notional)

os.makedirs("betting_backtest/data", exist_ok=True)

# ── HELPERS ─────────────────────────────────────────────────────────────────

def ml_to_implied_prob(ml_str):
    """Convert American moneyline string to implied probability."""
    try:
        ml = int(str(ml_str).replace("+", ""))
        if ml > 0:
            return 100 / (ml + 100)
        else:
            return abs(ml) / (abs(ml) + 100)
    except:
        return None


def get_msgs_prices(start_date, end_date):
    """Fetch MSGS daily OHLCV from yfinance."""
    try:
        ticker = yf.Ticker("MSGS")
        df = ticker.history(start=start_date, end=end_date, interval="1d")
        if df.empty:
            print("  ⚠️  yfinance returned empty data for MSGS")
            return None
        df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
        return df
    except Exception as e:
        print(f"  ⚠️  yfinance error: {e}")
        return None


def next_trading_day(date, price_df):
    """Return the next date in price_df on or after date."""
    d = pd.Timestamp(date)
    for i in range(14):
        if d in price_df.index:
            return d
        d += timedelta(days=1)
    return None


def get_trading_day_n(start_date, n, price_df):
    """Return the nth trading day after start_date."""
    d = pd.Timestamp(start_date)
    count = 0
    for i in range(60):
        d += timedelta(days=1)
        if d in price_df.index:
            count += 1
            if count == n:
                return d
    return None


# ── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    print("[MSGS Backtest v2]")
    print(f"Loading odds: {ODDS_CSV}")

    df = pd.read_csv(ODDS_CSV)
    print(f"  Loaded {len(df)} games")
    print(f"  Columns: {list(df.columns)}")

    # Parse dates
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    print(f"  Games with valid dates: {len(df)}")

    # Parse moneyline
    def parse_ml(x):
        try:
            return int(float(str(x).replace("+", "")))
        except:
            return None

    df["rangers_ml_num"] = df["rangers_ml"].apply(parse_ml)

    # Signal: Rangers are big underdogs (ML >= threshold)
    signals = df[df["rangers_ml_num"] >= ML_THRESHOLD].copy()
    print(f"\nSignal: Rangers ML >= +{ML_THRESHOLD}")
    print(f"  Signal games found: {len(signals)}")

    if signals.empty:
        print("No signals found. Try lowering ML_THRESHOLD.")
        return

    # Fetch MSGS prices
    start = (df["date"].min() - timedelta(days=5)).strftime("%Y-%m-%d")
    end   = (df["date"].max() + timedelta(days=30)).strftime("%Y-%m-%d")
    print(f"\nFetching MSGS prices {start} → {end} ...")
    prices = get_msgs_prices(start, end)

    if prices is None:
        print("Could not fetch MSGS prices.")
        return

    print(f"  Got {len(prices)} trading days of MSGS data")
    print(f"  Price range: ${prices['Close'].min():.2f} – ${prices['Close'].max():.2f}")

    # Build trade log
    trades = []
    skipped = 0

    for _, row in signals.iterrows():
        game_date = row["date"]

        # Enter short at open of next trading day after game
        entry_day = next_trading_day(game_date + timedelta(days=1), prices)
        if entry_day is None:
            skipped += 1
            continue

        # Exit at close of 3rd trading day after entry
        exit_day = get_trading_day_n(entry_day, HOLD_DAYS, prices)
        if exit_day is None:
            skipped += 1
            continue

        entry_price = prices.loc[entry_day, "Open"]
        exit_price  = prices.loc[exit_day, "Close"]

        # Short PnL: profit when price falls
        pnl_pct = (entry_price - exit_price) / entry_price
        pnl_dollar = pnl_pct * POSITION_SIZE

        trades.append({
            "game_date":    game_date.strftime("%Y-%m-%d"),
            "season":       row["season"],
            "opponent":     row["away_team"] if row["rangers_home"] else row["home_team"],
            "rangers_home": row["rangers_home"],
            "rangers_ml":   row["rangers_ml"],
            "entry_date":   entry_day.strftime("%Y-%m-%d"),
            "exit_date":    exit_day.strftime("%Y-%m-%d"),
            "entry_price":  round(entry_price, 2),
            "exit_price":   round(exit_price, 2),
            "pnl_pct":      round(pnl_pct * 100, 3),
            "pnl_dollar":   round(pnl_dollar, 2),
            "win":          pnl_pct > 0,
        })

    print(f"\nTrades executed: {len(trades)}  (skipped: {skipped})")

    if not trades:
        print("No trades could be executed.")
        return

    trade_df = pd.DataFrame(trades)
    trade_df.to_csv(TRADE_LOG, index=False)
    print(f"Trade log saved → {TRADE_LOG}")

    # ── METRICS ─────────────────────────────────────────────────────────────
    returns = trade_df["pnl_pct"].values / 100  # as decimals

    total_pnl     = trade_df["pnl_dollar"].sum()
    total_invested = POSITION_SIZE * len(trades)
    roi_pct        = (total_pnl / total_invested) * 100
    win_rate       = trade_df["win"].mean() * 100
    avg_win        = trade_df.loc[trade_df["win"], "pnl_dollar"].mean() if trade_df["win"].any() else 0
    avg_loss       = trade_df.loc[~trade_df["win"], "pnl_dollar"].mean() if (~trade_df["win"]).any() else 0

    # Sharpe (annualised, assuming ~252 trading days, 3-day hold)
    trades_per_year = 252 / HOLD_DAYS
    mean_r = np.mean(returns)
    std_r  = np.std(returns, ddof=1) if len(returns) > 1 else 1e-9
    sharpe = (mean_r / std_r) * math.sqrt(trades_per_year) if std_r > 0 else 0

    # Max drawdown on cumulative equity curve
    equity = np.cumsum(trade_df["pnl_dollar"].values)
    peak   = np.maximum.accumulate(equity)
    dd     = equity - peak
    max_dd = dd.min()

    # Per-season breakdown
    season_stats = {}
    for s, grp in trade_df.groupby("season"):
        season_stats[s] = {
            "trades":   len(grp),
            "win_rate": round(grp["win"].mean() * 100, 1),
            "total_pnl": round(grp["pnl_dollar"].sum(), 2),
            "roi_pct":  round((grp["pnl_dollar"].sum() / (POSITION_SIZE * len(grp))) * 100, 2),
        }

    metrics = {
        "strategy":         f"Short MSGS when Rangers ML >= +{ML_THRESHOLD}, hold {HOLD_DAYS} days",
        "ml_threshold":     ML_THRESHOLD,
        "hold_days":        HOLD_DAYS,
        "position_size":    POSITION_SIZE,
        "total_trades":     len(trades),
        "skipped_trades":   skipped,
        "win_rate_pct":     round(win_rate, 1),
        "total_pnl_usd":    round(total_pnl, 2),
        "total_invested":   total_invested,
        "roi_pct":          round(roi_pct, 2),
        "annualised_sharpe":round(sharpe, 2),
        "max_drawdown_usd": round(max_dd, 2),
        "avg_win_usd":      round(avg_win, 2),
        "avg_loss_usd":     round(avg_loss, 2),
        "per_season":       season_stats,
    }

    with open(METRICS_OUT, "w") as f:
        json.dump(metrics, f, indent=2)

    # ── PRINT RESULTS ────────────────────────────────────────────────────────
    print("\n" + "="*55)
    print("  BACKTEST RESULTS")
    print("="*55)
    print(f"  Strategy   : Short MSGS when Rangers ML >= +{ML_THRESHOLD}")
    print(f"  Hold       : {HOLD_DAYS} trading days")
    print(f"  Position   : ${POSITION_SIZE:,} per trade")
    print(f"  Trades     : {len(trades)}")
    print(f"  Win Rate   : {win_rate:.1f}%")
    print(f"  Total PnL  : ${total_pnl:+,.2f}")
    print(f"  ROI        : {roi_pct:+.2f}%")
    print(f"  Ann. Sharpe: {sharpe:.2f}")
    print(f"  Max Drawdown: ${max_dd:,.2f}")
    print(f"  Avg Win    : ${avg_win:,.2f}")
    print(f"  Avg Loss   : ${avg_loss:,.2f}")
    print("-"*55)
    print("  Per Season:")
    for s, st in season_stats.items():
        print(f"    {s}: {st['trades']} trades, {st['win_rate']}% wins, "
              f"ROI {st['roi_pct']:+.1f}%, PnL ${st['total_pnl']:+,.2f}")
    print("="*55)
    print(f"\n  Metrics saved → {METRICS_OUT}")
    print(f"  Trade log  → {TRADE_LOG}")

    # Sample trades
    print("\n  Sample trades:")
    print(trade_df[["game_date","opponent","rangers_ml","entry_price",
                     "exit_price","pnl_pct","pnl_dollar"]].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
