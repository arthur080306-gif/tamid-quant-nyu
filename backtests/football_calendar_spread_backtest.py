"""
Football Club Calendar Spread Backtest — Multi-Ticker
Applies the MSGS playoff vol strategy to publicly traded football clubs around
Champions League and Europa League knockout rounds.

Tickers: MANU (Man United), BVB.DE (Borussia Dortmund), JUVE.MI (Juventus)
Strategy: Sell 30-day ATM call, buy 90-day ATM call, 28 days before first knockout match.
Exit at 20% profit target or short leg expiry.
"""

import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from arch import arch_model
from datetime import timedelta
from scipy.stats import norm

# ─────────────────────────────────────────────
# BLACK-SCHOLES
# ─────────────────────────────────────────────
def bs_call(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0:
        return max(S - K, 0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
ENTRY_DAYS_BEFORE = 28
RISK_FREE_RATE = 0.04
SHORT_EXPIRY_DAYS = 30
LONG_EXPIRY_DAYS = 90
SPREAD_PROFIT_TARGET = 0.20

# All knockout dates by ticker
EVENTS = {
    "MANU": {
        # Champions League R16
        "CL 2012-13": pd.Timestamp("2013-02-13"),
        "CL 2013-14": pd.Timestamp("2014-02-25"),
        "CL 2017-18": pd.Timestamp("2018-02-21"),
        "CL 2018-19": pd.Timestamp("2019-02-12"),
        "CL 2021-22": pd.Timestamp("2022-02-23"),
        # Europa League knockout
        "EL 2016-17": pd.Timestamp("2017-02-16"),
        "EL 2019-20": pd.Timestamp("2020-02-20"),
        "EL 2020-21": pd.Timestamp("2021-02-18"),
        "EL 2022-23": pd.Timestamp("2023-02-16"),
    },
    "BVB.DE": {
        "CL 2012-13": pd.Timestamp("2013-02-20"),
        "CL 2013-14": pd.Timestamp("2014-02-25"),
        "CL 2014-15": pd.Timestamp("2015-02-24"),
        "CL 2016-17": pd.Timestamp("2017-02-14"),
        "CL 2018-19": pd.Timestamp("2019-02-13"),
        "CL 2019-20": pd.Timestamp("2020-02-18"),
        "CL 2020-21": pd.Timestamp("2021-02-17"),
        "CL 2022-23": pd.Timestamp("2023-02-15"),
        "CL 2023-24": pd.Timestamp("2024-02-20"),
    },
    "JUVE.MI": {
        "CL 2012-13": pd.Timestamp("2013-02-12"),
        "CL 2014-15": pd.Timestamp("2015-02-24"),
        "CL 2015-16": pd.Timestamp("2016-02-23"),
        "CL 2016-17": pd.Timestamp("2017-02-22"),
        "CL 2017-18": pd.Timestamp("2018-02-13"),
        "CL 2018-19": pd.Timestamp("2019-02-20"),
        "CL 2019-20": pd.Timestamp("2020-02-26"),
        "CL 2020-21": pd.Timestamp("2021-02-17"),
        "CL 2021-22": pd.Timestamp("2022-02-22"),
    },
}

# ─────────────────────────────────────────────
# PRINT STRATEGY DESCRIPTION
# ─────────────────────────────────────────────
print("=" * 120)
print("FOOTBALL CLUB CALENDAR SPREAD — MULTI-TICKER BACKTEST")
print("=" * 120)
print()
print("THESIS:")
print("  Publicly traded football clubs derive significant revenue from deep Champions")
print("  League / Europa League runs (prize money, match-day revenue, sponsorship bonuses).")
print("  These stocks tend to be low-volatility with predictable catalyst timing around")
print("  knockout stage matches.")
print()
print("STRATEGY:")
print(f"  Structure : Sell {SHORT_EXPIRY_DAYS}-day ATM call, buy {LONG_EXPIRY_DAYS}-day ATM call (calendar spread)")
print(f"  Entry     : Fixed calendar, {ENTRY_DAYS_BEFORE} days before first knockout match")
print(f"  Exit      : Close at {int(SPREAD_PROFIT_TARGET*100)}% profit target, or hold to short leg expiry ({SHORT_EXPIRY_DAYS} days)")
print("  Tickers   : MANU (CL+EL), BVB.DE (CL), JUVE.MI (CL)")
print()
print("─" * 120)
print()

# ─────────────────────────────────────────────
# RUN BACKTEST FOR EACH TICKER
# ─────────────────────────────────────────────
all_trades = []

for ticker, events in EVENTS.items():
    print(f"{'─'*60}")
    print(f"  {ticker}")
    print(f"{'─'*60}")

    # Fetch data
    print(f"  Fetching {ticker} data...")
    raw = yf.download(ticker, start="2011-01-01", end="2024-12-31", auto_adjust=True, progress=False)
    raw.columns = raw.columns.get_level_values(0) if isinstance(raw.columns, pd.MultiIndex) else raw.columns
    df = raw[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.index = pd.to_datetime(df.index)
    df = df.dropna()

    if len(df) < 252:
        print(f"  WARNING: Only {len(df)} trading days, skipping {ticker}")
        continue

    print(f"  Loaded {len(df)} days ({df.index[0].date()} → {df.index[-1].date()})")

    # Returns & GARCH
    df["ret"] = df["Close"].pct_change()
    returns_pct = df["ret"].dropna() * 100
    gjr = arch_model(returns_pct, vol="GARCH", p=1, o=1, q=1, dist="Normal")
    res = gjr.fit(disp="off")
    cond_vol = res.conditional_volatility / 100 * np.sqrt(252)
    df["garch_vol"] = np.nan
    df.loc[returns_pct.index, "garch_vol"] = cond_vol.values
    print(f"  GARCH fit: AIC={res.aic:.1f}")

    # Backtest each event
    for season, match_date in events.items():
        target_entry = match_date - timedelta(days=ENTRY_DAYS_BEFORE)
        valid_days = df[df.index >= target_entry]
        if valid_days.empty or valid_days.index[0] > match_date:
            print(f"    {season}: No data around entry date")
            continue

        entry_date = valid_days.index[0]
        entry_price = float(df.loc[entry_date, "Close"])
        entry_vol = float(df.loc[entry_date, "garch_vol"])

        if np.isnan(entry_vol) or entry_vol <= 0:
            print(f"    {season}: Invalid vol at entry, skipping")
            continue

        strike = entry_price
        T_short_entry = SHORT_EXPIRY_DAYS / 365
        T_long_entry = LONG_EXPIRY_DAYS / 365

        short_call_entry = bs_call(entry_price, strike, T_short_entry, RISK_FREE_RATE, entry_vol)
        long_call_entry = bs_call(entry_price, strike, T_long_entry, RISK_FREE_RATE, entry_vol)
        net_debit = long_call_entry - short_call_entry

        if net_debit <= 0:
            print(f"    {season}: Non-positive net debit, skipping")
            continue

        # Check daily for profit target
        exit_target = entry_date + timedelta(days=SHORT_EXPIRY_DAYS)
        hold_window = df[(df.index > entry_date) & (df.index <= exit_target)]

        if hold_window.empty:
            continue

        exit_date = None
        exit_reason = "Short Expiry"

        for dt, row in hold_window.iterrows():
            spot = float(row["Close"])
            vol = float(row["garch_vol"])
            if np.isnan(vol) or vol <= 0:
                continue
            days_elapsed = (dt - entry_date).days
            T_short_now = max((SHORT_EXPIRY_DAYS - days_elapsed) / 365, 0)
            T_long_now = max((LONG_EXPIRY_DAYS - days_elapsed) / 365, 0)

            short_now = bs_call(spot, strike, T_short_now, RISK_FREE_RATE, vol)
            long_now = bs_call(spot, strike, T_long_now, RISK_FREE_RATE, vol)
            spread_value = long_now - short_now
            current_pnl_pct = (spread_value - net_debit) / net_debit

            if current_pnl_pct >= SPREAD_PROFIT_TARGET:
                exit_date = dt
                exit_reason = f"Profit Target ({current_pnl_pct*100:.0f}%)"
                break

        if exit_date is None:
            exit_date = hold_window.index[-1]

        exit_price = float(df.loc[exit_date, "Close"])
        exit_vol = float(df.loc[exit_date, "garch_vol"])
        days_held = (exit_date - entry_date).days

        T_short_exit = max((SHORT_EXPIRY_DAYS - days_held) / 365, 0)
        T_long_exit = max((LONG_EXPIRY_DAYS - days_held) / 365, 0)

        short_call_exit = bs_call(exit_price, strike, T_short_exit, RISK_FREE_RATE, exit_vol)
        long_call_exit = bs_call(exit_price, strike, T_long_exit, RISK_FREE_RATE, exit_vol)

        spread_value_exit = long_call_exit - short_call_exit
        pnl = spread_value_exit - net_debit
        pnl_pct = (pnl / net_debit) * 100 if net_debit > 0 else 0

        all_trades.append({
            "Ticker": ticker,
            "Season": season,
            "Entry Date": entry_date.date(),
            "Entry Price": round(entry_price, 2),
            "Entry Vol": round(entry_vol * 100, 1),
            "Net Debit": round(net_debit, 2),
            "Exit Date": exit_date.date(),
            "Exit Reason": exit_reason,
            "Exit Price": round(exit_price, 2),
            "Holding Days": days_held,
            "Spread PnL": round(pnl, 2),
            "PnL %": round(pnl_pct, 1),
            "Win": pnl > 0,
        })

    print()

# ─────────────────────────────────────────────
# RESULTS
# ─────────────────────────────────────────────
results = pd.DataFrame(all_trades)

print("=" * 130)
print("ALL TRADES")
print("=" * 130)
display_cols = ["Ticker", "Season", "Entry Date", "Entry Price", "Entry Vol",
                "Net Debit", "Exit Date", "Exit Reason", "Holding Days",
                "Spread PnL", "PnL %", "Win"]
print(results[display_cols].to_string(index=False))
print()

# ─────────────────────────────────────────────
# PER-TICKER STATS
# ─────────────────────────────────────────────
print("=" * 130)
print("PER-TICKER STATISTICS")
print("=" * 130)
for ticker in EVENTS.keys():
    t = results[results["Ticker"] == ticker]
    if t.empty:
        continue
    n = len(t)
    wins = t["Win"].sum()
    avg_pnl = t["PnL %"].mean()
    std_pnl = t["PnL %"].std()
    sharpe = (avg_pnl / std_pnl) * np.sqrt(n) if std_pnl > 0 else np.nan
    print(f"\n  {ticker} ({n} trades)")
    print(f"    Win Rate    : {wins/n*100:.0f}% ({int(wins)}/{n})")
    print(f"    Avg PnL %   : {avg_pnl:.1f}%")
    print(f"    Std PnL %   : {std_pnl:.1f}%")
    print(f"    Sharpe-like : {sharpe:.2f}")
    print(f"    Best Trade  : {t['PnL %'].max():.1f}%")
    print(f"    Worst Trade : {t['PnL %'].min():.1f}%")

# ─────────────────────────────────────────────
# COMBINED STATS
# ─────────────────────────────────────────────
n_total = len(results)
wins_total = results["Win"].sum()
avg_pnl_total = results["PnL %"].mean()
std_pnl_total = results["PnL %"].std()
sharpe_total = (avg_pnl_total / std_pnl_total) * np.sqrt(n_total) if std_pnl_total > 0 else np.nan

print()
print("=" * 130)
print("COMBINED STATISTICS (ALL TICKERS)")
print("=" * 130)
print(f"  Total Trades      : {n_total}")
print(f"  Wins              : {int(wins_total)}")
print(f"  Win Rate          : {wins_total/n_total*100:.1f}%")
print(f"  Avg PnL %         : {avg_pnl_total:.1f}%")
print(f"  Std PnL %         : {std_pnl_total:.1f}%")
print(f"  Sharpe-like Ratio : {sharpe_total:.2f}")
print(f"  Best Trade        : {results['PnL %'].max():.1f}%")
print(f"  Worst Trade       : {results['PnL %'].min():.1f}%")
print(f"  Median PnL %      : {results['PnL %'].median():.1f}%")
print(f"  % Hitting Target  : {(results['Exit Reason'].str.contains('Profit')).sum()/n_total*100:.0f}%")
print("=" * 130)
print()

# ─────────────────────────────────────────────
# PLOT
# ─────────────────────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(14, 12))
fig.suptitle("Football Club Calendar Spread — Multi-Ticker Backtest", fontsize=14, fontweight="bold")

colors_ticker = {"MANU": "#DA291C", "BVB.DE": "#FDE100", "JUVE.MI": "#000000"}

for idx, (ticker, events) in enumerate(EVENTS.items()):
    ax = axes[idx]
    raw = yf.download(ticker, start="2011-01-01", end="2024-12-31", auto_adjust=True, progress=False)
    raw.columns = raw.columns.get_level_values(0) if isinstance(raw.columns, pd.MultiIndex) else raw.columns
    price = raw["Close"].dropna()
    ax.plot(price.index, price.values, color="#1f77b4", lw=1.0)

    ticker_trades = results[results["Ticker"] == ticker]
    for _, row in ticker_trades.iterrows():
        ed = pd.Timestamp(row["Entry Date"])
        xd = pd.Timestamp(row["Exit Date"])
        color = "#2ca02c" if row["Win"] else "#d62728"
        ax.axvspan(ed, xd, alpha=0.08, color=color)
        ax.scatter(ed, row["Entry Price"], marker="^", color="#2ca02c", s=60, zorder=5)
        ax.scatter(xd, row["Exit Price"], marker="v", color=color, s=60, zorder=5)

    t = ticker_trades
    n = len(t)
    wr = t["Win"].sum() / n * 100 if n > 0 else 0
    avg = t["PnL %"].mean() if n > 0 else 0
    ax.set_ylabel(f"{ticker}\nPrice")
    ax.set_title(f"{ticker} — {n} trades, {wr:.0f}% win rate, {avg:.1f}% avg PnL", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator(2))

plt.tight_layout()
plt.savefig("football_calendar_spread_backtest.png", dpi=150, bbox_inches="tight")
print("Plot saved to football_calendar_spread_backtest.png")
plt.close()
