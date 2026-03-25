"""
MANU Champions League — Stock-Only Backtest
Buy MANU shares 28 days before CL Round of 16 first leg, hold 30 days.
No options, no bid/ask spread problem. Tests whether the catalyst thesis
itself has edge when execution costs are minimal.

Also tests Europa League knockout dates and includes a benchmark
(buy-and-hold over same periods) for comparison.
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
from datetime import timedelta

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
TICKER = "MANU"
ENTRY_DAYS_BEFORE = 28
HOLD_DAYS = 30

# CL Round of 16 + Europa League knockout dates
EVENTS = {
    # Champions League
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
}

# ─────────────────────────────────────────────
# FETCH DATA
# ─────────────────────────────────────────────
print("Fetching MANU data from Yahoo Finance...")
raw = yf.download(TICKER, start="2012-01-01", end="2024-12-31", auto_adjust=True, progress=False)
raw.columns = raw.columns.get_level_values(0) if isinstance(raw.columns, pd.MultiIndex) else raw.columns
df = raw[["Open", "High", "Low", "Close", "Volume"]].copy()
df.index = pd.to_datetime(df.index)
df = df.dropna()
print(f"  Loaded {len(df)} trading days ({df.index[0].date()} → {df.index[-1].date()})\n")

# ─────────────────────────────────────────────
# BACKTEST
# ─────────────────────────────────────────────
trades = []

for event, match_date in EVENTS.items():
    target_entry = match_date - timedelta(days=ENTRY_DAYS_BEFORE)
    valid_days = df[df.index >= target_entry]
    if valid_days.empty:
        continue

    entry_date = valid_days.index[0]
    entry_price = float(df.loc[entry_date, "Close"])

    # Hold for HOLD_DAYS calendar days
    target_exit = entry_date + timedelta(days=HOLD_DAYS)
    exit_candidates = df[(df.index > entry_date) & (df.index <= target_exit)]
    if exit_candidates.empty:
        continue

    exit_date = exit_candidates.index[-1]
    exit_price = float(df.loc[exit_date, "Close"])
    days_held = (exit_date - entry_date).days

    pnl_pct = (exit_price - entry_price) / entry_price * 100

    # Track max drawdown and max gain during holding period
    hold_window = df[(df.index >= entry_date) & (df.index <= exit_date)]
    prices = hold_window["Close"].values
    max_gain = ((prices.max() - entry_price) / entry_price) * 100
    max_dd = ((prices.min() - entry_price) / entry_price) * 100

    trades.append({
        "Event": event,
        "Match Date": match_date.date(),
        "Entry Date": entry_date.date(),
        "Entry Price": round(entry_price, 2),
        "Exit Date": exit_date.date(),
        "Exit Price": round(exit_price, 2),
        "Days Held": days_held,
        "PnL %": round(pnl_pct, 1),
        "Max Gain %": round(max_gain, 1),
        "Max DD %": round(max_dd, 1),
        "Win": pnl_pct > 0,
    })

results = pd.DataFrame(trades)

# ─────────────────────────────────────────────
# BENCHMARK: random 30-day holds over same period
# ─────────────────────────────────────────────
np.random.seed(42)
benchmark_returns = []
all_dates = df.index.tolist()
for _ in range(1000):
    idx = np.random.randint(0, len(all_dates) - HOLD_DAYS - 1)
    b_entry = all_dates[idx]
    b_exit_candidates = df[(df.index > b_entry) & (df.index <= b_entry + timedelta(days=HOLD_DAYS))]
    if b_exit_candidates.empty:
        continue
    b_exit = b_exit_candidates.index[-1]
    b_ret = (float(df.loc[b_exit, "Close"]) - float(df.loc[b_entry, "Close"])) / float(df.loc[b_entry, "Close"]) * 100
    benchmark_returns.append(b_ret)

bench_avg = np.mean(benchmark_returns)
bench_std = np.std(benchmark_returns)
bench_win = np.mean([r > 0 for r in benchmark_returns]) * 100

# ─────────────────────────────────────────────
# OUTPUT
# ─────────────────────────────────────────────
print("=" * 110)
print("MANU CHAMPIONS LEAGUE / EUROPA LEAGUE — STOCK-ONLY BACKTEST")
print("=" * 110)
print()
print("THESIS:")
print("  Manchester United revenue is tied to CL/EL depth. Deep tournament runs")
print("  generate prize money, match-day revenue, and sponsorship bonuses.")
print("  Test: does simply buying the stock before knockout rounds generate alpha?")
print()
print("STRATEGY:")
print(f"  Buy MANU shares {ENTRY_DAYS_BEFORE} days before first knockout match")
print(f"  Hold for {HOLD_DAYS} calendar days, then sell")
print("  No options — avoids bid/ask spread and model error issues entirely")
print()
print("─" * 110)

# Trade table
print(f"\n  {'Event':<12} {'Match':<12} {'Entry':<12} {'Exit':<12} {'Days':>5} "
      f"{'Entry$':>8} {'Exit$':>8} {'PnL%':>8} {'MaxGain%':>9} {'MaxDD%':>8} {'Result':>7}")
print(f"  {'─'*105}")
for _, row in results.iterrows():
    w = "WIN" if row["Win"] else "LOSS"
    print(f"  {row['Event']:<12} {str(row['Match Date']):<12} {str(row['Entry Date']):<12} "
          f"{str(row['Exit Date']):<12} {row['Days Held']:>5} "
          f"${row['Entry Price']:>7.2f} ${row['Exit Price']:>7.2f} "
          f"{row['PnL %']:>+7.1f}% {row['Max Gain %']:>+8.1f}% {row['Max DD %']:>+7.1f}% {w:>7}")
print()

# Summary
n = len(results)
wins = results["Win"].sum()
avg_pnl = results["PnL %"].mean()
std_pnl = results["PnL %"].std()
sharpe = (avg_pnl / std_pnl) * np.sqrt(n) if std_pnl > 0 else np.nan
median_pnl = results["PnL %"].median()

# Split CL vs EL
cl = results[results["Event"].str.startswith("CL")]
el = results[results["Event"].str.startswith("EL")]

print("─" * 80)
print(f"  {'Metric':<30} {'CL+EL':>15} {'CL Only':>15} {'EL Only':>15}")
print("─" * 80)

for label, subset in [("CL+EL", results), ("CL Only", cl), ("EL Only", el)]:
    pass  # we'll print row by row

def stats(s):
    n = len(s)
    if n == 0:
        return "-", "-", "-", "-", "-", "-"
    w = s["Win"].sum()
    avg = s["PnL %"].mean()
    std = s["PnL %"].std()
    sh = (avg / std) * np.sqrt(n) if std > 0 else np.nan
    med = s["PnL %"].median()
    return n, f"{w/n*100:.0f}%", f"{avg:+.1f}%", f"{std:.1f}%", f"{sh:.2f}", f"{med:+.1f}%"

s_all = stats(results)
s_cl = stats(cl)
s_el = stats(el)

labels = ["Trades", "Win Rate", "Avg PnL", "Std PnL", "Sharpe-like", "Median PnL"]
for i, label in enumerate(labels):
    print(f"  {label:<30} {str(s_all[i]):>15} {str(s_cl[i]):>15} {str(s_el[i]):>15}")

print("─" * 80)
print()

# Benchmark comparison
print("─" * 60)
print("BENCHMARK: Random 30-day holds on MANU (n=1000)")
print("─" * 60)
print(f"  Avg return   : {bench_avg:+.1f}%")
print(f"  Std return   : {bench_std:.1f}%")
print(f"  Win rate     : {bench_win:.0f}%")
print(f"  Median return: {np.median(benchmark_returns):+.1f}%")
print()
print(f"  Strategy edge vs benchmark: {avg_pnl - bench_avg:+.1f}% per trade")
print("─" * 60)
print()

# ─────────────────────────────────────────────
# PLOT
# ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 7))
fig.suptitle("MANU Stock-Only Backtest — Buy Before CL/EL Knockout", fontsize=14, fontweight="bold")

ax.plot(df.index, df["Close"], color="#1f77b4", lw=1.0, label="MANU Close")

for _, row in results.iterrows():
    ed = pd.Timestamp(row["Entry Date"])
    xd = pd.Timestamp(row["Exit Date"])
    color = "#2ca02c" if row["Win"] else "#d62728"
    ax.axvspan(ed, xd, alpha=0.10, color=color)
    ax.scatter(ed, row["Entry Price"], marker="^", color="#2ca02c", s=80, zorder=5)
    ax.scatter(xd, row["Exit Price"], marker="v", color=color, s=80, zorder=5)
    ax.annotate(f"{row['Event']} ({row['PnL %']:+.1f}%)", xy=(ed, row["Entry Price"]),
                xytext=(5, 10), textcoords="offset points", fontsize=7, color=color)

ax.set_ylabel("Price ($)")
ax.set_xlabel("Date")
ax.grid(True, alpha=0.3)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.xaxis.set_major_locator(mdates.YearLocator())

from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker="^", color="w", markerfacecolor="#2ca02c", markersize=9, label="Entry (buy)"),
    Line2D([0], [0], marker="v", color="w", markerfacecolor="#d62728", markersize=9, label="Exit (sell)"),
    ax.get_lines()[0],
]
ax.legend(handles=legend_elements, loc="upper left", fontsize=9)

plt.tight_layout()
plt.savefig("manu_stock_backtest.png", dpi=150, bbox_inches="tight")
print("Plot saved to manu_stock_backtest.png")
plt.close()
