"""
Conviction-Weighted MANU Catalyst — Stock Backtest (Expanded)
Trades MANU around EVERY European competition stage from 2012-2024:
CL/EL group stages, R32, R16, QF, SF, and finals.

Position sizing based on conviction signals: event importance,
pre-event vol regime, and momentum.
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
ENTRY_DAYS_BEFORE = 28
HOLD_DAYS = 30

# Every European competition stage MANU played (2012-2024)
# Format: "Season Stage": first match date of that stage
EVENTS = {
    # 2012-13 Champions League
    "12-13 CL Group":  pd.Timestamp("2012-09-19"),
    "12-13 CL R16":    pd.Timestamp("2013-02-13"),
    # 2013-14 Champions League
    "13-14 CL Group":  pd.Timestamp("2013-09-17"),
    "13-14 CL R16":    pd.Timestamp("2014-02-25"),
    "13-14 CL QF":     pd.Timestamp("2014-04-01"),
    # 2015-16 CL → EL
    "15-16 CL Group":  pd.Timestamp("2015-09-15"),
    "15-16 EL R32":    pd.Timestamp("2016-02-18"),
    "15-16 EL R16":    pd.Timestamp("2016-03-10"),
    # 2016-17 Europa League (winners)
    "16-17 EL Group":  pd.Timestamp("2016-09-15"),
    "16-17 EL R32":    pd.Timestamp("2017-02-16"),
    "16-17 EL R16":    pd.Timestamp("2017-03-09"),
    "16-17 EL QF":     pd.Timestamp("2017-04-13"),
    "16-17 EL SF":     pd.Timestamp("2017-05-04"),
    "16-17 EL Final":  pd.Timestamp("2017-05-24"),
    # 2017-18 Champions League
    "17-18 CL Group":  pd.Timestamp("2017-09-12"),
    "17-18 CL R16":    pd.Timestamp("2018-02-21"),
    # 2018-19 Champions League
    "18-19 CL Group":  pd.Timestamp("2018-09-19"),
    "18-19 CL R16":    pd.Timestamp("2019-02-12"),
    "18-19 CL QF":     pd.Timestamp("2019-04-10"),
    # 2019-20 Europa League
    "19-20 EL Group":  pd.Timestamp("2019-09-19"),
    "19-20 EL R32":    pd.Timestamp("2020-02-20"),
    "19-20 EL R16":    pd.Timestamp("2020-03-12"),
    "19-20 EL QF":     pd.Timestamp("2020-08-10"),  # COVID delay
    "19-20 EL SF":     pd.Timestamp("2020-08-16"),  # COVID delay
    # 2020-21 CL → EL (finalists)
    "20-21 CL Group":  pd.Timestamp("2020-10-20"),
    "20-21 EL R32":    pd.Timestamp("2021-02-18"),
    "20-21 EL R16":    pd.Timestamp("2021-03-11"),
    "20-21 EL QF":     pd.Timestamp("2021-04-08"),
    "20-21 EL SF":     pd.Timestamp("2021-04-29"),
    "20-21 EL Final":  pd.Timestamp("2021-05-26"),
    # 2021-22 Champions League
    "21-22 CL Group":  pd.Timestamp("2021-09-14"),
    "21-22 CL R16":    pd.Timestamp("2022-02-23"),
    # 2022-23 Europa League
    "22-23 EL Group":  pd.Timestamp("2022-09-08"),
    "22-23 EL R32":    pd.Timestamp("2023-02-16"),
    "22-23 EL R16":    pd.Timestamp("2023-03-09"),
    "22-23 EL QF":     pd.Timestamp("2023-04-13"),
    # 2023-24 Champions League (group stage exit)
    "23-24 CL Group":  pd.Timestamp("2023-09-20"),
}

# ─────────────────────────────────────────────
# FETCH DATA
# ─────────────────────────────────────────────
print("Fetching MANU data from Yahoo Finance...")
raw = yf.download("MANU", start="2012-01-01", end="2024-12-31", auto_adjust=True, progress=False)
raw.columns = raw.columns.get_level_values(0) if isinstance(raw.columns, pd.MultiIndex) else raw.columns
df = raw[["Close", "Volume"]].dropna()
df.index = pd.to_datetime(df.index)
df["ret"] = df["Close"].pct_change()
df["vol_20d"] = df["ret"].rolling(20).std() * np.sqrt(252) * 100
df["vol_60d"] = df["ret"].rolling(60).std() * np.sqrt(252) * 100
df["mom_20d"] = df["Close"].pct_change(20) * 100
print(f"  Loaded {len(df)} trading days ({df.index[0].date()} → {df.index[-1].date()})\n")

# ─────────────────────────────────────────────
# CONVICTION SCORING
# ─────────────────────────────────────────────
def score_conviction(event, entry_date, df):
    """
    Score 0-100:
      Event Importance (0-40): knockout > group, later rounds > earlier
      Vol Regime      (0-30): compressed vol = more room to expand
      Momentum        (0-30): flat/slight positive = catalyst not priced in
    """
    scores = {}

    # 1. EVENT IMPORTANCE (0-40)
    event_lower = event.lower()
    if "final" in event_lower:
        scores["event"] = 40
    elif "sf" in event_lower:
        scores["event"] = 35
    elif "qf" in event_lower:
        scores["event"] = 30
    elif "r16" in event_lower:
        scores["event"] = 25
    elif "r32" in event_lower:
        scores["event"] = 20
    elif "group" in event_lower:
        scores["event"] = 10
    else:
        scores["event"] = 10

    # Bonus for EL (historically higher win rate for MANU stock)
    if "el" in event_lower:
        scores["event"] = min(40, scores["event"] + 5)

    # 2. VOL REGIME (0-30)
    vol_20 = df.loc[entry_date, "vol_20d"] if not np.isnan(df.loc[entry_date, "vol_20d"]) else 30
    vol_60 = df.loc[entry_date, "vol_60d"] if not np.isnan(df.loc[entry_date, "vol_60d"]) else 30
    vol_ratio = vol_20 / vol_60 if vol_60 > 0 else 1.0

    if vol_ratio < 0.8:
        scores["vol"] = 30
    elif vol_ratio < 1.0:
        scores["vol"] = 20
    elif vol_ratio < 1.2:
        scores["vol"] = 10
    else:
        scores["vol"] = 0

    # 3. MOMENTUM (0-30)
    mom = df.loc[entry_date, "mom_20d"] if not np.isnan(df.loc[entry_date, "mom_20d"]) else 0
    if -2 <= mom <= 5:
        scores["momentum"] = 30
    elif -5 <= mom < -2 or 5 < mom <= 10:
        scores["momentum"] = 15
    else:
        scores["momentum"] = 0

    return sum(scores.values()), scores

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

    # Skip if too close to IPO (need 60-day vol)
    if entry_date < pd.Timestamp("2012-11-01"):
        continue

    target_exit = entry_date + timedelta(days=HOLD_DAYS)
    hold_window = df[(df.index > entry_date) & (df.index <= target_exit)]
    if hold_window.empty:
        continue

    exit_date = hold_window.index[-1]
    exit_price = float(df.loc[exit_date, "Close"])
    days_held = (exit_date - entry_date).days
    pnl_pct = (exit_price - entry_price) / entry_price * 100

    # Max gain / drawdown during hold
    prices = df.loc[(df.index >= entry_date) & (df.index <= exit_date), "Close"].values
    max_gain = ((prices.max() - entry_price) / entry_price) * 100
    max_dd = ((prices.min() - entry_price) / entry_price) * 100

    # Conviction
    score, breakdown = score_conviction(event, entry_date, df)

    if score >= 80:
        size = 1.5
    elif score >= 60:
        size = 1.0
    elif score >= 30:
        size = 0.5
    else:
        size = 0.25

    trades.append({
        "Event": event,
        "Match": match_date.strftime("%Y-%m-%d"),
        "Entry Date": entry_date.date(),
        "Entry $": round(entry_price, 2),
        "Exit Date": exit_date.date(),
        "Exit $": round(exit_price, 2),
        "Days": days_held,
        "PnL %": round(pnl_pct, 1),
        "MaxG %": round(max_gain, 1),
        "MaxDD %": round(max_dd, 1),
        "Score": score,
        "Evt": breakdown["event"],
        "Vol": breakdown["vol"],
        "Mom": breakdown["momentum"],
        "Size": size,
        "Wtd PnL %": round(pnl_pct * size, 1),
        "Win": pnl_pct > 0,
    })

results = pd.DataFrame(trades).sort_values("Entry Date").reset_index(drop=True)

# ─────────────────────────────────────────────
# OUTPUT
# ─────────────────────────────────────────────
print("=" * 130)
print("CONVICTION-WEIGHTED MANU CATALYST — EXPANDED STOCK BACKTEST")
print("=" * 130)
print()
print("THESIS:")
print("  MANU stock shows pre-catalyst alpha around European competition stages.")
print("  By trading every stage (not just R16), we get ~3x more trades per year.")
print()
print("CONVICTION SCORING (0-100):")
print("  Event Importance (0-40): Final > SF > QF > R16 > R32 > Group (+5 for EL)")
print("  Vol Regime       (0-30): Low 20d/60d vol ratio = compressed, room to expand")
print("  Momentum         (0-30): Flat/slight positive = catalyst not priced in")
print()
print("POSITION SIZING:")
print("  Score 80-100 → 1.5x  |  60-79 → 1.0x  |  30-59 → 0.5x  |  0-29 → 0.25x")
print()
print("─" * 130)

# Trade table
print(f"\n  {'Event':<18} {'Entry':<12} {'Exit':<12} {'Days':>4} "
      f"{'PnL%':>7} {'MaxG%':>7} {'MaxDD%':>7} {'Score':>5} "
      f"{'E':>3} {'V':>3} {'M':>3} {'Size':>5} {'WtdPnL%':>8} {'':>5}")
print(f"  {'─'*110}")
for _, row in results.iterrows():
    w = "WIN" if row["Win"] else "LOSS"
    print(f"  {row['Event']:<18} {str(row['Entry Date']):<12} {str(row['Exit Date']):<12} {row['Days']:>4} "
          f"{row['PnL %']:>+6.1f}% {row['MaxG %']:>+6.1f}% {row['MaxDD %']:>+6.1f}% {row['Score']:>5} "
          f"{row['Evt']:>3} {row['Vol']:>3} {row['Mom']:>3} {row['Size']:>4.1f}x {row['Wtd PnL %']:>+7.1f}% "
          f"{w:>5}")
print()

# ─────────────────────────────────────────────
# STATS BY EVENT TYPE
# ─────────────────────────────────────────────
print("─" * 80)
print("BREAKDOWN BY EVENT TYPE")
print("─" * 80)

def classify_event(event):
    e = event.lower()
    if "group" in e:
        return "Group Stage"
    elif "r32" in e:
        return "Round of 32"
    elif "r16" in e:
        return "Round of 16"
    elif "qf" in e:
        return "Quarter-final"
    elif "sf" in e:
        return "Semi-final"
    elif "final" in e:
        return "Final"
    return "Other"

results["Stage"] = results["Event"].apply(classify_event)

stage_order = ["Group Stage", "Round of 32", "Round of 16", "Quarter-final", "Semi-final", "Final"]
for stage in stage_order:
    s = results[results["Stage"] == stage]
    if len(s) == 0:
        continue
    n = len(s)
    wr = s["Win"].mean() * 100
    avg = s["PnL %"].mean()
    std = s["PnL %"].std()
    print(f"  {stage:<18} | {n:>2} trades | WR: {wr:>5.0f}% | Avg: {avg:>+5.1f}% | Std: {std:>5.1f}%")

# CL vs EL
print()
results["Comp"] = results["Event"].apply(lambda e: "EL" if "EL" in e else "CL")
for comp in ["CL", "EL"]:
    s = results[results["Comp"] == comp]
    n = len(s)
    wr = s["Win"].mean() * 100
    avg = s["PnL %"].mean()
    print(f"  {comp:<18} | {n:>2} trades | WR: {wr:>5.0f}% | Avg: {avg:>+5.1f}%")
print()

# ─────────────────────────────────────────────
# SUMMARY COMPARISON
# ─────────────────────────────────────────────
n = len(results)
first_entry = pd.Timestamp(results["Entry Date"].iloc[0])
last_exit = pd.Timestamp(results["Exit Date"].iloc[-1])
years = (last_exit - first_entry).days / 365.25

def calc_stats(rets, label, n_trades, years):
    returns = rets / 100
    total = np.prod(1 + returns)
    cagr = (total ** (1 / years) - 1) * 100 if years > 0 else 0
    avg = rets.mean()
    std = rets.std()
    tpy = n_trades / years if years > 0 else 0
    annual_ret = avg * tpy
    annual_vol = std * np.sqrt(tpy) if tpy > 0 else 0
    sharpe = (annual_ret - 4.0) / annual_vol if annual_vol > 0 else np.nan
    wr = (rets > 0).mean() * 100
    return {
        "label": label, "n": n_trades, "wr": wr, "avg": avg, "std": std,
        "tpy": tpy, "total": (total - 1) * 100, "cagr": cagr,
        "annual_ret": annual_ret, "annual_vol": annual_vol, "sharpe": sharpe,
    }

eq = calc_stats(results["PnL %"], "Equal Weight", n, years)
wt = calc_stats(results["Wtd PnL %"], "Conviction Wtd", n, years)

# High conviction only
hc = results[results["Score"] >= 60]
n_hc = len(hc)
hc_years = years  # same time span
hc_stats = calc_stats(hc["PnL %"], "High Conv Only", n_hc, hc_years) if n_hc > 0 else None

# Knockout only (no group stages)
ko = results[results["Stage"] != "Group Stage"]
ko_stats = calc_stats(ko["PnL %"], "Knockout Only", len(ko), years) if len(ko) > 0 else None

print("=" * 100)
print(f"  {'Metric':<28} {'Equal Wt':>16} {'Conviction Wtd':>16} {'High Conv':>16} {'Knockout Only':>16}")
print(f"  {'':28} {'(all)':>16} {'(sized)':>16} {'(score≥60)':>16} {'(no groups)':>16}")
print("=" * 100)

rows = [
    ("Trades", "n", "{}"),
    ("Win Rate", "wr", "{:.0f}%"),
    ("Avg PnL / Trade", "avg", "{:+.1f}%"),
    ("Std PnL / Trade", "std", "{:.1f}%"),
    ("Trades / Year", "tpy", "{:.1f}"),
    ("Total Return", "total", "{:+.1f}%"),
    ("CAGR", "cagr", "{:+.1f}%"),
    ("Annualized Return", "annual_ret", "{:+.1f}%"),
    ("Annualized Vol", "annual_vol", "{:.1f}%"),
    ("Sharpe Ratio", "sharpe", "{:.2f}"),
]

for label, key, fmt in rows:
    v_eq = fmt.format(eq[key])
    v_wt = fmt.format(wt[key])
    v_hc = fmt.format(hc_stats[key]) if hc_stats else "-"
    v_ko = fmt.format(ko_stats[key]) if ko_stats else "-"
    print(f"  {label:<28} {v_eq:>16} {v_wt:>16} {v_hc:>16} {v_ko:>16}")

print("=" * 100)
print()

# Benchmark
np.random.seed(42)
bench = []
all_dates = df.index.tolist()
for _ in range(1000):
    idx = np.random.randint(0, len(all_dates) - HOLD_DAYS - 1)
    b_entry = all_dates[idx]
    b_exit_cands = df[(df.index > b_entry) & (df.index <= b_entry + timedelta(days=HOLD_DAYS))]
    if b_exit_cands.empty:
        continue
    b_exit = b_exit_cands.index[-1]
    b_ret = (float(df.loc[b_exit, "Close"]) - float(df.loc[b_entry, "Close"])) / float(df.loc[b_entry, "Close"]) * 100
    bench.append(b_ret)

print("─" * 60)
print("BENCHMARK: Random 30-day holds on MANU (n=1000)")
print("─" * 60)
print(f"  Avg return   : {np.mean(bench):+.1f}%")
print(f"  Win rate     : {np.mean([b > 0 for b in bench])*100:.0f}%")
print(f"  Median       : {np.median(bench):+.1f}%")
print()
print(f"  Strategy edge: {eq['avg'] - np.mean(bench):+.1f}% per trade (equal weight)")
print("─" * 60)
print()

# Conviction score distribution
print("─" * 60)
print("CONVICTION SCORE DISTRIBUTION")
print("─" * 60)
for bucket, label in [(range(0, 30), "0-29  (0.25x)"),
                       (range(30, 60), "30-59 (0.50x)"),
                       (range(60, 80), "60-79 (1.00x)"),
                       (range(80, 101), "80-100 (1.50x)")]:
    b = results[results["Score"].isin(bucket)]
    if len(b) > 0:
        wr = b["Win"].mean() * 100
        avg = b["PnL %"].mean()
        print(f"  {label:<16} {len(b):>3} trades | WR: {wr:>5.0f}% | Avg PnL: {avg:>+5.1f}%")
    else:
        print(f"  {label:<16}   0 trades")
print("─" * 60)
print()

# ─────────────────────────────────────────────
# PLOTS
# ─────────────────────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(14, 14))
fig.suptitle("MANU Conviction-Weighted Catalyst Backtest (Expanded)", fontsize=14, fontweight="bold")

# 1. Price chart with trade markers
ax = axes[0]
ax.plot(df.index, df["Close"], color="#1f77b4", lw=0.8, label="MANU Close")
for _, row in results.iterrows():
    ed = pd.Timestamp(row["Entry Date"])
    xd = pd.Timestamp(row["Exit Date"])
    color = "#2ca02c" if row["Win"] else "#d62728"
    ax.axvspan(ed, xd, alpha=0.08, color=color)
    ax.scatter(ed, row["Entry $"], marker="^", color="#2ca02c", s=40, zorder=5)
    ax.scatter(xd, row["Exit $"], marker="v", color=color, s=40, zorder=5)
ax.set_ylabel("Price ($)")
ax.set_title(f"MANU Price with {n} Trade Windows")
ax.grid(True, alpha=0.3)
ax.legend(loc="upper left", fontsize=9)

# 2. Equity curves
ax = axes[1]
eq_cum = [1.0]
wt_cum = [1.0]
dates = [first_entry]
for _, row in results.iterrows():
    eq_cum.append(eq_cum[-1] * (1 + row["PnL %"] / 100))
    wt_cum.append(wt_cum[-1] * (1 + row["Wtd PnL %"] / 100))
    dates.append(pd.Timestamp(row["Exit Date"]))

ax.plot(dates, eq_cum, color="#1f77b4", lw=2.0, marker="o", markersize=3, label="Equal Weight")
ax.plot(dates, wt_cum, color="#ff7f0e", lw=2.0, marker="s", markersize=3, label="Conviction Weighted")
ax.axhline(1.0, color="gray", lw=0.8, linestyle="--")
ax.set_ylabel("Cumulative Return")
ax.set_title("Equity Curve Comparison")
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# 3. Trade PnL bars colored by conviction
ax = axes[2]
colors = []
for _, row in results.iterrows():
    if row["Score"] >= 80:
        colors.append("#2ca02c")
    elif row["Score"] >= 60:
        colors.append("#1f77b4")
    elif row["Score"] >= 30:
        colors.append("#ff7f0e")
    else:
        colors.append("#d62728")

trade_dates = [pd.Timestamp(d) for d in results["Entry Date"]]
ax.bar(trade_dates, results["PnL %"], width=15, color=colors, alpha=0.8, edgecolor="white")
ax.axhline(0, color="black", lw=0.8)
ax.set_ylabel("Trade PnL %")
ax.set_xlabel("Entry Date")
ax.set_title("Individual Trades (green=1.5x, blue=1.0x, orange=0.5x, red=0.25x)")
ax.grid(True, alpha=0.3, axis="y")

plt.tight_layout()
plt.savefig("conviction_weighted_backtest.png", dpi=150, bbox_inches="tight")
print("Plot saved to conviction_weighted_backtest.png")
plt.close()
