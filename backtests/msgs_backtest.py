"""
MSGS Playoff Volatility Trade Backtest
Strategy: Buy ATM straddle 3 weeks before NBA playoffs when realized vol is unusually low.
"""

import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")  # non-interactive backend so plt.close() doesn't block
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from arch import arch_model
from datetime import timedelta

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
TICKER = "MSGS"
PLAYOFF_STARTS = {
    2019: pd.Timestamp("2019-04-13"),
    2021: pd.Timestamp("2021-05-22"),
    2022: pd.Timestamp("2022-04-16"),
    2023: pd.Timestamp("2023-04-15"),
    2024: pd.Timestamp("2024-04-20"),
}
ENTRY_WINDOW_DAYS = 21          # 3 weeks before playoff start
ROUND1_DURATION_DAYS = 28       # ~4 weeks for Round 1
IV_SPIKE_THRESHOLD = 0.30       # exit if IV rises 30%+ above entry
ATM_PREMIUM_PCT = 0.05          # 5% of spot = straddle cost
REALIZED_VOL_WINDOW = 30        # days for rolling realized vol
PERCENTILE_WINDOW = 504         # ~2 trading years
ENTRY_PERCENTILE = 20           # signal if rv < 20th pct

# ─────────────────────────────────────────────
# 1. FETCH DATA
# ─────────────────────────────────────────────
print("Fetching MSGS data from Yahoo Finance...")
raw = yf.download(TICKER, start="2018-01-01", end="2024-12-31", auto_adjust=True, progress=False)
raw.columns = raw.columns.get_level_values(0) if isinstance(raw.columns, pd.MultiIndex) else raw.columns
df = raw[["Open", "High", "Low", "Close", "Volume"]].copy()
df.index = pd.to_datetime(df.index)
df = df.dropna()
print(f"  Loaded {len(df)} trading days ({df.index[0].date()} → {df.index[-1].date()})\n")

# ─────────────────────────────────────────────
# 2. RETURNS & REALIZED VOLATILITY
# ─────────────────────────────────────────────
df["ret"] = df["Close"].pct_change()

# 30-day realized vol (annualized)
df["rv30"] = df["ret"].rolling(REALIZED_VOL_WINDOW).std() * np.sqrt(252)

# 20th percentile of rv30 over rolling 2-year window
df["rv30_pct20"] = df["rv30"].rolling(PERCENTILE_WINDOW).quantile(ENTRY_PERCENTILE / 100)

# ─────────────────────────────────────────────
# 3. GJR-GARCH(1,1) CONDITIONAL VOLATILITY
# ─────────────────────────────────────────────
print("Fitting GJR-GARCH(1,1) model...")
returns_pct = df["ret"].dropna() * 100  # arch wants %-scale returns
gjr = arch_model(returns_pct, vol="GARCH", p=1, o=1, q=1, dist="Normal")
res = gjr.fit(disp="off")
cond_vol = res.conditional_volatility / 100 * np.sqrt(252)  # annualized
df["garch_vol"] = np.nan
df.loc[returns_pct.index, "garch_vol"] = cond_vol.values
print(f"  GARCH fit complete. AIC={res.aic:.1f}\n")

# ─────────────────────────────────────────────
# 4. BACKTEST LOOP
# ─────────────────────────────────────────────
trades = []

for year, playoff_date in PLAYOFF_STARTS.items():
    window_start = playoff_date - timedelta(days=ENTRY_WINDOW_DAYS)

    # Find first signal day inside the entry window
    mask = (
        (df.index >= window_start)
        & (df.index < playoff_date)
        & (df["rv30"] < df["rv30_pct20"])
    )
    signal_days = df[mask]

    if signal_days.empty:
        print(f"  {year}: No entry signal found in window "
              f"({window_start.date()} → {playoff_date.date()})")
        trades.append({
            "Year": year, "Entry Date": None, "Entry Price": None,
            "Exit Date": None, "Exit Price": None,
            "Holding Days": None, "Straddle PnL": None, "Win": None,
        })
        continue

    entry_date = signal_days.index[0]
    entry_price = float(df.loc[entry_date, "Close"])
    atm_premium = ATM_PREMIUM_PCT * entry_price
    entry_garch_vol = float(df.loc[entry_date, "garch_vol"])

    # Max exit = end of Round 1
    max_exit = playoff_date + timedelta(days=ROUND1_DURATION_DAYS)
    hold_window = df[(df.index > entry_date) & (df.index <= max_exit)]

    exit_date = hold_window.index[-1]   # default: end of Round 1
    exit_reason = "Round 1 End"

    # IV spike proxy: use GARCH vol rising 30%+ above entry level
    for dt, row in hold_window.iterrows():
        gv = row["garch_vol"]
        if not np.isnan(gv) and gv >= entry_garch_vol * (1 + IV_SPIKE_THRESHOLD):
            exit_date = dt
            exit_reason = "IV Spike"
            break

    exit_price = float(df.loc[exit_date, "Close"])
    price_move = abs(exit_price - entry_price)
    straddle_pnl = price_move - 2 * atm_premium   # per-share proxy
    pnl_pct = straddle_pnl / (2 * atm_premium)     # return on premium paid

    trades.append({
        "Year": year,
        "Entry Date": entry_date.date(),
        "Entry Price": round(entry_price, 2),
        "Exit Date": exit_date.date(),
        "Exit Price": round(exit_price, 2),
        "Holding Days": (exit_date - entry_date).days,
        "Exit Reason": exit_reason,
        "ATM Premium": round(atm_premium, 2),
        "Price Move": round(price_move, 2),
        "Straddle PnL": round(straddle_pnl, 2),
        "PnL %": round(pnl_pct * 100, 1),
        "Win": straddle_pnl > 0,
    })

# ─────────────────────────────────────────────
# 5. RESULTS TABLE
# ─────────────────────────────────────────────
results = pd.DataFrame(trades).dropna(subset=["Entry Date"])

print("=" * 90)
print("MSGS PLAYOFF VOLATILITY TRADE — BACKTEST RESULTS")
print("=" * 90)
display_cols = ["Year", "Entry Date", "Entry Price", "Exit Date", "Exit Price",
                "Holding Days", "Exit Reason", "ATM Premium", "Price Move",
                "Straddle PnL", "PnL %", "Win"]
print(results[display_cols].to_string(index=False))
print()

# ─────────────────────────────────────────────
# 6. SUMMARY STATS
# ─────────────────────────────────────────────
n_trades = len(results)
wins = results["Win"].sum()
win_rate = wins / n_trades * 100
avg_pnl = results["Straddle PnL"].mean()
avg_pnl_pct = results["PnL %"].mean()
std_pnl = results["PnL %"].std()
sharpe_like = (avg_pnl_pct / std_pnl) * np.sqrt(n_trades) if std_pnl > 0 else np.nan

print("─" * 50)
print("OVERALL STATISTICS")
print("─" * 50)
print(f"  Total Trades      : {n_trades}")
print(f"  Wins              : {int(wins)}")
print(f"  Win Rate          : {win_rate:.1f}%")
print(f"  Avg Straddle PnL  : ${avg_pnl:.2f} per share")
print(f"  Avg PnL %         : {avg_pnl_pct:.1f}%")
print(f"  Std PnL %         : {std_pnl:.1f}%")
print(f"  Sharpe-like Ratio : {sharpe_like:.2f}")
print("─" * 50)
print()

# ─────────────────────────────────────────────
# 7. PLOT
# ─────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9), sharex=True,
                                gridspec_kw={"height_ratios": [2, 1]})
fig.suptitle("MSGS Playoff Volatility Trade — Backtest", fontsize=14, fontweight="bold")

# ── Panel 1: Price ──
ax1.plot(df.index, df["Close"], color="#1f77b4", lw=1.2, label="MSGS Close")
for _, row in results.iterrows():
    ed = pd.Timestamp(row["Entry Date"])
    xd = pd.Timestamp(row["Exit Date"])
    ep = row["Entry Price"]
    xp = row["Exit Price"]
    color = "#2ca02c" if row["Win"] else "#d62728"
    ax1.axvspan(ed, xd, alpha=0.08, color=color)
    ax1.scatter(ed, ep, marker="^", color="#2ca02c", s=100, zorder=5)
    ax1.scatter(xd, xp, marker="v", color="#d62728", s=100, zorder=5)
    ax1.annotate(f"{row['Year']}", xy=(ed, ep), xytext=(5, 8),
                 textcoords="offset points", fontsize=8, color="#2ca02c")

ax1.set_ylabel("Price ($)")
ax1.legend(loc="upper left", fontsize=9)
ax1.grid(True, alpha=0.3)

# Add legend proxies for entry/exit markers
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker="^", color="w", markerfacecolor="#2ca02c", markersize=9, label="Entry"),
    Line2D([0], [0], marker="v", color="w", markerfacecolor="#d62728", markersize=9, label="Exit"),
]
ax1.legend(handles=legend_elements + [ax1.get_lines()[0]], loc="upper left", fontsize=9)

# ── Panel 2: Realized Vol ──
ax2.plot(df.index, df["rv30"] * 100, color="#ff7f0e", lw=1.0, label="30-Day Realized Vol (%)")
ax2.plot(df.index, df["rv30_pct20"] * 100, color="#9467bd", lw=1.0,
         linestyle="--", label="20th Percentile Threshold")
ax2.plot(df.index, df["garch_vol"] * 100, color="#8c564b", lw=0.8,
         alpha=0.6, label="GJR-GARCH Cond. Vol (%)")

for _, row in results.iterrows():
    ax2.axvline(pd.Timestamp(row["Entry Date"]), color="#2ca02c", lw=1.0, linestyle=":")

ax2.set_ylabel("Volatility (%)")
ax2.set_xlabel("Date")
ax2.legend(loc="upper right", fontsize=8)
ax2.grid(True, alpha=0.3)
ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax2.xaxis.set_major_locator(mdates.YearLocator())

plt.tight_layout()
plt.savefig("msgs_backtest.png", dpi=150, bbox_inches="tight")
print("Plot saved to msgs_backtest.png")
plt.close()
