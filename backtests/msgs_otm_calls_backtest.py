"""
MSGS Playoff Volatility Trade — OTM Calls Backtest
Trade 1: Buy OTM calls instead of ATM straddles. Directional, cheaper premium.
Uses Black-Scholes to price calls with GARCH vol as IV proxy.
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
# BLACK-SCHOLES CALL PRICING
# ─────────────────────────────────────────────
def bs_call(S, K, T, r, sigma):
    """Black-Scholes call price. T in years."""
    if T <= 0 or sigma <= 0:
        return max(S - K, 0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

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
ENTRY_WINDOW_DAYS = 21
ROUND1_DURATION_DAYS = 28
OTM_STRIKE_PCT = 0.05           # 5% out of the money
RISK_FREE_RATE = 0.05           # ~5% risk-free rate
OPTION_EXPIRY_DAYS = 45         # option expires ~45 days out
REALIZED_VOL_WINDOW = 30
PERCENTILE_WINDOW = 504
ENTRY_PERCENTILE = 20

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
df["rv30"] = df["ret"].rolling(REALIZED_VOL_WINDOW).std() * np.sqrt(252)
df["rv30_pct20"] = df["rv30"].rolling(PERCENTILE_WINDOW).quantile(ENTRY_PERCENTILE / 100)

# ─────────────────────────────────────────────
# 3. GJR-GARCH(1,1) CONDITIONAL VOLATILITY
# ─────────────────────────────────────────────
print("Fitting GJR-GARCH(1,1) model...")
returns_pct = df["ret"].dropna() * 100
gjr = arch_model(returns_pct, vol="GARCH", p=1, o=1, q=1, dist="Normal")
res = gjr.fit(disp="off")
cond_vol = res.conditional_volatility / 100 * np.sqrt(252)
df["garch_vol"] = np.nan
df.loc[returns_pct.index, "garch_vol"] = cond_vol.values
print(f"  GARCH fit complete. AIC={res.aic:.1f}\n")

# ─────────────────────────────────────────────
# 4. BACKTEST LOOP
# ─────────────────────────────────────────────
trades = []

for year, playoff_date in PLAYOFF_STARTS.items():
    window_start = playoff_date - timedelta(days=ENTRY_WINDOW_DAYS)

    mask = (
        (df.index >= window_start)
        & (df.index < playoff_date)
        & (df["rv30"] < df["rv30_pct20"])
    )
    signal_days = df[mask]

    if signal_days.empty:
        print(f"  {year}: No entry signal found")
        trades.append({
            "Year": year, "Entry Date": None, "Entry Price": None,
            "Exit Date": None, "Exit Price": None,
            "Holding Days": None, "Call PnL": None, "Win": None,
        })
        continue

    entry_date = signal_days.index[0]
    entry_price = float(df.loc[entry_date, "Close"])
    entry_vol = float(df.loc[entry_date, "garch_vol"])
    strike = entry_price * (1 + OTM_STRIKE_PCT)
    T_entry = OPTION_EXPIRY_DAYS / 365

    # Price the call at entry
    call_entry = bs_call(entry_price, strike, T_entry, RISK_FREE_RATE, entry_vol)

    # Hold until end of Round 1
    max_exit = playoff_date + timedelta(days=ROUND1_DURATION_DAYS)
    hold_window = df[(df.index > entry_date) & (df.index <= max_exit)]

    if hold_window.empty:
        continue

    exit_date = hold_window.index[-1]
    exit_price = float(df.loc[exit_date, "Close"])
    exit_vol = float(df.loc[exit_date, "garch_vol"])
    days_held = (exit_date - entry_date).days
    T_exit = max((OPTION_EXPIRY_DAYS - days_held) / 365, 0)

    # Reprice the call at exit
    call_exit = bs_call(exit_price, strike, T_exit, RISK_FREE_RATE, exit_vol)

    pnl = call_exit - call_entry
    pnl_pct = (pnl / call_entry) * 100 if call_entry > 0 else 0

    trades.append({
        "Year": year,
        "Entry Date": entry_date.date(),
        "Entry Price": round(entry_price, 2),
        "Strike": round(strike, 2),
        "Entry Vol": round(entry_vol * 100, 1),
        "Call Premium": round(call_entry, 2),
        "Exit Date": exit_date.date(),
        "Exit Price": round(exit_price, 2),
        "Exit Vol": round(exit_vol * 100, 1),
        "Call Value at Exit": round(call_exit, 2),
        "Holding Days": days_held,
        "Call PnL": round(pnl, 2),
        "PnL %": round(pnl_pct, 1),
        "Win": pnl > 0,
    })

# ─────────────────────────────────────────────
# 5. RESULTS TABLE
# ─────────────────────────────────────────────
results = pd.DataFrame(trades).dropna(subset=["Entry Date"])

print("=" * 100)
print("MSGS OTM CALLS BACKTEST — RESULTS")
print("=" * 100)
display_cols = ["Year", "Entry Date", "Entry Price", "Strike", "Entry Vol",
                "Call Premium", "Exit Date", "Exit Price", "Exit Vol",
                "Call Value at Exit", "Holding Days", "Call PnL", "PnL %", "Win"]
print(results[display_cols].to_string(index=False))
print()

# ─────────────────────────────────────────────
# 6. SUMMARY STATS
# ─────────────────────────────────────────────
n_trades = len(results)
wins = results["Win"].sum()
win_rate = wins / n_trades * 100 if n_trades > 0 else 0
avg_pnl = results["Call PnL"].mean()
avg_pnl_pct = results["PnL %"].mean()
std_pnl = results["PnL %"].std()
sharpe_like = (avg_pnl_pct / std_pnl) * np.sqrt(n_trades) if std_pnl > 0 else np.nan

print("─" * 50)
print("OVERALL STATISTICS")
print("─" * 50)
print(f"  Total Trades      : {n_trades}")
print(f"  Wins              : {int(wins)}")
print(f"  Win Rate          : {win_rate:.1f}%")
print(f"  Avg Call PnL      : ${avg_pnl:.2f} per contract")
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
fig.suptitle("MSGS OTM Calls Backtest — Playoff Volatility Trade", fontsize=14, fontweight="bold")

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
    ax1.axhline(row["Strike"], xmin=0, xmax=1, color="gray", ls="--", lw=0.7, alpha=0.5)
    ax1.annotate(f"{row['Year']} (K={row['Strike']:.0f})", xy=(ed, ep), xytext=(5, 8),
                 textcoords="offset points", fontsize=8, color="#2ca02c")

ax1.set_ylabel("Price ($)")
ax1.grid(True, alpha=0.3)
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker="^", color="w", markerfacecolor="#2ca02c", markersize=9, label="Entry"),
    Line2D([0], [0], marker="v", color="w", markerfacecolor="#d62728", markersize=9, label="Exit"),
    Line2D([0], [0], color="gray", ls="--", lw=0.7, label="OTM Strike"),
]
ax1.legend(handles=legend_elements + [ax1.get_lines()[0]], loc="upper left", fontsize=9)

ax2.plot(df.index, df["rv30"] * 100, color="#ff7f0e", lw=1.0, label="30-Day Realized Vol (%)")
ax2.plot(df.index, df["garch_vol"] * 100, color="#8c564b", lw=0.8, alpha=0.6, label="GJR-GARCH Vol (%)")
for _, row in results.iterrows():
    ax2.axvline(pd.Timestamp(row["Entry Date"]), color="#2ca02c", lw=1.0, linestyle=":")
ax2.set_ylabel("Volatility (%)")
ax2.set_xlabel("Date")
ax2.legend(loc="upper right", fontsize=8)
ax2.grid(True, alpha=0.3)
ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax2.xaxis.set_major_locator(mdates.YearLocator())

plt.tight_layout()
plt.savefig("msgs_otm_calls_backtest.png", dpi=150, bbox_inches="tight")
print("Plot saved to msgs_otm_calls_backtest.png")
plt.close()
