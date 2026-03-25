"""
MSGS Playoff Volatility Trade — Calendar Spread Backtest
Trade 2: Buy longer-dated ATM call, sell shorter-dated ATM call.
Profits from vol expansion without needing a large spot move.
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

def bs_vega(S, K, T, r, sigma):
    """Black-Scholes vega (sensitivity to vol)."""
    if T <= 0 or sigma <= 0:
        return 0
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return S * norm.pdf(d1) * np.sqrt(T)

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
ENTRY_DAYS_BEFORE = 28          # enter 4 weeks before playoffs (fixed calendar)
RISK_FREE_RATE = 0.05
SHORT_EXPIRY_DAYS = 30          # front-month: expires ~30 days out
LONG_EXPIRY_DAYS = 90           # back-month: expires ~90 days out
SPREAD_PROFIT_TARGET = 0.20     # close spread at 20% profit

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
# 2. RETURNS
# ─────────────────────────────────────────────
df["ret"] = df["Close"].pct_change()

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
    # Fixed calendar entry: 4 weeks before playoffs
    target_entry = playoff_date - timedelta(days=ENTRY_DAYS_BEFORE)
    # Find the nearest trading day on or after target
    valid_days = df[df.index >= target_entry]
    if valid_days.empty:
        print(f"  {year}: No data around entry date")
        continue

    entry_date = valid_days.index[0]
    entry_price = float(df.loc[entry_date, "Close"])
    entry_vol = float(df.loc[entry_date, "garch_vol"])
    strike = entry_price  # ATM

    T_short_entry = SHORT_EXPIRY_DAYS / 365
    T_long_entry = LONG_EXPIRY_DAYS / 365

    # Price both legs at entry
    short_call_entry = bs_call(entry_price, strike, T_short_entry, RISK_FREE_RATE, entry_vol)
    long_call_entry = bs_call(entry_price, strike, T_long_entry, RISK_FREE_RATE, entry_vol)
    net_debit = long_call_entry - short_call_entry  # cost to enter

    # Compute entry vegas for context
    vega_short_entry = bs_vega(entry_price, strike, T_short_entry, RISK_FREE_RATE, entry_vol)
    vega_long_entry = bs_vega(entry_price, strike, T_long_entry, RISK_FREE_RATE, entry_vol)
    net_vega_entry = vega_long_entry - vega_short_entry

    # Check daily for profit target, otherwise hold to short expiry
    exit_target = entry_date + timedelta(days=SHORT_EXPIRY_DAYS)
    hold_window = df[(df.index > entry_date) & (df.index <= exit_target)]

    if hold_window.empty:
        continue

    exit_date = None
    exit_reason = "Short Expiry"

    for dt, row in hold_window.iterrows():
        spot = float(row["Close"])
        vol = float(row["garch_vol"])
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

    # If no early exit, close at short expiry
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

    trades.append({
        "Year": year,
        "Entry Date": entry_date.date(),
        "Entry Price": round(entry_price, 2),
        "Entry Vol": round(entry_vol * 100, 1),
        "Short Call Entry": round(short_call_entry, 2),
        "Long Call Entry": round(long_call_entry, 2),
        "Net Debit": round(net_debit, 2),
        "Net Vega": round(net_vega_entry, 2),
        "Exit Date": exit_date.date(),
        "Exit Reason": exit_reason,
        "Exit Price": round(exit_price, 2),
        "Exit Vol": round(exit_vol * 100, 1),
        "Spread Value Exit": round(spread_value_exit, 2),
        "Holding Days": days_held,
        "Spread PnL": round(pnl, 2),
        "PnL %": round(pnl_pct, 1),
        "Win": pnl > 0,
    })

# ─────────────────────────────────────────────
# 5. RESULTS TABLE
# ─────────────────────────────────────────────
results = pd.DataFrame(trades).dropna(subset=["Entry Date"])

print("=" * 120)
print("MSGS PLAYOFF VOLATILITY TRADE — CALENDAR SPREAD WITH PROFIT TARGET")
print("=" * 120)
print()
print("THESIS:")
print("  MSGS (Madison Square Garden Sports) owns the Knicks and Rangers. Revenue is")
print("  directly tied to playoff depth, but low analyst coverage and thin options markets")
print("  lead to underpriced volatility before the NBA playoffs.")
print()
print("STRATEGY:")
print(f"  Structure : Sell {SHORT_EXPIRY_DAYS}-day ATM call, buy {LONG_EXPIRY_DAYS}-day ATM call (calendar spread)")
print(f"  Entry     : Fixed calendar, {ENTRY_DAYS_BEFORE} days before NBA playoff start")
print(f"  Exit      : Close at {int(SPREAD_PROFIT_TARGET*100)}% profit target, or hold to short leg expiry ({SHORT_EXPIRY_DAYS} days)")
print("  Edge      : Theta decay (short leg decays faster) + vol expansion, without")
print("              needing a large spot move from MSGS")
print()
print("─" * 120)
display_cols = ["Year", "Entry Date", "Entry Price", "Entry Vol", "Net Debit",
                "Net Vega", "Exit Date", "Exit Price", "Exit Vol",
                "Spread Value Exit", "Holding Days", "Spread PnL", "PnL %", "Win"]
print(results[display_cols].to_string(index=False))
print()

# ─────────────────────────────────────────────
# 6. SUMMARY STATS
# ─────────────────────────────────────────────
n_trades = len(results)
wins = results["Win"].sum()
win_rate = wins / n_trades * 100 if n_trades > 0 else 0
avg_pnl = results["Spread PnL"].mean()
avg_pnl_pct = results["PnL %"].mean()
std_pnl = results["PnL %"].std()
sharpe_like = (avg_pnl_pct / std_pnl) * np.sqrt(n_trades) if std_pnl > 0 else np.nan

print("─" * 50)
print("OVERALL STATISTICS")
print("─" * 50)
print(f"  Total Trades      : {n_trades}")
print(f"  Wins              : {int(wins)}")
print(f"  Win Rate          : {win_rate:.1f}%")
print(f"  Avg Spread PnL    : ${avg_pnl:.2f} per spread")
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
fig.suptitle("MSGS Calendar Spread Backtest — Playoff Volatility Trade", fontsize=14, fontweight="bold")

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
    ax1.annotate(f"{row['Year']} (debit=${row['Net Debit']:.0f})", xy=(ed, ep), xytext=(5, 8),
                 textcoords="offset points", fontsize=8, color="#2ca02c")

ax1.set_ylabel("Price ($)")
ax1.grid(True, alpha=0.3)
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker="^", color="w", markerfacecolor="#2ca02c", markersize=9, label="Entry"),
    Line2D([0], [0], marker="v", color="w", markerfacecolor="#d62728", markersize=9, label="Exit (Short Expiry)"),
]
ax1.legend(handles=legend_elements + [ax1.get_lines()[0]], loc="upper left", fontsize=9)

ax2.plot(df.index, df["garch_vol"] * 100, color="#ff7f0e", lw=1.0, label="GJR-GARCH Vol (%)")
ax2.plot(df.index, df["garch_vol"] * 100, color="#8c564b", lw=0.8, alpha=0.6, label="GJR-GARCH Cond. Vol (%)")
for _, row in results.iterrows():
    ax2.axvline(pd.Timestamp(row["Entry Date"]), color="#2ca02c", lw=1.0, linestyle=":")
ax2.set_ylabel("Volatility (%)")
ax2.set_xlabel("Date")
ax2.legend(loc="upper right", fontsize=8)
ax2.grid(True, alpha=0.3)
ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax2.xaxis.set_major_locator(mdates.YearLocator())

plt.tight_layout()
plt.savefig("msgs_calendar_spread_backtest.png", dpi=150, bbox_inches="tight")
print("Plot saved to msgs_calendar_spread_backtest.png")
plt.close()
