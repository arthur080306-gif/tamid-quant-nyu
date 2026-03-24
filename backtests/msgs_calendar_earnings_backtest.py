"""
MSGS Playoff Volatility Trade — Calendar Spread + Earnings Hold
Phase 1: Calendar spread (sell 30-day, buy 90-day ATM call) with 20% profit target.
Phase 2: After short leg expires/closed, hold the long call through Q4 earnings (~August).
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
# Approximate Q4 earnings dates (MSGS reports fiscal Q3/Q4 around Aug/Nov)
# These are approximate — MSGS fiscal year ends June 30, so Q4 earnings ~ Aug/Sep
EARNINGS_DATES = {
    2019: pd.Timestamp("2019-08-14"),
    2021: pd.Timestamp("2021-08-12"),
    2022: pd.Timestamp("2022-08-11"),
    2023: pd.Timestamp("2023-08-10"),
    2024: pd.Timestamp("2024-08-08"),
}
ENTRY_DAYS_BEFORE = 28          # enter 4 weeks before playoffs
RISK_FREE_RATE = 0.05
SHORT_EXPIRY_DAYS = 30          # front-month: expires ~30 days out
LONG_EXPIRY_DAYS = 150          # back-month: extended to cover through earnings
SPREAD_PROFIT_TARGET = 0.20     # close spread phase at 20% profit

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
    earnings_date = EARNINGS_DATES[year]

    # ── ENTRY ──
    target_entry = playoff_date - timedelta(days=ENTRY_DAYS_BEFORE)
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
    net_debit = long_call_entry - short_call_entry
    net_vega = bs_vega(entry_price, strike, T_long_entry, RISK_FREE_RATE, entry_vol) - \
               bs_vega(entry_price, strike, T_short_entry, RISK_FREE_RATE, entry_vol)

    # ── PHASE 1: CALENDAR SPREAD ──
    # Check daily for profit target, otherwise hold to short expiry
    short_expiry_target = entry_date + timedelta(days=SHORT_EXPIRY_DAYS)
    phase1_window = df[(df.index > entry_date) & (df.index <= short_expiry_target)]

    phase1_exit_date = None
    phase1_exit_reason = "Short Expiry"
    spread_pnl = 0
    short_call_recovered = 0

    for dt, row in phase1_window.iterrows():
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
            phase1_exit_date = dt
            phase1_exit_reason = f"Profit Target ({current_pnl_pct*100:.0f}%)"
            spread_pnl = spread_value - net_debit
            short_call_recovered = short_now  # buy back the short
            break

    # If no early exit, close at short expiry
    if phase1_exit_date is None and not phase1_window.empty:
        phase1_exit_date = phase1_window.index[-1]
        spot = float(df.loc[phase1_exit_date, "Close"])
        vol = float(df.loc[phase1_exit_date, "garch_vol"])
        days_elapsed = (phase1_exit_date - entry_date).days
        T_short_now = max((SHORT_EXPIRY_DAYS - days_elapsed) / 365, 0)
        T_long_now = max((LONG_EXPIRY_DAYS - days_elapsed) / 365, 0)
        short_now = bs_call(spot, strike, T_short_now, RISK_FREE_RATE, vol)
        long_now = bs_call(spot, strike, T_long_now, RISK_FREE_RATE, vol)
        spread_pnl = (long_now - short_now) - net_debit
        short_call_recovered = short_now

    phase1_days = (phase1_exit_date - entry_date).days if phase1_exit_date else 0

    # ── PHASE 2: HOLD LONG CALL THROUGH EARNINGS ──
    # Value the long call at phase 1 exit (this is what we're holding)
    p1_exit_spot = float(df.loc[phase1_exit_date, "Close"])
    p1_exit_vol = float(df.loc[phase1_exit_date, "garch_vol"])
    p1_days_elapsed = (phase1_exit_date - entry_date).days
    T_long_at_p1_exit = max((LONG_EXPIRY_DAYS - p1_days_elapsed) / 365, 0)
    long_call_at_p1_exit = bs_call(p1_exit_spot, strike, T_long_at_p1_exit, RISK_FREE_RATE, p1_exit_vol)

    # Find earnings date exit (or nearest trading day)
    earnings_window = df[(df.index >= earnings_date - timedelta(days=3)) & (df.index <= earnings_date + timedelta(days=3))]
    if earnings_window.empty:
        # Fall back to long expiry
        earnings_exit = df[df.index <= entry_date + timedelta(days=LONG_EXPIRY_DAYS)].index[-1]
    else:
        earnings_exit = earnings_window.index[-1]

    # Make sure we have data
    if earnings_exit not in df.index:
        earnings_exit = df[df.index <= earnings_exit].index[-1]

    final_price = float(df.loc[earnings_exit, "Close"])
    final_vol = float(df.loc[earnings_exit, "garch_vol"])
    total_days = (earnings_exit - entry_date).days
    T_long_final = max((LONG_EXPIRY_DAYS - total_days) / 365, 0)
    long_call_final = bs_call(final_price, strike, T_long_final, RISK_FREE_RATE, final_vol)

    # Phase 2 PnL: change in long call value from phase 1 exit to earnings
    phase2_pnl = long_call_final - long_call_at_p1_exit
    phase2_days = (earnings_exit - phase1_exit_date).days

    # ── TOTAL PnL ──
    # Total cost basis = net_debit (what we paid for the spread)
    # Total return = spread_pnl (phase 1) + phase2_pnl (long call appreciation)
    total_pnl = spread_pnl + phase2_pnl
    total_pnl_pct = (total_pnl / net_debit) * 100 if net_debit > 0 else 0

    trades.append({
        "Year": year,
        "Entry Date": entry_date.date(),
        "Entry Price": round(entry_price, 2),
        "Entry Vol": round(entry_vol * 100, 1),
        "Net Debit": round(net_debit, 2),
        # Phase 1
        "P1 Exit": phase1_exit_date.date(),
        "P1 Reason": phase1_exit_reason,
        "P1 Days": phase1_days,
        "Spread PnL": round(spread_pnl, 2),
        "Spread PnL %": round((spread_pnl / net_debit) * 100, 1) if net_debit > 0 else 0,
        # Phase 2
        "Earnings Date": earnings_exit.date(),
        "Earnings Price": round(final_price, 2),
        "P2 Days": phase2_days,
        "Long Call PnL": round(phase2_pnl, 2),
        "Long Call PnL %": round((phase2_pnl / long_call_at_p1_exit) * 100, 1) if long_call_at_p1_exit > 0 else 0,
        # Total
        "Total PnL": round(total_pnl, 2),
        "Total PnL %": round(total_pnl_pct, 1),
        "Total Days": total_days,
        "Win": total_pnl > 0,
    })

# ─────────────────────────────────────────────
# 5. RESULTS TABLE
# ─────────────────────────────────────────────
results = pd.DataFrame(trades)

print("=" * 130)
print("MSGS CALENDAR SPREAD + EARNINGS HOLD — BACKTEST RESULTS")
print("=" * 130)

print("\n── PHASE 1: Calendar Spread ──")
p1_cols = ["Year", "Entry Date", "Entry Price", "Net Debit", "P1 Exit", "P1 Reason", "P1 Days", "Spread PnL", "Spread PnL %"]
print(results[p1_cols].to_string(index=False))

print("\n── PHASE 2: Long Call → Earnings ──")
p2_cols = ["Year", "P1 Exit", "Earnings Date", "Earnings Price", "P2 Days", "Long Call PnL", "Long Call PnL %"]
print(results[p2_cols].to_string(index=False))

print("\n── COMBINED ──")
total_cols = ["Year", "Entry Date", "Earnings Date", "Total Days", "Net Debit", "Spread PnL", "Long Call PnL", "Total PnL", "Total PnL %", "Win"]
print(results[total_cols].to_string(index=False))
print()

# ─────────────────────────────────────────────
# 6. SUMMARY STATS
# ─────────────────────────────────────────────
n_trades = len(results)
wins = results["Win"].sum()
win_rate = wins / n_trades * 100 if n_trades > 0 else 0
avg_pnl = results["Total PnL"].mean()
avg_pnl_pct = results["Total PnL %"].mean()
std_pnl = results["Total PnL %"].std()
sharpe_like = (avg_pnl_pct / std_pnl) * np.sqrt(n_trades) if std_pnl > 0 else np.nan
max_drawdown = results["Total PnL %"].min()
best_trade = results["Total PnL %"].max()

# Phase 1 stats
p1_win_rate = (results["Spread PnL"] > 0).sum() / n_trades * 100
p1_avg = results["Spread PnL %"].mean()

# Phase 2 stats
p2_win_rate = (results["Long Call PnL"] > 0).sum() / n_trades * 100
p2_avg = results["Long Call PnL %"].mean()

print("─" * 60)
print("OVERALL STATISTICS")
print("─" * 60)
print(f"  Total Trades        : {n_trades}")
print(f"  Wins                : {int(wins)}")
print(f"  Win Rate            : {win_rate:.1f}%")
print(f"  Avg Total PnL       : ${avg_pnl:.2f} per trade")
print(f"  Avg Total PnL %     : {avg_pnl_pct:.1f}%")
print(f"  Std PnL %           : {std_pnl:.1f}%")
print(f"  Sharpe-like Ratio   : {sharpe_like:.2f}")
print(f"  Best Trade          : {best_trade:.1f}%")
print(f"  Worst Trade         : {max_drawdown:.1f}%")
print()
print(f"  Phase 1 (Spread)    : {p1_avg:.1f}% avg, {p1_win_rate:.0f}% win rate")
print(f"  Phase 2 (Earnings)  : {p2_avg:.1f}% avg, {p2_win_rate:.0f}% win rate")
print("─" * 60)
print()

# ─────────────────────────────────────────────
# 7. PLOT
# ─────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9), sharex=True,
                                gridspec_kw={"height_ratios": [2, 1]})
fig.suptitle("MSGS Calendar Spread + Earnings Hold — Backtest", fontsize=14, fontweight="bold")

ax1.plot(df.index, df["Close"], color="#1f77b4", lw=1.2, label="MSGS Close")
for _, row in results.iterrows():
    ed = pd.Timestamp(row["Entry Date"])
    p1x = pd.Timestamp(row["P1 Exit"])
    earnd = pd.Timestamp(row["Earnings Date"])
    ep = row["Entry Price"]
    earnp = row["Earnings Price"]
    total_color = "#2ca02c" if row["Win"] else "#d62728"

    # Phase 1 shading (blue)
    ax1.axvspan(ed, p1x, alpha=0.06, color="#1f77b4")
    # Phase 2 shading (orange)
    ax1.axvspan(p1x, earnd, alpha=0.06, color="#ff7f0e")

    ax1.scatter(ed, ep, marker="^", color="#2ca02c", s=100, zorder=5)
    ax1.scatter(earnd, earnp, marker="v", color=total_color, s=100, zorder=5)
    ax1.annotate(f"{row['Year']} ({row['Total PnL %']:+.0f}%)", xy=(ed, ep), xytext=(5, 8),
                 textcoords="offset points", fontsize=8, color=total_color)

ax1.set_ylabel("Price ($)")
ax1.grid(True, alpha=0.3)
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
legend_elements = [
    Line2D([0], [0], marker="^", color="w", markerfacecolor="#2ca02c", markersize=9, label="Entry"),
    Line2D([0], [0], marker="v", color="w", markerfacecolor="#d62728", markersize=9, label="Earnings Exit"),
    Patch(facecolor="#1f77b4", alpha=0.15, label="Phase 1: Spread"),
    Patch(facecolor="#ff7f0e", alpha=0.15, label="Phase 2: Long Call"),
]
ax1.legend(handles=legend_elements + [ax1.get_lines()[0]], loc="upper left", fontsize=8)

ax2.plot(df.index, df["garch_vol"] * 100, color="#8c564b", lw=1.0, label="GJR-GARCH Vol (%)")
for _, row in results.iterrows():
    ax2.axvline(pd.Timestamp(row["Entry Date"]), color="#2ca02c", lw=1.0, linestyle=":")
    ax2.axvline(pd.Timestamp(row["Earnings Date"]), color="#d62728", lw=1.0, linestyle=":")
ax2.set_ylabel("Volatility (%)")
ax2.set_xlabel("Date")
ax2.legend(loc="upper right", fontsize=8)
ax2.grid(True, alpha=0.3)
ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax2.xaxis.set_major_locator(mdates.YearLocator())

plt.tight_layout()
plt.savefig("msgs_calendar_earnings_backtest.png", dpi=150, bbox_inches="tight")
print("Plot saved to msgs_calendar_earnings_backtest.png")
plt.close()
