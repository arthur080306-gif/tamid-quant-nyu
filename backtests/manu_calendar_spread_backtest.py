"""
MANU Champions League Volatility Trade — Calendar Spread with Profit Target
Strategy: Sell 30-day ATM call, buy 90-day ATM call, 4 weeks before CL Round of 16.
Same thesis as MSGS playoff trade: deep tournament runs lift revenue, but the
stock is low-vol and the market underprices the catalyst.
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
TICKER = "MANU"
# CL Round of 16 first leg dates (seasons MANU qualified)
CL_R16_DATES = {
    "2012-13": pd.Timestamp("2013-02-13"),
    "2013-14": pd.Timestamp("2014-02-25"),
    "2017-18": pd.Timestamp("2018-02-21"),
    "2018-19": pd.Timestamp("2019-02-12"),
    "2021-22": pd.Timestamp("2022-02-23"),
}
ENTRY_DAYS_BEFORE = 28          # enter 4 weeks before R16 first leg
RISK_FREE_RATE = 0.04           # avg risk-free rate over period
SHORT_EXPIRY_DAYS = 30          # front-month
LONG_EXPIRY_DAYS = 90           # back-month
SPREAD_PROFIT_TARGET = 0.20     # close at 20% profit

# EXECUTION COST ADJUSTMENTS (from real data comparison on 2026-03-24)
# Model underprices by ~24% on avg → real options cost more
# MANU avg bid/ask spread: ~111% → massive execution cost

# Realistic: limit orders fill ~midpoint, partial model error
REALISTIC_MODEL_ERR = 0.12      # half of full model error
REALISTIC_SPREAD_PCT = 0.25     # limit orders capture ~50% of spread back

# Worst case: market orders, full model error, wide spreads
WORST_MODEL_ERR = 0.24          # full model underpricing
WORST_SPREAD_PCT = 0.50         # conservative (real was 111%)

# ─────────────────────────────────────────────
# 1. FETCH DATA
# ─────────────────────────────────────────────
print("Fetching MANU data from Yahoo Finance...")
raw = yf.download(TICKER, start="2012-01-01", end="2024-12-31", auto_adjust=True, progress=False)
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

for season, r16_date in CL_R16_DATES.items():
    target_entry = r16_date - timedelta(days=ENTRY_DAYS_BEFORE)
    valid_days = df[df.index >= target_entry]
    if valid_days.empty:
        print(f"  {season}: No data around entry date")
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

    # Compute entry vegas
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

    # ── REALISTIC SCENARIO ──
    # Assumes limit orders capture ~half the spread, half the model error
    long_entry_r = long_call_entry * (1 + REALISTIC_MODEL_ERR) * (1 + REALISTIC_SPREAD_PCT / 2)
    short_entry_r = short_call_entry * (1 + REALISTIC_MODEL_ERR) * (1 - REALISTIC_SPREAD_PCT / 2)
    net_debit_real = long_entry_r - short_entry_r

    long_exit_r = long_call_exit * (1 + REALISTIC_MODEL_ERR) * (1 - REALISTIC_SPREAD_PCT / 2)
    short_exit_r = short_call_exit * (1 + REALISTIC_MODEL_ERR) * (1 + REALISTIC_SPREAD_PCT / 2)
    pnl_realistic = (short_entry_r - short_exit_r) + (long_exit_r - long_entry_r)
    pnl_pct_realistic = (pnl_realistic / net_debit_real) * 100 if net_debit_real > 0 else 0

    # ── WORST-CASE SCENARIO ──
    # Full model error + wide spreads, market orders
    long_entry_w = long_call_entry * (1 + WORST_MODEL_ERR) * (1 + WORST_SPREAD_PCT / 2)
    short_entry_w = short_call_entry * (1 + WORST_MODEL_ERR) * (1 - WORST_SPREAD_PCT / 2)
    net_debit_worst = long_entry_w - short_entry_w

    long_exit_w = long_call_exit * (1 + WORST_MODEL_ERR) * (1 - WORST_SPREAD_PCT / 2)
    short_exit_w = short_call_exit * (1 + WORST_MODEL_ERR) * (1 + WORST_SPREAD_PCT / 2)
    pnl_worst = (short_entry_w - short_exit_w) + (long_exit_w - long_entry_w)
    pnl_pct_worst = (pnl_worst / net_debit_worst) * 100 if net_debit_worst > 0 else 0

    trades.append({
        "Season": season,
        "Entry Date": entry_date.date(),
        "Entry Price": round(entry_price, 2),
        "Entry Vol": round(entry_vol * 100, 1),
        "Net Debit": round(net_debit, 2),
        "Net Vega": round(net_vega_entry, 2),
        "Exit Date": exit_date.date(),
        "Exit Reason": exit_reason,
        "Exit Price": round(exit_price, 2),
        "Holding Days": days_held,
        # Model (mid-price)
        "PnL %": round(pnl_pct, 1),
        "Win": pnl > 0,
        # Realistic
        "PnL % (real)": round(pnl_pct_realistic, 1),
        "Win (real)": pnl_realistic > 0,
        # Worst case
        "PnL % (worst)": round(pnl_pct_worst, 1),
        "Win (worst)": pnl_worst > 0,
    })

# ─────────────────────────────────────────────
# 5. RESULTS TABLE
# ─────────────────────────────────────────────
results = pd.DataFrame(trades)

print("=" * 120)
print("MANU CHAMPIONS LEAGUE VOLATILITY TRADE — CALENDAR SPREAD WITH PROFIT TARGET")
print("=" * 120)
print()
print("THESIS:")
print("  Manchester United (MANU) revenue is tied to Champions League depth. Deep runs")
print("  generate significant prize money ($100M+), match-day revenue, and sponsorship")
print("  bonuses. Low-vol stock with predictable catalyst timing.")
print()
print("STRATEGY:")
print(f"  Structure : Sell {SHORT_EXPIRY_DAYS}-day ATM call, buy {LONG_EXPIRY_DAYS}-day ATM call (calendar spread)")
print(f"  Entry     : Fixed calendar, {ENTRY_DAYS_BEFORE} days before CL Round of 16 first leg")
print(f"  Exit      : Close at {int(SPREAD_PROFIT_TARGET*100)}% profit target, or hold to short leg expiry ({SHORT_EXPIRY_DAYS} days)")
print()
print("EXECUTION SCENARIOS (from real data comparison, 2026-03-24):")
print(f"  Realistic : {REALISTIC_MODEL_ERR*100:.0f}% model error, {REALISTIC_SPREAD_PCT*100:.0f}% eff. spread (limit orders)")
print(f"  Worst case: {WORST_MODEL_ERR*100:.0f}% model error, {WORST_SPREAD_PCT*100:.0f}% eff. spread (market orders)")
print(f"  Both applied to entry AND exit (4 spread crossings total)")
print()
print("─" * 120)

# ── TRADE-BY-TRADE COMPARISON ──
print(f"\n  {'Season':<10} {'Entry':<12} {'Days':>5} {'Exit Reason':<22} {'Model%':>8} {'Realistic%':>11} {'Worst%':>8} {'M':>4} {'R':>4} {'W':>4}")
print(f"  {'─'*95}")
for _, row in results.iterrows():
    m = "W" if row["Win"] else "L"
    r = "W" if row["Win (real)"] else "L"
    w = "W" if row["Win (worst)"] else "L"
    print(f"  {row['Season']:<10} {str(row['Entry Date']):<12} {row['Holding Days']:>5} "
          f"{row['Exit Reason']:<22} {row['PnL %']:>+7.1f}% {row['PnL % (real)']:>+10.1f}% {row['PnL % (worst)']:>+7.1f}% "
          f"{m:>4} {r:>4} {w:>4}")
print()

# ─────────────────────────────────────────────
# 6. SUMMARY STATS — BOTH SCENARIOS
# ─────────────────────────────────────────────
n_trades = len(results)

# Model (mid-price)
wins_m = results["Win"].sum()
avg_pnl_m = results["PnL %"].mean()
std_pnl_m = results["PnL %"].std()
sharpe_m = (avg_pnl_m / std_pnl_m) * np.sqrt(n_trades) if std_pnl_m > 0 else np.nan

# Realistic
wins_r = results["Win (real)"].sum()
avg_pnl_r = results["PnL % (real)"].mean()
std_pnl_r = results["PnL % (real)"].std()
sharpe_r = (avg_pnl_r / std_pnl_r) * np.sqrt(n_trades) if std_pnl_r > 0 else np.nan

# Worst case
wins_w = results["Win (worst)"].sum()
avg_pnl_w = results["PnL % (worst)"].mean()
std_pnl_w = results["PnL % (worst)"].std()
sharpe_w = (avg_pnl_w / std_pnl_w) * np.sqrt(n_trades) if std_pnl_w > 0 else np.nan

print("─" * 90)
print(f"  {'Metric':<25} {'Model (mid)':>20} {'Realistic':>20} {'Worst Case':>20}")
print("─" * 90)
print(f"  {'Total Trades':<25} {n_trades:>20} {n_trades:>20} {n_trades:>20}")
print(f"  {'Wins':<25} {int(wins_m):>20} {int(wins_r):>20} {int(wins_w):>20}")
print(f"  {'Win Rate':<25} {wins_m/n_trades*100:>19.1f}% {wins_r/n_trades*100:>19.1f}% {wins_w/n_trades*100:>19.1f}%")
print(f"  {'Avg PnL %':<25} {avg_pnl_m:>+19.1f}% {avg_pnl_r:>+19.1f}% {avg_pnl_w:>+19.1f}%")
print(f"  {'Std PnL %':<25} {std_pnl_m:>19.1f}% {std_pnl_r:>19.1f}% {std_pnl_w:>19.1f}%")
print(f"  {'Sharpe-like Ratio':<25} {sharpe_m:>20.2f} {sharpe_r:>20.2f} {sharpe_w:>20.2f}")
print(f"  {'Best Trade':<25} {results['PnL %'].max():>+19.1f}% {results['PnL % (real)'].max():>+19.1f}% {results['PnL % (worst)'].max():>+19.1f}%")
print(f"  {'Worst Trade':<25} {results['PnL %'].min():>+19.1f}% {results['PnL % (real)'].min():>+19.1f}% {results['PnL % (worst)'].min():>+19.1f}%")
print("─" * 90)
print()

# ─────────────────────────────────────────────
# 7. PLOT
# ─────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9), sharex=True,
                                gridspec_kw={"height_ratios": [2, 1]})
fig.suptitle("MANU CL Calendar Spread Backtest — Playoff Volatility Trade", fontsize=14, fontweight="bold")

ax1.plot(df.index, df["Close"], color="#1f77b4", lw=1.2, label="MANU Close")
for _, row in results.iterrows():
    ed = pd.Timestamp(row["Entry Date"])
    xd = pd.Timestamp(row["Exit Date"])
    ep = row["Entry Price"]
    xp = row["Exit Price"]
    color = "#2ca02c" if row["Win"] else "#d62728"
    ax1.axvspan(ed, xd, alpha=0.08, color=color)
    ax1.scatter(ed, ep, marker="^", color="#2ca02c", s=100, zorder=5)
    ax1.scatter(xd, xp, marker="v", color=color, s=100, zorder=5)
    ax1.annotate(f"{row['Season']} ({row['PnL %']:+.0f}%)", xy=(ed, ep), xytext=(5, 8),
                 textcoords="offset points", fontsize=8, color=color)

ax1.set_ylabel("Price ($)")
ax1.grid(True, alpha=0.3)
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker="^", color="w", markerfacecolor="#2ca02c", markersize=9, label="Entry"),
    Line2D([0], [0], marker="v", color="w", markerfacecolor="#d62728", markersize=9, label="Exit"),
]
ax1.legend(handles=legend_elements + [ax1.get_lines()[0]], loc="upper left", fontsize=9)

ax2.plot(df.index, df["garch_vol"] * 100, color="#8c564b", lw=1.0, label="GJR-GARCH Vol (%)")
for _, row in results.iterrows():
    ax2.axvline(pd.Timestamp(row["Entry Date"]), color="#2ca02c", lw=1.0, linestyle=":")
ax2.set_ylabel("Volatility (%)")
ax2.set_xlabel("Date")
ax2.legend(loc="upper right", fontsize=8)
ax2.grid(True, alpha=0.3)
ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax2.xaxis.set_major_locator(mdates.YearLocator())

plt.tight_layout()
plt.savefig("manu_calendar_spread_backtest.png", dpi=150, bbox_inches="tight")
print("Plot saved to manu_calendar_spread_backtest.png")
plt.close()
