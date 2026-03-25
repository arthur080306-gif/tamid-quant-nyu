"""
Nike NBA Draft #1 Pick Endorsement Strategy Backtest
=====================================================
Buy 1-month ATM NKE call 1 week before draft night, exit 5 trading days after draft.
Call P&L proxy: max(0, price_exit - price_entry) - 0.03 * price_entry
"""

import warnings
warnings.filterwarnings("ignore")

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
from datetime import date, timedelta

# ---------------------------------------------------------------------------
# 1. DRAFT DATES & NIKE SIGNAL
# ---------------------------------------------------------------------------
# Approximate draft night as June 26 each year (always a weekday or nearby)
DRAFT_YEARS = list(range(2003, 2025))

draft_dates_raw = {yr: date(yr, 6, 26) for yr in DRAFT_YEARS}

# Nike signed #1 pick (simplified: 1 every year — Nike dominates ~85%+ of top picks)
nike_signed = {yr: 1 for yr in DRAFT_YEARS}

# ---------------------------------------------------------------------------
# 2. FETCH DATA
# ---------------------------------------------------------------------------
print("Fetching NKE and SPY data from Yahoo Finance...")
nke = yf.download("NKE", start="2002-01-01", end="2025-12-31", auto_adjust=True, progress=False)
spy = yf.download("SPY", start="2002-01-01", end="2025-12-31", auto_adjust=True, progress=False)

# Flatten multi-level columns if present
if isinstance(nke.columns, pd.MultiIndex):
    nke.columns = nke.columns.get_level_values(0)
if isinstance(spy.columns, pd.MultiIndex):
    spy.columns = spy.columns.get_level_values(0)

nke_close = nke["Close"].dropna()
spy_close = spy["Close"].dropna()

trading_days = nke_close.index  # DatetimeIndex

def nearest_trading_day(target_date, direction="forward"):
    """Find nearest trading day on or after (forward) / on or before (backward) target."""
    ts = pd.Timestamp(target_date)
    if direction == "forward":
        candidates = trading_days[trading_days >= ts]
    else:
        candidates = trading_days[trading_days <= ts]
    if len(candidates) == 0:
        return None
    return candidates[0] if direction == "forward" else candidates[-1]

def nth_trading_day_after(start_ts, n):
    """Return the nth trading day strictly after start_ts."""
    candidates = trading_days[trading_days > start_ts]
    if len(candidates) < n:
        return None
    return candidates[n - 1]

def nth_trading_day_before(start_ts, n):
    """Return the nth trading day strictly before start_ts."""
    candidates = trading_days[trading_days < start_ts]
    if len(candidates) < n:
        return None
    return candidates[-n]

# ---------------------------------------------------------------------------
# 3. BACKTEST LOOP
# ---------------------------------------------------------------------------
results = []

for yr in DRAFT_YEARS:
    draft_raw = draft_dates_raw[yr]
    draft_ts = nearest_trading_day(draft_raw, "forward")   # draft night ~ next trading day open

    # Entry: 5 trading days BEFORE draft night (≈ 1 calendar week before)
    entry_ts = nth_trading_day_before(draft_ts, 5)

    # Exit: 5 trading days AFTER draft night
    exit_ts = nth_trading_day_after(draft_ts, 5)

    if entry_ts is None or exit_ts is None:
        print(f"  Skipping {yr}: missing trading day data")
        continue
    if entry_ts not in nke_close.index or exit_ts not in nke_close.index:
        print(f"  Skipping {yr}: price data missing for entry/exit")
        continue

    price_entry = float(nke_close.loc[entry_ts])
    price_exit  = float(nke_close.loc[exit_ts])

    spy_entry = float(spy_close.loc[entry_ts])
    spy_exit  = float(spy_close.loc[exit_ts])

    nke_ret = (price_exit - price_entry) / price_entry
    spy_ret = (spy_exit  - spy_entry)  / spy_entry
    abnormal_ret = nke_ret - spy_ret

    # Call P&L proxy (long ATM call, premium = 3% of stock price)
    call_pnl = (max(0, price_exit - price_entry) - 0.03 * price_entry) / price_entry

    signed = nike_signed[yr]
    win = 1 if abnormal_ret > 0 else 0

    results.append({
        "Year":          yr,
        "Draft Date":    draft_ts.date(),
        "Entry Date":    entry_ts.date(),
        "Exit Date":     exit_ts.date(),
        "Nike Signed":   signed,
        "NKE Entry ($)": round(price_entry, 2),
        "NKE Exit ($)":  round(price_exit,  2),
        "NKE Ret (%)":   round(nke_ret * 100, 2),
        "SPY Ret (%)":   round(spy_ret * 100, 2),
        "Abnormal Ret (%)": round(abnormal_ret * 100, 2),
        "Call P&L (%)":  round(call_pnl * 100, 2),
        "Win":           win,
    })

df = pd.DataFrame(results)

# ---------------------------------------------------------------------------
# 4. PRINT TRADE TABLE
# ---------------------------------------------------------------------------
print("\n" + "=" * 110)
print("NIKE NBA DRAFT #1 PICK ENDORSEMENT STRATEGY — TRADE LOG")
print("=" * 110)

display_cols = [
    "Year", "Entry Date", "Exit Date",
    "NKE Ret (%)", "SPY Ret (%)", "Abnormal Ret (%)", "Call P&L (%)", "Win"
]
print(df[display_cols].to_string(index=False))

# ---------------------------------------------------------------------------
# 5. OVERALL STATS
# ---------------------------------------------------------------------------
n_trades    = len(df)
win_rate    = df["Win"].mean()
avg_abn     = df["Abnormal Ret (%)"].mean()
std_abn     = df["Abnormal Ret (%)"].std()
avg_call    = df["Call P&L (%)"].mean()
sharpe_like = avg_abn / std_abn if std_abn > 0 else np.nan
total_call  = df["Call P&L (%)"].sum()

print("\n" + "=" * 50)
print("OVERALL STRATEGY STATISTICS")
print("=" * 50)
print(f"  Total trades          : {n_trades}")
print(f"  Win rate (abn ret > 0): {win_rate:.1%}")
print(f"  Avg abnormal return   : {avg_abn:+.2f}%")
print(f"  Std abnormal return   : {std_abn:.2f}%")
print(f"  Avg call P&L          : {avg_call:+.2f}%")
print(f"  Total call P&L        : {total_call:+.2f}%")
print(f"  Sharpe-like ratio     : {sharpe_like:.3f}")

# ---------------------------------------------------------------------------
# 6. OLS REGRESSION (test if mean abnormal return ≠ 0)
# ---------------------------------------------------------------------------
y = df["Abnormal Ret (%)"].values
X = np.ones(len(y))
result = stats.ttest_1samp(y, popmean=0)
t_stat = result.statistic
p_val  = result.pvalue

# Also via OLS manually for completeness
beta     = np.mean(y)
se_beta  = stats.sem(y)
t_ols    = beta / se_beta

print("\n" + "=" * 50)
print("OLS REGRESSION: Abnormal Return ~ Constant")
print("=" * 50)
print(f"  Intercept (mean)      : {beta:+.4f}%")
print(f"  Std Error             : {se_beta:.4f}%")
print(f"  t-statistic           : {t_ols:.3f}")
print(f"  p-value (two-tailed)  : {p_val:.4f}")
significance = "*** p<0.01" if p_val < 0.01 else ("** p<0.05" if p_val < 0.05 else ("* p<0.10" if p_val < 0.10 else "not significant"))
print(f"  Significance          : {significance}")

# ---------------------------------------------------------------------------
# 7. PLOTS
# ---------------------------------------------------------------------------
fig = plt.figure(figsize=(14, 10))
fig.suptitle("Nike NBA Draft #1 Pick Endorsement Strategy\n(Entry: -5 trading days, Exit: +5 trading days)",
             fontsize=14, fontweight="bold", y=0.98)

gs = gridspec.GridSpec(2, 1, hspace=0.45)

years = df["Year"].values
abn   = df["Abnormal Ret (%)"].values
cum   = np.cumsum(abn)

# --- Panel 1: Bar chart of annual abnormal returns ---
ax1 = fig.add_subplot(gs[0])
colors = ["#2ecc71" if v > 0 else "#e74c3c" for v in abn]
bars = ax1.bar(years, abn, color=colors, edgecolor="white", linewidth=0.5, width=0.7)
ax1.axhline(0, color="black", linewidth=0.8)
ax1.axhline(avg_abn, color="#3498db", linewidth=1.5, linestyle="--", label=f"Mean = {avg_abn:+.2f}%")
ax1.set_xlabel("Draft Year", fontsize=11)
ax1.set_ylabel("Abnormal Return (%)", fontsize=11)
ax1.set_title(f"Annual Abnormal Return (NKE − SPY) | Win Rate: {win_rate:.0%} | t={t_ols:.2f}, p={p_val:.3f}",
              fontsize=11)
ax1.set_xticks(years)
ax1.set_xticklabels(years, rotation=45, ha="right", fontsize=9)
ax1.legend(fontsize=10)
ax1.grid(axis="y", alpha=0.3)

# Value labels on bars
for bar, val in zip(bars, abn):
    va = "bottom" if val >= 0 else "top"
    offset = 0.1 if val >= 0 else -0.1
    ax1.text(bar.get_x() + bar.get_width() / 2, val + offset,
             f"{val:+.1f}", ha="center", va=va, fontsize=7.5, fontweight="bold")

# --- Panel 2: Cumulative abnormal return ---
ax2 = fig.add_subplot(gs[1])
ax2.plot(years, cum, marker="o", color="#2c3e50", linewidth=2, markersize=6, label="Cumulative Abnormal Return")
ax2.fill_between(years, 0, cum, where=(np.array(cum) >= 0), alpha=0.15, color="#2ecc71")
ax2.fill_between(years, 0, cum, where=(np.array(cum) < 0),  alpha=0.15, color="#e74c3c")
ax2.axhline(0, color="black", linewidth=0.8)
ax2.set_xlabel("Draft Year", fontsize=11)
ax2.set_ylabel("Cumulative Abnormal Return (%)", fontsize=11)
ax2.set_title(f"Cumulative Abnormal Return | Total: {cum[-1]:+.1f}%", fontsize=11)
ax2.set_xticks(years)
ax2.set_xticklabels(years, rotation=45, ha="right", fontsize=9)
ax2.legend(fontsize=10)
ax2.grid(alpha=0.3)

# Annotate final value
ax2.annotate(f"{cum[-1]:+.1f}%", xy=(years[-1], cum[-1]),
             xytext=(years[-1] - 1.5, cum[-1] + 1.5),
             fontsize=9, fontweight="bold", color="#2c3e50",
             arrowprops=dict(arrowstyle="->", color="#2c3e50", lw=1))

plt.savefig("nike_draft_backtest.png", dpi=150, bbox_inches="tight")
print("\nPlot saved to nike_draft_backtest.png")
print("Done.")
