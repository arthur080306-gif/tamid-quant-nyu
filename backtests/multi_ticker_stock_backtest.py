"""
Multi-Ticker Sports Catalyst — Stock-Only Backtest with Profit Target & Stop-Loss
Rotates capital across MANU, BVB.DE, JUVE.MI, and MSGS around predictable
sporting catalysts (CL/EL knockout, NBA playoffs).

Buy stock 28 days before catalyst, exit at profit target, stop-loss, or 30-day hold.
Tests whether capital rotation across multiple tickers improves Sharpe.
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
PROFIT_TARGET = 0.10    # 10% take-profit
STOP_LOSS = -0.05       # 5% stop-loss

# All events by ticker
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
    "MSGS": {
        "NBA 2019": pd.Timestamp("2019-04-13"),
        "NBA 2021": pd.Timestamp("2021-05-22"),
        "NBA 2022": pd.Timestamp("2022-04-16"),
        "NBA 2023": pd.Timestamp("2023-04-15"),
        "NBA 2024": pd.Timestamp("2024-04-20"),
    },
}

# ─────────────────────────────────────────────
# FETCH ALL DATA
# ─────────────────────────────────────────────
print("Fetching price data...")
price_data = {}
for ticker in EVENTS.keys():
    raw = yf.download(ticker, start="2012-01-01", end="2024-12-31", auto_adjust=True, progress=False)
    raw.columns = raw.columns.get_level_values(0) if isinstance(raw.columns, pd.MultiIndex) else raw.columns
    df = raw[["Close"]].dropna()
    df.index = pd.to_datetime(df.index)
    price_data[ticker] = df
    print(f"  {ticker}: {len(df)} days ({df.index[0].date()} → {df.index[-1].date()})")
print()

# ─────────────────────────────────────────────
# BACKTEST — WITH AND WITHOUT PROFIT TARGET / STOP-LOSS
# ─────────────────────────────────────────────
def run_backtest(use_exits=True):
    """Run backtest. use_exits=True applies profit target & stop-loss."""
    trades = []
    for ticker, events in EVENTS.items():
        df = price_data[ticker]
        for event, match_date in events.items():
            target_entry = match_date - timedelta(days=ENTRY_DAYS_BEFORE)
            valid_days = df[df.index >= target_entry]
            if valid_days.empty:
                continue

            entry_date = valid_days.index[0]
            entry_price = float(df.loc[entry_date, "Close"])

            # Hold window
            target_exit = entry_date + timedelta(days=HOLD_DAYS)
            hold_window = df[(df.index > entry_date) & (df.index <= target_exit)]
            if hold_window.empty:
                continue

            exit_date = None
            exit_reason = "Hold Expiry"

            if use_exits:
                for dt in hold_window.index:
                    price = float(df.loc[dt, "Close"])
                    ret = (price - entry_price) / entry_price
                    if ret >= PROFIT_TARGET:
                        exit_date = dt
                        exit_reason = f"Profit Target ({ret*100:+.1f}%)"
                        break
                    if ret <= STOP_LOSS:
                        exit_date = dt
                        exit_reason = f"Stop-Loss ({ret*100:+.1f}%)"
                        break

            if exit_date is None:
                exit_date = hold_window.index[-1]

            exit_price = float(df.loc[exit_date, "Close"])
            days_held = (exit_date - entry_date).days
            pnl_pct = (exit_price - entry_price) / entry_price * 100

            trades.append({
                "Ticker": ticker,
                "Event": event,
                "Entry Date": entry_date.date(),
                "Entry Price": round(entry_price, 2),
                "Exit Date": exit_date.date(),
                "Exit Price": round(exit_price, 2),
                "Days Held": days_held,
                "PnL %": round(pnl_pct, 1),
                "Exit Reason": exit_reason,
                "Win": pnl_pct > 0,
            })

    return pd.DataFrame(trades)

results_exit = run_backtest(use_exits=True)
results_hold = run_backtest(use_exits=False)

# ─────────────────────────────────────────────
# ANNUALIZED METRICS
# ─────────────────────────────────────────────
def compute_annual_metrics(results, label):
    """Compute annualized return and Sharpe from trade results."""
    n = len(results)
    if n == 0:
        return {}

    # Sort by entry date for compounding
    r = results.sort_values("Entry Date")
    returns = r["PnL %"].values / 100

    # Compounded total return
    total = np.prod(1 + returns)
    first_entry = pd.Timestamp(r["Entry Date"].iloc[0])
    last_exit = pd.Timestamp(r["Exit Date"].iloc[-1])
    years = (last_exit - first_entry).days / 365.25
    cagr = (total ** (1 / years) - 1) * 100 if years > 0 else 0

    # Avg days held
    avg_days = r["Days Held"].mean()

    # Annualize per-trade stats
    trades_per_year = n / years if years > 0 else 0
    avg_pnl = r["PnL %"].mean()
    std_pnl = r["PnL %"].std()

    # Annual return = avg_pnl * trades_per_year (simple approx)
    annual_ret = avg_pnl * trades_per_year
    annual_vol = std_pnl * np.sqrt(trades_per_year)
    sharpe = (annual_ret - 4.0) / annual_vol if annual_vol > 0 else np.nan  # rf = 4%

    return {
        "label": label,
        "n": n,
        "wins": int(r["Win"].sum()),
        "win_rate": r["Win"].mean() * 100,
        "avg_pnl": avg_pnl,
        "std_pnl": std_pnl,
        "median_pnl": r["PnL %"].median(),
        "avg_days": avg_days,
        "trades_per_year": trades_per_year,
        "total_return": (total - 1) * 100,
        "cagr": cagr,
        "annual_ret": annual_ret,
        "annual_vol": annual_vol,
        "sharpe": sharpe,
        "years": years,
    }

m_exit = compute_annual_metrics(results_exit, "With Exits")
m_hold = compute_annual_metrics(results_hold, "Hold Only")

# ─────────────────────────────────────────────
# OUTPUT
# ─────────────────────────────────────────────
print("=" * 120)
print("MULTI-TICKER SPORTS CATALYST — STOCK-ONLY BACKTEST")
print("=" * 120)
print()
print("THESIS:")
print("  Publicly traded sports teams see revenue boosts from deep playoff/tournament")
print("  runs. By rotating capital across multiple tickers and events, we increase")
print("  deployment frequency and improve risk-adjusted returns.")
print()
print("STRATEGY:")
print(f"  Buy stock {ENTRY_DAYS_BEFORE} days before catalyst, hold up to {HOLD_DAYS} days")
print(f"  Profit target: {PROFIT_TARGET*100:.0f}% | Stop-loss: {STOP_LOSS*100:.0f}%")
print("  Tickers: MANU (CL+EL), BVB.DE (CL), JUVE.MI (CL), MSGS (NBA)")
print()
print("─" * 120)

# Trade table (with exits)
print(f"\n  {'Ticker':<8} {'Event':<12} {'Entry':<12} {'Exit':<12} {'Days':>5} "
      f"{'Entry$':>8} {'Exit$':>8} {'PnL%':>8} {'Exit Reason':<28} {'Result':>7}")
print(f"  {'─'*115}")
for _, row in results_exit.sort_values("Entry Date").iterrows():
    w = "WIN" if row["Win"] else "LOSS"
    print(f"  {row['Ticker']:<8} {row['Event']:<12} {str(row['Entry Date']):<12} "
          f"{str(row['Exit Date']):<12} {row['Days Held']:>5} "
          f"${row['Entry Price']:>7.2f} ${row['Exit Price']:>7.2f} "
          f"{row['PnL %']:>+7.1f}% {row['Exit Reason']:<28} {w:>7}")
print()

# Per-ticker stats
print("─" * 100)
print("PER-TICKER BREAKDOWN (with profit target & stop-loss)")
print("─" * 100)
for ticker in EVENTS.keys():
    t = results_exit[results_exit["Ticker"] == ticker]
    n = len(t)
    if n == 0:
        continue
    wins = t["Win"].sum()
    avg = t["PnL %"].mean()
    avg_days = t["Days Held"].mean()
    print(f"  {ticker:<8} | {n:>2} trades | WR: {wins/n*100:>5.0f}% | "
          f"Avg PnL: {avg:>+5.1f}% | Avg hold: {avg_days:.0f} days")
print()

# Summary comparison
print("=" * 90)
print(f"  {'Metric':<30} {'With Exits':>25} {'Hold Only (no exits)':>25}")
print("=" * 90)

for key, fmt in [
    ("n", "{}"),
    ("wins", "{}"),
    ("win_rate", "{:.0f}%"),
    ("avg_pnl", "{:+.1f}%"),
    ("std_pnl", "{:.1f}%"),
    ("median_pnl", "{:+.1f}%"),
    ("avg_days", "{:.0f} days"),
    ("trades_per_year", "{:.1f}"),
    ("total_return", "{:+.1f}%"),
    ("cagr", "{:+.1f}%"),
    ("annual_ret", "{:+.1f}%"),
    ("annual_vol", "{:.1f}%"),
    ("sharpe", "{:.2f}"),
]:
    label = key.replace("_", " ").title()
    v1 = fmt.format(m_exit[key])
    v2 = fmt.format(m_hold[key])
    print(f"  {label:<30} {v1:>25} {v2:>25}")

print("=" * 90)
print()

# Benchmark
print("─" * 60)
print("CONTEXT")
print("─" * 60)
print(f"  Period         : {m_exit['years']:.1f} years")
print(f"  Trades/year    : {m_exit['trades_per_year']:.1f} (with exits) vs {m_hold['trades_per_year']:.1f} (hold)")
print(f"  Avg hold       : {m_exit['avg_days']:.0f} days (with exits) vs {m_hold['avg_days']:.0f} days (hold)")
print(f"  Capital util.  : {m_exit['avg_days'] * m_exit['trades_per_year'] / 365 * 100:.0f}% of year (with exits)")
print()
print("  MANU-only (prev backtest): Sharpe 0.57, CAGR 4.3%")
print("─" * 60)
print()

# ─────────────────────────────────────────────
# PLOT
# ─────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle("Multi-Ticker Sports Catalyst — Stock-Only Backtest", fontsize=14, fontweight="bold")

ticker_colors = {"MANU": "#DA291C", "BVB.DE": "#FDE100", "JUVE.MI": "#000000", "MSGS": "#006BB6"}

for idx, (ticker, events) in enumerate(EVENTS.items()):
    ax = axes[idx // 2][idx % 2]
    df = price_data[ticker]
    ax.plot(df.index, df["Close"], color="#1f77b4", lw=0.8)

    ticker_trades = results_exit[results_exit["Ticker"] == ticker].sort_values("Entry Date")
    for _, row in ticker_trades.iterrows():
        ed = pd.Timestamp(row["Entry Date"])
        xd = pd.Timestamp(row["Exit Date"])
        color = "#2ca02c" if row["Win"] else "#d62728"
        ax.axvspan(ed, xd, alpha=0.12, color=color)
        ax.scatter(ed, row["Entry Price"], marker="^", color="#2ca02c", s=50, zorder=5)
        ax.scatter(xd, row["Exit Price"], marker="v", color=color, s=50, zorder=5)

    n = len(ticker_trades)
    wr = ticker_trades["Win"].mean() * 100 if n > 0 else 0
    avg = ticker_trades["PnL %"].mean() if n > 0 else 0
    ax.set_title(f"{ticker} — {n} trades, {wr:.0f}% WR, {avg:+.1f}% avg", fontsize=10)
    ax.set_ylabel("Price")
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator(2))

plt.tight_layout()
plt.savefig("multi_ticker_stock_backtest.png", dpi=150, bbox_inches="tight")
print("Plot saved to multi_ticker_stock_backtest.png")
plt.close()

# ─────────────────────────────────────────────
# CUMULATIVE EQUITY CURVE
# ─────────────────────────────────────────────
fig2, ax2 = plt.subplots(figsize=(14, 6))
fig2.suptitle("Cumulative Equity Curve — Multi-Ticker Sports Catalyst", fontsize=14, fontweight="bold")

sorted_trades = results_exit.sort_values("Entry Date")
cumulative = [1.0]
dates = [pd.Timestamp(sorted_trades["Entry Date"].iloc[0])]
for _, row in sorted_trades.iterrows():
    ret = row["PnL %"] / 100
    cumulative.append(cumulative[-1] * (1 + ret))
    dates.append(pd.Timestamp(row["Exit Date"]))

ax2.plot(dates, cumulative, color="#1f77b4", lw=2.0, marker="o", markersize=4)
ax2.axhline(1.0, color="gray", lw=0.8, linestyle="--")
ax2.set_ylabel("Cumulative Return (1.0 = start)")
ax2.set_xlabel("Date")
ax2.grid(True, alpha=0.3)
ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

# Annotate final
ax2.annotate(f"{(cumulative[-1]-1)*100:+.0f}%", xy=(dates[-1], cumulative[-1]),
             xytext=(10, 0), textcoords="offset points", fontsize=11, fontweight="bold")

plt.tight_layout()
plt.savefig("multi_ticker_equity_curve.png", dpi=150, bbox_inches="tight")
print("Plot saved to multi_ticker_equity_curve.png")
plt.close()
