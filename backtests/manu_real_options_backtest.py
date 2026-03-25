"""
MANU Calendar Spread — REAL OPTIONS DATA
Uses MarketData.app API to pull actual bid/ask quotes.
Tests current MANU options liquidity with the same calendar spread structure.
Free tier only covers 1 year of history (March 2025+), so historical CL/EL
trades can't be validated. Instead we test current market conditions.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import requests
import yfinance as yf
import time
from datetime import timedelta

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
TOKEN = "MHp4RkpObGVTQlNCcmQ5OGVXZHNOdmkxdUpaZEhFcWFWZE5vbWFtRDZkYz0"
HEADERS = {"Authorization": f"Token {TOKEN}"}
BASE_URL = "https://api.marketdata.app/v1"
SPREAD_PROFIT_TARGET = 0.20

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def get_chain(ticker, date, expiration, side="call"):
    r = requests.get(f"{BASE_URL}/options/chain/{ticker}/",
        params={"date": date, "expiration": expiration, "side": side},
        headers=HEADERS, timeout=15)
    if r.status_code in (200, 203):
        return r.json()
    return None

def get_expirations(ticker, date):
    r = requests.get(f"{BASE_URL}/options/expirations/{ticker}/",
        params={"date": date}, headers=HEADERS, timeout=15)
    if r.status_code in (200, 203):
        return r.json().get("expirations", [])
    return []

def find_strike_option(chain_data, target_strike):
    """Find option closest to target strike."""
    if not chain_data or chain_data.get("s") != "ok":
        return None
    strikes = chain_data["strike"]
    idx = min(range(len(strikes)), key=lambda i: abs(strikes[i] - target_strike))
    return {
        "symbol": chain_data["optionSymbol"][idx],
        "strike": strikes[idx],
        "bid": chain_data["bid"][idx] or 0,
        "ask": chain_data["ask"][idx] or 0,
        "mid": chain_data["mid"][idx] or 0,
        "last": chain_data["last"][idx] or 0,
        "iv": chain_data["iv"][idx] or 0,
        "volume": chain_data["volume"][idx] or 0,
        "openInterest": chain_data["openInterest"][idx] or 0,
        "dte": chain_data["dte"][idx] or 0,
    }

# ─────────────────────────────────────────────
# TEST MULTIPLE ENTRY DATES
# ─────────────────────────────────────────────
# We'll test the calendar spread at multiple points in accessible history
# to see how bid/ask spreads affect returns over time

print("=" * 110)
print("MANU CALENDAR SPREAD — REAL OPTIONS DATA")
print("=" * 110)
print()
print("NOTE: MarketData.app free tier only covers March 2025 – present.")
print("      Historical CL/EL trades (2013–2023) cannot be validated with real data.")
print("      Instead, we test the strategy on accessible dates to measure real execution costs.")
print()
print("STRATEGY:")
print("  Sell short-dated ATM call (~30 DTE), buy longer-dated ATM call (~90 DTE)")
print("  Exit: 20% profit target or short leg expiry")
print()
print("─" * 110)

# Test dates spread across the accessible window
test_entries = [
    "2025-04-01",
    "2025-07-01",
    "2025-10-01",
    "2026-01-05",
]

all_trades = []

for entry_date in test_entries:
    print(f"\n{'─'*60}")
    print(f"  ENTRY: {entry_date}")
    print(f"{'─'*60}")

    # Get stock price
    raw = yf.download("MANU", start=entry_date,
                      end=(pd.Timestamp(entry_date) + timedelta(days=5)).strftime("%Y-%m-%d"),
                      auto_adjust=True, progress=False)
    if raw.empty:
        print(f"  No stock data, skipping")
        continue
    raw.columns = raw.columns.get_level_values(0) if isinstance(raw.columns, pd.MultiIndex) else raw.columns
    entry_price = float(raw.iloc[0]["Close"])
    actual_entry_date = raw.index[0].strftime("%Y-%m-%d")
    print(f"  Spot: ${entry_price:.2f} (on {actual_entry_date})")

    # Get expirations
    time.sleep(1)
    exps = get_expirations("MANU", actual_entry_date)
    if not exps:
        print(f"  No expirations available, skipping")
        continue

    entry_dt = pd.Timestamp(actual_entry_date)
    short_exp = None
    long_exp = None
    for e in exps:
        dte = (pd.Timestamp(e) - entry_dt).days
        if 20 <= dte <= 45 and short_exp is None:
            short_exp = e
        if 75 <= dte <= 150 and long_exp is None:
            long_exp = e

    if not short_exp or not long_exp:
        print(f"  No suitable expirations (need ~30 and ~90 DTE)")
        print(f"  Available: {exps}")
        continue

    short_dte = (pd.Timestamp(short_exp) - entry_dt).days
    long_dte = (pd.Timestamp(long_exp) - entry_dt).days
    print(f"  Short: {short_exp} ({short_dte} DTE) | Long: {long_exp} ({long_dte} DTE)")

    # Get entry chains
    time.sleep(1)
    s_chain = get_chain("MANU", actual_entry_date, short_exp)
    time.sleep(1)
    l_chain = get_chain("MANU", actual_entry_date, long_exp)

    if not s_chain or not l_chain:
        print(f"  Could not fetch options chains")
        continue

    s_entry = find_strike_option(s_chain, entry_price)
    l_entry = find_strike_option(l_chain, entry_price)

    if not s_entry or not l_entry:
        print(f"  Could not find ATM options")
        continue

    strike = s_entry["strike"]

    # Entry costs
    sell_short = s_entry["bid"]     # sell at bid
    buy_long = l_entry["ask"]       # buy at ask
    net_debit_real = buy_long - sell_short
    net_debit_mid = l_entry["mid"] - s_entry["mid"]

    s_spread_pct = ((s_entry["ask"] - s_entry["bid"]) / s_entry["mid"] * 100) if s_entry["mid"] > 0 else 999
    l_spread_pct = ((l_entry["ask"] - l_entry["bid"]) / l_entry["mid"] * 100) if l_entry["mid"] > 0 else 999

    print(f"  SHORT ATM (K={strike}): bid=${s_entry['bid']:.2f} ask=${s_entry['ask']:.2f} "
          f"spread={s_spread_pct:.0f}% vol={s_entry['volume']} OI={s_entry['openInterest']}")
    print(f"  LONG  ATM (K={strike}): bid=${l_entry['bid']:.2f} ask=${l_entry['ask']:.2f} "
          f"spread={l_spread_pct:.0f}% vol={l_entry['volume']} OI={l_entry['openInterest']}")
    print(f"  Net debit (real): ${net_debit_real:.2f} | (mid): ${net_debit_mid:.2f} "
          f"| slippage: ${net_debit_real - net_debit_mid:.2f}")

    if net_debit_real <= 0:
        print(f"  Non-positive debit, skipping")
        continue

    # Track to short expiry — sample every 5 trading days to conserve API calls
    exit_target = pd.Timestamp(short_exp)
    stock_data = yf.download("MANU", start=actual_entry_date,
                             end=(exit_target + timedelta(days=3)).strftime("%Y-%m-%d"),
                             auto_adjust=True, progress=False)
    stock_data.columns = stock_data.columns.get_level_values(0) if isinstance(stock_data.columns, pd.MultiIndex) else stock_data.columns

    trading_days = stock_data.index.tolist()
    # Check every 5th day + last day
    check_days = trading_days[::5]
    if trading_days[-1] not in check_days:
        check_days.append(trading_days[-1])

    best_pnl_real = -999
    best_pnl_mid = -999
    final_pnl_real = None
    final_pnl_mid = None
    profit_target_hit = False
    exit_info = None

    for check_date in check_days:
        date_str = check_date.strftime("%Y-%m-%d")
        spot = float(stock_data.loc[check_date, "Close"])

        time.sleep(1)
        sc = get_chain("MANU", date_str, short_exp)
        time.sleep(1)
        lc = get_chain("MANU", date_str, long_exp)

        if not sc or not lc:
            continue

        s_now = find_strike_option(sc, strike)
        l_now = find_strike_option(lc, strike)
        if not s_now or not l_now:
            continue

        # Close: buy back short at ask, sell long at bid
        pnl_real = (sell_short - s_now["ask"]) + (l_now["bid"] - buy_long)
        pnl_pct_real = (pnl_real / net_debit_real) * 100

        pnl_mid = (l_now["mid"] - s_now["mid"]) - net_debit_mid
        pnl_pct_mid = (pnl_mid / net_debit_mid) * 100 if net_debit_mid > 0 else 0

        days_held = (check_date - pd.Timestamp(actual_entry_date)).days

        print(f"    {date_str} | ${spot:.2f} | real: {pnl_pct_real:+.1f}% | mid: {pnl_pct_mid:+.1f}%", end="")

        if pnl_pct_real > best_pnl_real:
            best_pnl_real = pnl_pct_real
        if pnl_pct_mid > best_pnl_mid:
            best_pnl_mid = pnl_pct_mid

        final_pnl_real = pnl_pct_real
        final_pnl_mid = pnl_pct_mid

        if pnl_pct_real >= SPREAD_PROFIT_TARGET * 100 and not profit_target_hit:
            profit_target_hit = True
            exit_info = {"date": date_str, "pnl": pnl_pct_real, "days": days_held}
            print(f" <<< TARGET HIT", end="")

        print()

    all_trades.append({
        "Entry": actual_entry_date,
        "Spot": round(entry_price, 2),
        "Strike": strike,
        "Debit (real)": round(net_debit_real, 2),
        "Debit (mid)": round(net_debit_mid, 2),
        "Short Spread %": round(s_spread_pct, 0),
        "Long Spread %": round(l_spread_pct, 0),
        "Final PnL % (real)": round(final_pnl_real, 1) if final_pnl_real is not None else None,
        "Final PnL % (mid)": round(final_pnl_mid, 1) if final_pnl_mid is not None else None,
        "Best PnL % (real)": round(best_pnl_real, 1),
        "Best PnL % (mid)": round(best_pnl_mid, 1),
        "Target Hit": profit_target_hit,
        "Target Date": exit_info["date"] if exit_info else None,
    })

# ─────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────
results = pd.DataFrame(all_trades)

print(f"\n{'='*110}")
print("SUMMARY — MANU CALENDAR SPREAD WITH REAL OPTIONS DATA")
print(f"{'='*110}")
print()

if not results.empty:
    for _, row in results.iterrows():
        print(f"  Entry {row['Entry']} | Spot ${row['Spot']} | K={row['Strike']}")
        print(f"    Debit: real=${row['Debit (real)']:.2f}, mid=${row['Debit (mid)']:.2f}")
        print(f"    Bid/Ask spreads: short={row['Short Spread %']:.0f}%, long={row['Long Spread %']:.0f}%")
        print(f"    Final PnL: real={row['Final PnL % (real)']}%, mid={row['Final PnL % (mid)']}%")
        print(f"    Best PnL:  real={row['Best PnL % (real)']}%, mid={row['Best PnL % (mid)']}%")
        print(f"    Target hit: {'YES — ' + row['Target Date'] if row['Target Hit'] else 'NO'}")
        print()

    n = len(results)
    hits = results["Target Hit"].sum()
    avg_real = results["Final PnL % (real)"].mean()
    avg_mid = results["Final PnL % (mid)"].mean()
    avg_short_spread = results["Short Spread %"].mean()
    avg_long_spread = results["Long Spread %"].mean()

    print(f"  {'─'*60}")
    print(f"  Trades tested       : {n}")
    print(f"  Profit targets hit  : {int(hits)}/{n}")
    print(f"  Avg final PnL (real): {avg_real:.1f}%")
    print(f"  Avg final PnL (mid) : {avg_mid:.1f}%")
    print(f"  Avg slippage drag   : {avg_real - avg_mid:.1f}%")
    print(f"  Avg short spread    : {avg_short_spread:.0f}%")
    print(f"  Avg long spread     : {avg_long_spread:.0f}%")
    print(f"  {'─'*60}")

print()
print("COMPARISON TO MSGS:")
print("  MSGS 2025 real trade: -41.8% (real) vs +4.0% (mid)")
print("  MSGS short spread: 70%, long spread: 15%")
print(f"{'='*110}")
