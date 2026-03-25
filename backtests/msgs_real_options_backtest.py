"""
MSGS Calendar Spread Backtest — REAL OPTIONS DATA
Uses MarketData.app API to pull actual historical bid/ask quotes.
Tests the 2025 NBA Playoff trade with real options prices.
"""

import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import requests
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import time

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
TOKEN = "MHp4RkpObGVTQlNCcmQ5OGVXZHNOdmkxdUpaZEhFcWFWZE5vbWFtRDZkYz0"
HEADERS = {"Authorization": f"Token {TOKEN}"}
BASE_URL = "https://api.marketdata.app/v1"
TICKER = "MSGS"
SPREAD_PROFIT_TARGET = 0.20

# 2025 NBA Playoffs start April 19, 2025
# Entry: 28 days before = ~March 22, 2025
PLAYOFF_START = pd.Timestamp("2025-04-19")
ENTRY_TARGET = PLAYOFF_START - timedelta(days=28)  # March 22

# ─────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────
def get_options_chain(ticker, date, expiration, side="call"):
    """Get options chain for a specific date and expiration."""
    r = requests.get(f"{BASE_URL}/options/chain/{ticker}/",
        params={"date": date, "expiration": expiration, "side": side},
        headers=HEADERS, timeout=15)
    if r.status_code in (200, 203):
        return r.json()
    else:
        print(f"  API error ({r.status_code}): {r.text[:200]}")
        return None

def get_expirations(ticker, date):
    """Get available expirations for a ticker on a given date."""
    r = requests.get(f"{BASE_URL}/options/expirations/{ticker}/",
        params={"date": date},
        headers=HEADERS, timeout=15)
    if r.status_code in (200, 203):
        return r.json().get("expirations", [])
    return []

def get_stock_price(ticker, date):
    """Get stock closing price on a date."""
    r = requests.get(f"{BASE_URL}/v1/stocks/candles/daily/{ticker}/",
        params={"from": date, "to": date},
        headers=HEADERS, timeout=15)
    if r.status_code in (200, 203):
        data = r.json()
        if data.get("s") == "ok" and data.get("c"):
            return data["c"][0]
    return None

def find_atm_option(chain_data, spot_price):
    """Find the ATM option from chain data."""
    if not chain_data or chain_data.get("s") != "ok":
        return None
    strikes = chain_data["strike"]
    idx = min(range(len(strikes)), key=lambda i: abs(strikes[i] - spot_price))
    return {
        "symbol": chain_data["optionSymbol"][idx],
        "strike": strikes[idx],
        "bid": chain_data["bid"][idx] or 0,
        "ask": chain_data["ask"][idx] or 0,
        "mid": chain_data["mid"][idx] or 0,
        "last": chain_data["last"][idx] or 0,
        "iv": chain_data["iv"][idx] or 0,
        "delta": chain_data["delta"][idx] or 0,
        "theta": chain_data["theta"][idx] or 0,
        "vega": chain_data["vega"][idx] or 0,
        "volume": chain_data["volume"][idx] or 0,
        "openInterest": chain_data["openInterest"][idx] or 0,
        "dte": chain_data["dte"][idx] or 0,
    }

# ─────────────────────────────────────────────
# PRINT HEADER
# ─────────────────────────────────────────────
print("=" * 100)
print("MSGS CALENDAR SPREAD — REAL OPTIONS DATA BACKTEST (2025)")
print("=" * 100)
print()
print("STRATEGY:")
print("  Sell short-dated ATM call (~30 DTE), buy longer-dated ATM call (~90 DTE)")
print("  Entry: 28 days before NBA playoff start (March 22, 2025)")
print("  Exit: 20% profit target or short leg expiry")
print("  Data: REAL bid/ask quotes from MarketData.app")
print()
print("─" * 100)

# ─────────────────────────────────────────────
# 1. GET SPOT PRICE AND EXPIRATIONS AT ENTRY
# ─────────────────────────────────────────────
entry_date = "2025-03-24"  # nearest trading day to March 22

print(f"\n[ENTRY: {entry_date}]")
print(f"  Fetching MSGS stock price and options...")

# Get stock price from yfinance (more reliable for spot)
raw = yf.download(TICKER, start="2025-03-20", end="2025-03-28", auto_adjust=True, progress=False)
raw.columns = raw.columns.get_level_values(0) if isinstance(raw.columns, pd.MultiIndex) else raw.columns
entry_price = float(raw.loc[raw.index >= entry_date].iloc[0]["Close"])
print(f"  MSGS spot price: ${entry_price:.2f}")

# Get available expirations
expirations = get_expirations(TICKER, entry_date)
print(f"  Available expirations: {expirations}")

# Find short-dated (~30 DTE) and long-dated (~90 DTE) expirations
entry_dt = pd.Timestamp(entry_date)
short_exp = None
long_exp = None
for exp in expirations:
    exp_dt = pd.Timestamp(exp)
    dte = (exp_dt - entry_dt).days
    if 20 <= dte <= 40 and short_exp is None:
        short_exp = exp
    if 80 <= dte <= 120 and long_exp is None:
        long_exp = exp
    if 140 <= dte <= 180 and long_exp is None:
        long_exp = exp  # fallback to longer if no ~90 DTE

print(f"  Short leg expiration: {short_exp} ({(pd.Timestamp(short_exp) - entry_dt).days} DTE)")
print(f"  Long leg expiration:  {long_exp} ({(pd.Timestamp(long_exp) - entry_dt).days} DTE)")

# ─────────────────────────────────────────────
# 2. GET REAL OPTIONS PRICES AT ENTRY
# ─────────────────────────────────────────────
time.sleep(1)  # rate limit
print(f"\n  Fetching options chains...")
short_chain = get_options_chain(TICKER, entry_date, short_exp, "call")
time.sleep(1)
long_chain = get_options_chain(TICKER, entry_date, long_exp, "call")

short_opt = find_atm_option(short_chain, entry_price)
long_opt = find_atm_option(long_chain, entry_price)

print(f"\n  ── SHORT LEG (SELL) ──")
print(f"    Symbol: {short_opt['symbol']}")
print(f"    Strike: ${short_opt['strike']}")
print(f"    Bid/Ask: ${short_opt['bid']:.2f} / ${short_opt['ask']:.2f}")
print(f"    Mid: ${short_opt['mid']:.2f}")
print(f"    IV: {short_opt['iv']*100:.1f}%")
print(f"    Delta: {short_opt['delta']:.3f}")
print(f"    Volume: {short_opt['volume']}, OI: {short_opt['openInterest']}")

print(f"\n  ── LONG LEG (BUY) ──")
print(f"    Symbol: {long_opt['symbol']}")
print(f"    Strike: ${long_opt['strike']}")
print(f"    Bid/Ask: ${long_opt['bid']:.2f} / ${long_opt['ask']:.2f}")
print(f"    Mid: ${long_opt['mid']:.2f}")
print(f"    IV: {long_opt['iv']*100:.1f}%")
print(f"    Delta: {long_opt['delta']:.3f}")
print(f"    Volume: {long_opt['volume']}, OI: {long_opt['openInterest']}")

# REALISTIC ENTRY: sell short at bid, buy long at ask
entry_short_price = short_opt["bid"]    # we sell at bid
entry_long_price = long_opt["ask"]      # we buy at ask
net_debit_real = entry_long_price - entry_short_price

# MID-PRICE entry for comparison
net_debit_mid = long_opt["mid"] - short_opt["mid"]

print(f"\n  ── SPREAD ENTRY ──")
print(f"    Net debit (real bid/ask): ${net_debit_real:.2f}")
print(f"    Net debit (mid-price):    ${net_debit_mid:.2f}")
print(f"    Bid/ask slippage cost:    ${net_debit_real - net_debit_mid:.2f} ({(net_debit_real - net_debit_mid)/net_debit_mid*100:.1f}%)")

# ─────────────────────────────────────────────
# 3. TRACK DAILY P&L WITH REAL PRICES
# ─────────────────────────────────────────────
print(f"\n{'─'*100}")
print(f"DAILY TRACKING")
print(f"{'─'*100}")

# Get trading days from entry to short expiry
short_exp_dt = pd.Timestamp(short_exp)
stock_data = yf.download(TICKER, start=entry_date, end=str(short_exp_dt.date() + timedelta(days=5)),
                         auto_adjust=True, progress=False)
stock_data.columns = stock_data.columns.get_level_values(0) if isinstance(stock_data.columns, pd.MultiIndex) else stock_data.columns

daily_results = []
profit_target_hit = False
exit_date_real = None
exit_pnl_real = None
requests_made = 0

# Sample weekly to conserve API calls (100/day limit)
trading_days = stock_data.index.tolist()
# Check every 3rd trading day + last day
check_days = trading_days[::3]
if trading_days[-1] not in check_days:
    check_days.append(trading_days[-1])

for check_date in check_days:
    date_str = check_date.strftime("%Y-%m-%d")
    spot = float(stock_data.loc[check_date, "Close"])

    time.sleep(1)  # rate limit
    requests_made += 1

    # Get current option prices
    s_chain = get_options_chain(TICKER, date_str, short_exp, "call")
    time.sleep(1)
    requests_made += 1
    l_chain = get_options_chain(TICKER, date_str, long_exp, "call")

    if s_chain is None or l_chain is None:
        print(f"  {date_str}: API error, skipping")
        continue

    s_opt = find_atm_option(s_chain, short_opt["strike"])  # same strike
    l_opt = find_atm_option(l_chain, long_opt["strike"])    # same strike

    if s_opt is None or l_opt is None:
        print(f"  {date_str}: Could not find options, skipping")
        continue

    # To close: buy back short at ask, sell long at bid
    close_cost = s_opt["ask"]     # buy back short at ask
    close_recv = l_opt["bid"]     # sell long at bid
    spread_value_real = close_recv - close_cost
    pnl_real = spread_value_real - net_debit_real  # negative debit = we paid
    # Actually: we received entry_short_price and paid entry_long_price
    # To close: we pay close_cost and receive close_recv
    # PnL = (entry_short_price - close_cost) + (close_recv - entry_long_price)
    pnl_real = (entry_short_price - close_cost) + (close_recv - entry_long_price)
    pnl_pct_real = (pnl_real / net_debit_real) * 100

    # Mid-price P&L for comparison
    spread_mid = l_opt["mid"] - s_opt["mid"]
    pnl_mid = spread_mid - net_debit_mid
    pnl_pct_mid = (pnl_mid / net_debit_mid) * 100 if net_debit_mid > 0 else 0

    days_held = (check_date - pd.Timestamp(entry_date)).days

    daily_results.append({
        "Date": date_str,
        "Spot": round(spot, 2),
        "Short Bid/Ask": f"${s_opt['bid']:.2f}/${s_opt['ask']:.2f}",
        "Long Bid/Ask": f"${l_opt['bid']:.2f}/${l_opt['ask']:.2f}",
        "PnL (real)": round(pnl_real, 2),
        "PnL % (real)": round(pnl_pct_real, 1),
        "PnL (mid)": round(pnl_mid, 2),
        "PnL % (mid)": round(pnl_pct_mid, 1),
        "Days": days_held,
    })

    status = "WIN" if pnl_pct_real >= SPREAD_PROFIT_TARGET * 100 else ("LOSS" if pnl_real < 0 else "open")
    print(f"  {date_str} | Spot ${spot:.2f} | Short {s_opt['bid']:.2f}/{s_opt['ask']:.2f} "
          f"| Long {l_opt['bid']:.2f}/{l_opt['ask']:.2f} "
          f"| PnL(real): ${pnl_real:.2f} ({pnl_pct_real:.1f}%) "
          f"| PnL(mid): ${pnl_mid:.2f} ({pnl_pct_mid:.1f}%) | {status}")

    if pnl_pct_real >= SPREAD_PROFIT_TARGET * 100 and not profit_target_hit:
        profit_target_hit = True
        exit_date_real = date_str
        exit_pnl_real = pnl_pct_real
        print(f"  >>> PROFIT TARGET HIT at {pnl_pct_real:.1f}% <<<")

    if requests_made >= 80:  # leave buffer on 100/day limit
        print(f"\n  (Approaching API rate limit, stopping daily tracking)")
        break

# ─────────────────────────────────────────────
# 4. FINAL RESULTS
# ─────────────────────────────────────────────
print(f"\n{'='*100}")
print("RESULTS COMPARISON: REAL vs MODEL")
print(f"{'='*100}")

if daily_results:
    last = daily_results[-1]
    print(f"\n  Entry Date:    {entry_date}")
    print(f"  Entry Spot:    ${entry_price:.2f}")
    print(f"  Last Check:    {last['Date']}")
    print(f"  Last Spot:     ${last['Spot']}")
    print()
    print(f"  {'Metric':<25} {'Real (bid/ask)':<20} {'Mid-price':<20}")
    print(f"  {'─'*65}")
    print(f"  {'Net Debit':<25} ${net_debit_real:<19.2f} ${net_debit_mid:<19.2f}")
    print(f"  {'Current PnL':<25} ${last['PnL (real)']:<19.2f} ${last['PnL (mid)']:<19.2f}")
    print(f"  {'Current PnL %':<25} {last['PnL % (real)']:<19.1f}% {last['PnL % (mid)']:<18.1f}%")
    if profit_target_hit:
        print(f"\n  PROFIT TARGET HIT: {exit_date_real} at {exit_pnl_real:.1f}%")
    else:
        print(f"\n  Profit target NOT yet hit")

    slippage = last["PnL % (real)"] - last["PnL % (mid)"]
    print(f"\n  Bid/Ask Slippage Impact: {slippage:.1f}% drag on returns")

print(f"\n  API requests used: {requests_made}")
print(f"{'='*100}")
