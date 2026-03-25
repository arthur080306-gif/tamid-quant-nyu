import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy import stats
import warnings
warnings.filterwarnings("ignore")


# ── Daylight Hours ────────────────────────────────────────────────────────────

def daylight_hours(doy: np.ndarray, lat_deg: float = 40.0) -> np.ndarray:
    """Astronomical day length (Spencer / Cooper formula) in hours."""
    lat = np.radians(lat_deg)
    # Solar declination (radians)
    delta = 0.4093 * np.sin(2 * np.pi * (doy - 81) / 365)
    # Hour angle at sunrise/sunset
    cos_ha = -np.tan(lat) * np.tan(delta)
    cos_ha = np.clip(cos_ha, -1.0, 1.0)          # polar clamp
    ha = np.arccos(cos_ha)
    return 24.0 * ha / np.pi                       # hours


# ── Data ──────────────────────────────────────────────────────────────────────

print("Downloading data …")
raw = yf.download(["IWM", "XLU"], start="2010-01-01", auto_adjust=True, progress=False)
close = raw["Close"].dropna()

ret = close.pct_change().dropna()
ret.columns = ["IWM", "XLU"]

# Day-of-year for each trading day
doy = ret.index.day_of_year.values
D_t = daylight_hours(doy)
D_series = pd.Series(D_t, index=ret.index, name="daylight")


# ── Rolling Signal Construction ───────────────────────────────────────────────

WINDOW = 252

D_mean   = D_series.rolling(WINDOW).mean()
D_std    = D_series.rolling(WINDOW).std()
D_dev    = D_series - D_mean                      # D_t - D_mean

R_spread = ret["IWM"] - ret["XLU"]

# Rolling OLS beta: regress R_spread on D_dev
def rolling_ols_beta(y: pd.Series, x: pd.Series, window: int) -> pd.Series:
    betas = np.full(len(y), np.nan)
    y_arr, x_arr = y.values, x.values
    for i in range(window - 1, len(y)):
        yi = y_arr[i - window + 1 : i + 1]
        xi = x_arr[i - window + 1 : i + 1]
        mask = np.isfinite(yi) & np.isfinite(xi)
        if mask.sum() < 30:
            continue
        slope, *_ = stats.linregress(xi[mask], yi[mask])
        betas[i] = slope
    return pd.Series(betas, index=y.index)

print("Computing rolling OLS betas (this may take ~30 s) …")
beta_SAD = rolling_ols_beta(R_spread, D_dev, WINDOW)

# Raw position size
w_raw = beta_SAD * D_dev / D_std
w_t   = w_raw.clip(-1.0, 1.0)                     # clamp to [-1, 1]

# Strategy return: use prior-day signal (no look-ahead)
strat_ret = w_t.shift(1) * R_spread

# Drop warm-up period
strat_ret = strat_ret.dropna()
iwm_ret   = ret["IWM"].reindex(strat_ret.index)
w_plot    = w_t.reindex(strat_ret.index)


# ── Performance Stats ─────────────────────────────────────────────────────────

def max_drawdown(cum: pd.Series) -> float:
    roll_max = cum.cummax()
    dd = (cum - roll_max) / roll_max
    return dd.min()

def annualised_sharpe(r: pd.Series) -> float:
    return r.mean() / r.std() * np.sqrt(252)

ann_ret   = strat_ret.mean() * 252
sharpe    = annualised_sharpe(strat_ret)
cum_strat = (1 + strat_ret).cumprod()
mdd       = max_drawdown(cum_strat)
win_rate  = (strat_ret > 0).mean()

print("\n── SAD Seasonality Backtest ─────────────────────────")
print(f"  Period          : {strat_ret.index[0].date()} → {strat_ret.index[-1].date()}")
print(f"  Annualised Ret  : {ann_ret*100:+.2f}%")
print(f"  Annualised Sharpe: {sharpe:.3f}")
print(f"  Max Drawdown    : {mdd*100:.2f}%")
print(f"  Win Rate        : {win_rate*100:.1f}%")
print("─────────────────────────────────────────────────────\n")


# ── Rolling Sharpe (252-day) ──────────────────────────────────────────────────

roll_sharpe = strat_ret.rolling(WINDOW).apply(annualised_sharpe, raw=True)


# ── Plot ──────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(3, 1, figsize=(13, 11), sharex=True)
fig.suptitle("SAD Seasonality Strategy  |  Long IWM / Short XLU on Daylight Deviation",
             fontsize=13, fontweight="bold")

cum_iwm = (1 + iwm_ret).cumprod()

# Panel 1 – cumulative returns
ax = axes[0]
ax.plot(cum_strat.index, cum_strat.values, color="steelblue", lw=1.4, label="SAD Strategy")
ax.plot(cum_iwm.index,   cum_iwm.values,   color="darkorange", lw=1.2, alpha=0.8, label="IWM Buy & Hold")
ax.set_ylabel("Growth of $1")
ax.set_title("Cumulative Return")
ax.legend(loc="upper left", fontsize=9)
ax.grid(axis="y", alpha=0.3)

# Panel 2 – daily position size
ax = axes[1]
ax.fill_between(w_plot.index, w_plot.values, 0,
                where=w_plot.values >= 0, color="steelblue", alpha=0.6, label="Long IWM")
ax.fill_between(w_plot.index, w_plot.values, 0,
                where=w_plot.values < 0,  color="tomato",    alpha=0.6, label="Short IWM")
ax.axhline(0, color="k", lw=0.7)
ax.set_ylabel("Position size (w)")
ax.set_title("Daily Position Size")
ax.set_ylim(-1.1, 1.1)
ax.legend(loc="upper left", fontsize=9)
ax.grid(axis="y", alpha=0.3)

# Panel 3 – rolling Sharpe
ax = axes[2]
ax.plot(roll_sharpe.index, roll_sharpe.values, color="purple", lw=1.2)
ax.axhline(0, color="k", lw=0.7)
ax.axhline(1, color="green",  lw=0.7, ls="--", alpha=0.7, label="Sharpe = 1")
ax.axhline(-1, color="red",   lw=0.7, ls="--", alpha=0.7, label="Sharpe = −1")
ax.set_ylabel("Sharpe ratio")
ax.set_title("Rolling 252-Day Sharpe")
ax.legend(loc="upper left", fontsize=9)
ax.grid(axis="y", alpha=0.3)

ax.xaxis.set_major_locator(mdates.YearLocator(2))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
plt.xticks(rotation=30)

plt.tight_layout()
plt.savefig("sad_backtest.png", dpi=150, bbox_inches="tight")
print("Plot saved → sad_backtest.png")
plt.show()
