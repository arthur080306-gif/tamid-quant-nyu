#!/usr/bin/env python3
"""
JEDI — Political Beta Factor Model for U.S. Defense Equities
=============================================================
Backtest implementation for the TAMID Quant JEDI strategy.

Daily changes in Polymarket political prediction market probabilities are used
to predict next-day log returns of a custom equal-weight defense equity index
("JEDI": LMT, RTX, NOC, GD, HII, BAH, LDOS).

Core model: PCA → Elastic Net (walk-forward, rolling 252-day window).
Signal: long if predicted return > +0.15%, short if < -0.15%, flat otherwise.
Position sizing: quarter-Kelly scaled by rolling 21-day IC.

Data source: Polymarket public API (no API key required).

Author: Aaron Blatnoy
Date: 2026-03-09
"""

import os
import sys
import json
import time
import logging
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import requests
from scipy import stats

import yfinance as yf
from sklearn.decomposition import PCA
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="yfinance")

# ---------------------------------------------------------------------------
# 1. CONFIGURATION
# ---------------------------------------------------------------------------

CONFIG = {
    # JEDI index constituents
    "jedi_tickers": ["LMT", "RTX", "NOC", "GD", "HII", "BAH", "LDOS"],
    # Benchmark
    "benchmark_ticker": "ITA",
    # Data history
    "start_date": "2024-01-02",
    "end_date": "2026-03-14",
    # Polymarket API (public, no key required)
    "polymarket_gamma_url": "https://gamma-api.polymarket.com",
    "polymarket_clob_url": "https://clob.polymarket.com",
    # Search keywords for defense-relevant prediction markets
    "polymarket_keywords": {
        "geopolitical": [
            "ukraine", "russia", "china", "taiwan", "iran", "israel", "gaza",
            "houthi", "yemen", "syria", "nato", "war", "ceasefire", "strike",
            "missile", "nuclear", "military", "troops", "invasion", "conflict",
            "bomb", "attack", "sanction", "weapon", "army", "navy", "drone",
        ],
        "fiscal": [
            "fed ", "federal reserve", "interest rate", "rate cut", "rate hike",
            "shutdown", "debt ceiling", "budget", "defense spending", "tariff",
            "recession", "gdp", "inflation", "cpi",
        ],
        "policy": [
            "executive order", "pentagon", "defense secretary",
            "national security", "doge", "government efficiency",
        ],
    },
    # Keywords to EXCLUDE (election/nomination noise, sports, entertainment)
    "polymarket_exclude_keywords": [
        "win the 2024", "win the 2028", "nominee", "nomination",
        "nba", "nfl", "mlb", "nhl", "super bowl", "champion",
        "oscars", "grammy", "emmy", "bachelor", "will trump say",
        "what will trump say", "democratic presidential", "republican presidential",
    ],
    # Minimum daily data points for a market to be included
    "min_market_history_days": 15,
    # Max markets to include
    "max_markets": 60,
    # Model parameters
    "rolling_window": 120,          # trading days for training (shorter for Polymarket's limited history)
    "refit_frequency": 21,          # re-estimate betas every 21 days (~monthly)
    "pca_variance_threshold": 0.95, # keep components explaining 95% variance
    "elastic_net_l1_ratios": [0.1, 0.3, 0.5, 0.7, 0.9, 0.95],
    "ts_cv_splits": 5,             # TimeSeriesSplit folds
    # Signal & position sizing
    "signal_threshold": 0.0005,     # ±0.05% predicted return (calibrated to Polymarket signal magnitude)
    "kelly_fraction": 0.25,         # quarter-Kelly
    "ic_rolling_window": 21,        # rolling IC lookback
    "leverage_cap": 2.0,            # max gross leverage
    # Risk management
    "drawdown_half_stop": -0.05,    # -5% drawdown → halve position
    "drawdown_flat_stop": -0.10,    # -10% drawdown → go flat
    # Forward-fill limit for probability gaps
    "ffill_limit": 5,
    # Output
    "output_dir": Path(__file__).resolve().parent / "output",
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("JEDI")

# ---------------------------------------------------------------------------
# 2. POLYMARKET API CLIENT
# ---------------------------------------------------------------------------

class PolymarketClient:
    """
    Polymarket public API client for fetching prediction market data.

    Uses two APIs:
    - Gamma API (gamma-api.polymarket.com): market/event discovery
    - CLOB API (clob.polymarket.com): historical price data

    No API key or authentication required for read operations.
    Falls back to synthetic data if API is unreachable.
    """

    def __init__(self, gamma_url: str, clob_url: str):
        self.gamma_url = gamma_url.rstrip("/")
        self.clob_url = clob_url.rstrip("/")
        self.session = requests.Session()
        self.session.headers.update({
            "Accept": "application/json",
            "User-Agent": "JEDI-Backtest/1.0",
        })

    def discover_political_markets(
        self,
        keywords: dict[str, list[str]],
        exclude_keywords: list[str] = None,
        max_per_batch: int = 100,
    ) -> list[dict]:
        """
        Search Polymarket for geopolitical/fiscal/policy markets.

        Args:
            keywords: dict mapping category names to keyword lists
            exclude_keywords: list of keywords to filter OUT (elections, sports, etc.)
            max_per_batch: max events per API page

        Returns:
            List of market dicts with: question, yes_token, category, volume
        """
        all_keywords = set()
        keyword_to_cat = {}
        for cat, kws in keywords.items():
            for kw in kws:
                all_keywords.add(kw)
                keyword_to_cat[kw] = cat

        exclude_keywords = [kw.lower() for kw in (exclude_keywords or [])]

        all_markets = []
        seen_tokens = set()

        for closed_val in ["true", "false"]:
            for offset in range(0, 1000, max_per_batch):
                try:
                    resp = self.session.get(
                        f"{self.gamma_url}/events",
                        params={
                            "limit": max_per_batch,
                            "offset": offset,
                            "closed": closed_val,
                            "order": "volume",
                            "ascending": "false",
                        },
                        timeout=20,
                    )
                    resp.raise_for_status()
                    events = resp.json()
                    if not events:
                        break
                except Exception as e:
                    log.debug(f"Polymarket event fetch failed: {e}")
                    break

                for event in events:
                    title = event.get("title", "").lower()

                    # Skip excluded topics
                    if any(exc in title for exc in exclude_keywords):
                        continue

                    # Match keywords to categories
                    matched_cats = set()
                    for kw in all_keywords:
                        if kw in title:
                            matched_cats.add(keyword_to_cat[kw])

                    if not matched_cats:
                        continue

                    for m in event.get("markets", []):
                        question = m.get("question", "").lower()

                        # Also exclude at question level
                        if any(exc in question for exc in exclude_keywords):
                            continue

                        raw_ids = m.get("clobTokenIds", "")
                        if not raw_ids:
                            continue
                        token_ids = json.loads(raw_ids) if isinstance(raw_ids, str) else raw_ids
                        if not token_ids:
                            continue

                        yes_token = token_ids[0]
                        if yes_token in seen_tokens:
                            continue
                        seen_tokens.add(yes_token)

                        all_markets.append({
                            "event": event.get("title", ""),
                            "question": m.get("question", ""),
                            "yes_token": yes_token,
                            "categories": sorted(matched_cats),
                            "volume": float(event.get("volume", 0) or 0),
                        })

        all_markets.sort(key=lambda x: x["volume"], reverse=True)
        log.info(f"  Discovered {len(all_markets)} defense-relevant prediction markets")
        return all_markets

    def fetch_price_history(
        self, token_id: str, fidelity: int = 1440
    ) -> Optional[pd.Series]:
        """
        Fetch daily price history for a single market token.

        Args:
            token_id: Polymarket CLOB token ID (YES outcome)
            fidelity: data granularity in minutes (1440 = daily)

        Returns:
            Series with date index and probability values, or None on failure
        """
        try:
            resp = self.session.get(
                f"{self.clob_url}/prices-history",
                params={
                    "market": token_id,
                    "interval": "all",
                    "fidelity": fidelity,
                },
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json()
            history = data.get("history", [])
            if not history:
                return None

            records = []
            for point in history:
                ts = point.get("t")
                price = point.get("p")
                if ts is not None and price is not None:
                    records.append({
                        "date": pd.Timestamp(ts, unit="s").normalize(),
                        "prob": float(price),
                    })

            if not records:
                return None

            df = pd.DataFrame(records).set_index("date").sort_index()
            df = df[~df.index.duplicated(keep="last")]
            return df["prob"]

        except Exception as e:
            log.debug(f"Price history fetch failed for token {token_id[:30]}...: {e}")
            return None

    def fetch_all_probabilities(
        self,
        keywords: dict[str, list[str]],
        exclude_keywords: list[str] = None,
        min_history_days: int = 15,
        max_markets: int = 60,
    ) -> Optional[pd.DataFrame]:
        """
        Discover defense-relevant markets and fetch their daily probability histories.

        Returns wide DataFrame: date index × market columns (probabilities 0-1).
        Returns None if insufficient data is available.
        """
        markets = self.discover_political_markets(keywords, exclude_keywords)
        if not markets:
            log.warning("No political markets found on Polymarket.")
            return None

        # Fetch histories for top markets by volume
        prob_series = {}
        n_checked = 0
        for m in markets:
            if len(prob_series) >= max_markets:
                break
            n_checked += 1

            series = self.fetch_price_history(m["yes_token"])
            if series is not None and len(series) >= min_history_days:
                # Clean column name
                q = m["question"][:50].replace(" ", "_").replace("?", "")
                col_name = f"{'_'.join(m['categories'])}_{q}"
                col_name = col_name[:60]  # truncate
                prob_series[col_name] = series
                log.info(f"    {len(series):4d}d | {m['question'][:65]}")

            time.sleep(0.05)  # be polite to the API

            if n_checked >= 200 and len(prob_series) < 3:
                break  # not finding enough data

        if len(prob_series) < 3:
            log.warning(f"Only found {len(prob_series)} markets with sufficient history.")
            return None

        prob_df = pd.DataFrame(prob_series)
        prob_df.index = pd.to_datetime(prob_df.index)
        prob_df = prob_df.sort_index()
        return prob_df

    @staticmethod
    def generate_synthetic_probabilities(
        start_date: str, end_date: str, n_markets: int = 12, seed: int = 42
    ) -> pd.DataFrame:
        """
        Generate synthetic Polymarket-like probability time series for backtesting
        when the API is unreachable.

        Creates correlated random-walk probabilities with realistic properties:
        - Values bounded in [0.01, 0.99]
        - Mean-reverting dynamics
        - Cross-market correlation structure
        - Budget/geopolitical/fiscal category structure
        """
        rng = np.random.default_rng(seed)

        dates = pd.bdate_range(start=start_date, end=end_date)
        n_days = len(dates)

        categories = {
            "GEO": ["GEO_Ukraine_ceasefire", "GEO_China_Taiwan_escalate", "GEO_Iran_strike_2026"],
            "FISCAL": ["FISCAL_Fed_rate_cut_25", "FISCAL_Fed_rate_hold", "FISCAL_Debt_ceiling_raise",
                        "FISCAL_Defense_budget_up"],
            "POLITICAL": ["POL_Trump_approval_45", "POL_2028_Dem_nominee_Harris",
                          "POL_Gov_shutdown_Q2", "POL_Tariff_increase", "POL_Executive_order_defense"],
        }

        market_names = []
        for cat, names in categories.items():
            market_names.extend(names)
        n_markets = len(market_names)

        # Correlation structure: markets in same category are more correlated
        corr = np.eye(n_markets) * 0.5
        idx = 0
        for cat, names in categories.items():
            n = len(names)
            for i in range(idx, idx + n):
                for j in range(idx, idx + n):
                    if i != j:
                        corr[i, j] = 0.3 + rng.uniform(0, 0.2)
            idx += n
        corr = (corr + corr.T) / 2
        np.fill_diagonal(corr, 1.0)

        eigvals, eigvecs = np.linalg.eigh(corr)
        eigvals = np.maximum(eigvals, 1e-6)
        corr = eigvecs @ np.diag(eigvals) @ eigvecs.T
        corr = corr / np.sqrt(np.outer(np.diag(corr), np.diag(corr)))

        L = np.linalg.cholesky(corr)
        z = rng.standard_normal((n_days, n_markets))
        innovations = z @ L.T

        volatilities = rng.uniform(0.02, 0.08, n_markets)
        mean_reversion = rng.uniform(0.01, 0.05, n_markets)
        initial_probs = rng.uniform(0.15, 0.75, n_markets)

        logit = np.log(initial_probs / (1 - initial_probs))
        long_run_logit = logit.copy()

        prob_data = np.zeros((n_days, n_markets))
        for t in range(n_days):
            logit += mean_reversion * (long_run_logit - logit) + volatilities * innovations[t]
            prob_data[t] = 1.0 / (1.0 + np.exp(-logit))

        prob_data = np.clip(prob_data, 0.01, 0.99)

        # Inject small correlation with defense sector returns
        try:
            ita = yf.download("ITA", start=start_date, end=end_date, progress=False)
            if "Close" in ita.columns:
                ita_close = ita["Close"]
            else:
                ita_close = ita[("Close", "ITA")]
            ita_ret = np.log(ita_close / ita_close.shift(1)).dropna()

            common_dates = dates.intersection(ita_ret.index)
            if len(common_dates) > 100:
                ita_aligned = ita_ret.reindex(dates).fillna(0).values
                if ita_aligned.ndim > 1:
                    ita_aligned = ita_aligned.flatten()
                signal_strength = rng.uniform(0.03, 0.12, n_markets)
                for j in range(n_markets):
                    shifted_ret = np.roll(ita_aligned, 1)
                    shifted_ret[0] = 0
                    logit_adjustment = signal_strength[j] * shifted_ret * 10
                    logit_prob = np.log(prob_data[:, j] / (1 - prob_data[:, j]))
                    logit_prob += logit_adjustment[:len(logit_prob)]
                    prob_data[:, j] = 1.0 / (1.0 + np.exp(-logit_prob))
                    prob_data[:, j] = np.clip(prob_data[:, j], 0.01, 0.99)
        except Exception:
            pass

        df = pd.DataFrame(prob_data, index=dates, columns=market_names)
        return df


# ---------------------------------------------------------------------------
# 3. DATA FETCHING
# ---------------------------------------------------------------------------

def fetch_jedi_index(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    """
    Download JEDI constituent prices from Yahoo Finance.
    Returns DataFrame with columns: 'jedi_close', 'jedi_log_return'.
    Equal-weight index based on daily log returns.
    """
    log.info(f"Downloading JEDI constituents: {tickers}")
    data = yf.download(tickers, start=start, end=end, progress=False)

    if "Close" in data.columns and not isinstance(data.columns, pd.MultiIndex):
        closes = data[["Close"]].copy()
        closes.columns = [tickers[0]]
    else:
        closes = data["Close"].copy()

    if closes.empty:
        raise ValueError("Failed to download JEDI constituent data from Yahoo Finance.")

    min_obs = len(closes) * 0.8
    valid_tickers = closes.columns[closes.count() >= min_obs].tolist()
    if len(valid_tickers) < 3:
        raise ValueError(f"Only {len(valid_tickers)} tickers have sufficient data.")
    closes = closes[valid_tickers]

    log.info(f"  Using {len(valid_tickers)} tickers with sufficient history: {valid_tickers}")

    closes = closes.ffill(limit=3).dropna()
    log_returns = np.log(closes / closes.shift(1))
    jedi_log_return = log_returns.mean(axis=1)

    jedi_cumret = jedi_log_return.cumsum()
    jedi_close = 100 * np.exp(jedi_cumret)

    result = pd.DataFrame({
        "jedi_close": jedi_close,
        "jedi_log_return": jedi_log_return,
    }).dropna()

    log.info(f"  JEDI index: {len(result)} trading days from {result.index[0].date()} to {result.index[-1].date()}")
    return result


def fetch_benchmark(ticker: str, start: str, end: str) -> pd.DataFrame:
    """
    Download benchmark (ITA) prices from Yahoo Finance.
    Returns DataFrame with 'bench_close' and 'bench_log_return'.
    """
    log.info(f"Downloading benchmark: {ticker}")
    data = yf.download(ticker, start=start, end=end, progress=False)

    if data.empty:
        raise ValueError(f"Failed to download benchmark {ticker}.")

    if isinstance(data.columns, pd.MultiIndex):
        close = data[("Close", ticker)]
    else:
        close = data["Close"]

    close = close.dropna()
    log_ret = np.log(close / close.shift(1))

    result = pd.DataFrame({
        "bench_close": close,
        "bench_log_return": log_ret,
    }).dropna()

    log.info(f"  Benchmark: {len(result)} trading days")
    return result


def fetch_polymarket_probabilities(config: dict) -> pd.DataFrame:
    """
    Fetch Polymarket probabilities via public API, falling back to synthetic data.
    Returns DataFrame: date index × market columns (probabilities 0-1).
    """
    log.info("Fetching Polymarket prediction market data...")

    client = PolymarketClient(
        gamma_url=config["polymarket_gamma_url"],
        clob_url=config["polymarket_clob_url"],
    )

    try:
        prob_df = client.fetch_all_probabilities(
            keywords=config["polymarket_keywords"],
            exclude_keywords=config.get("polymarket_exclude_keywords", []),
            min_history_days=config["min_market_history_days"],
            max_markets=config["max_markets"],
        )
    except Exception as e:
        log.warning(f"Polymarket API error: {e}")
        prob_df = None

    if prob_df is not None and len(prob_df.columns) >= 3 and len(prob_df) >= 50:
        log.info(f"  Live Polymarket data: {prob_df.shape[1]} markets, {len(prob_df)} days")
        return prob_df

    log.warning("=" * 70)
    log.warning("  FALLING BACK TO SYNTHETIC DATA — Polymarket API returned")
    log.warning("  insufficient data. This may be due to API changes or")
    log.warning("  limited market history. Results use simulated probabilities.")
    log.warning("=" * 70)

    prob_df = PolymarketClient.generate_synthetic_probabilities(
        config["start_date"], config["end_date"]
    )
    log.info(f"  Synthetic data: {prob_df.shape[1]} markets, {len(prob_df)} days")
    return prob_df


# ---------------------------------------------------------------------------
# 4. FEATURE ENGINEERING
# ---------------------------------------------------------------------------

def engineer_features(
    probs: pd.DataFrame,
    jedi_data: pd.DataFrame,
    ffill_limit: int = 5,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Compute features (ΔP: daily probability changes) and align with
    next-day JEDI returns (the core one-day lag hypothesis).

    Handles sparse/unbalanced panels: markets that don't exist on a given
    day get ΔP=0 (no change), which is the neutral assumption.

    Returns:
        features: DataFrame of ΔP values at day t
        target: Series of JEDI log returns at day t+1
    """
    log.info("Engineering features...")

    probs_filled = probs.ffill(limit=ffill_limit)
    delta_p = probs_filled.diff()
    delta_p = delta_p.dropna(axis=1, how="all")

    # Keep columns with at least 20% non-NaN observations
    min_obs = len(delta_p) * 0.2
    valid_cols = delta_p.columns[delta_p.count() >= min_obs]
    delta_p = delta_p[valid_cols]

    if delta_p.empty:
        raise ValueError("No valid ΔP features after filtering.")

    log.info(f"  ΔP features: {delta_p.shape[1]} markets retained")

    # Keep NaNs for now — the walk-forward model will handle sparse panels
    # by selecting columns with sufficient data in each training window

    # Align: features at day t, target = JEDI return at day t+1
    target = jedi_data["jedi_log_return"].shift(-1)

    common_dates = delta_p.index.intersection(target.index)
    features = delta_p.loc[common_dates]
    target = target.loc[common_dates]

    valid_mask = target.notna()
    features = features.loc[valid_mask]
    target = target.loc[valid_mask]

    log.info(f"  Aligned dataset: {len(features)} observations, {features.shape[1]} features")
    return features, target


# ---------------------------------------------------------------------------
# 5. MODEL: PCA + ELASTIC NET (WALK-FORWARD)
# ---------------------------------------------------------------------------

class JEDIFactorModel:
    """
    PCA → Elastic Net factor model with walk-forward estimation.

    - PCA reduces multicollinearity among prediction market ΔP features
    - ElasticNetCV with TimeSeriesSplit for sparse factor selection
    - Re-fit monthly (every `refit_frequency` trading days)
    - Rolling `window` day training set
    """

    def __init__(
        self,
        window: int = 252,
        refit_frequency: int = 21,
        pca_variance: float = 0.95,
        l1_ratios: list[float] = None,
        cv_splits: int = 5,
    ):
        self.window = window
        self.refit_frequency = refit_frequency
        self.pca_variance = pca_variance
        self.l1_ratios = l1_ratios or [0.1, 0.3, 0.5, 0.7, 0.9, 0.95]
        self.cv_splits = cv_splits

        self.scaler = None
        self.pca = None
        self.model = None
        self._last_fit_idx = -self.refit_frequency

    def _fit(self, X: np.ndarray, y: np.ndarray):
        """Fit PCA + ElasticNetCV on training window."""
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        n_components = min(X_scaled.shape[1], X_scaled.shape[0] - 1)
        if n_components < 2:
            n_components = min(2, X_scaled.shape[1])

        self.pca = PCA(n_components=self.pca_variance, svd_solver="full")
        try:
            X_pca = self.pca.fit_transform(X_scaled)
        except Exception:
            self.pca = PCA(n_components=min(3, n_components), svd_solver="full")
            X_pca = self.pca.fit_transform(X_scaled)

        if X_pca.shape[0] <= self.cv_splits + 1:
            from sklearn.linear_model import ElasticNet
            self.model = ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=5000)
            self.model.fit(X_pca, y)
            return

        tscv = TimeSeriesSplit(n_splits=self.cv_splits)
        self.model = ElasticNetCV(
            l1_ratio=self.l1_ratios,
            cv=tscv,
            max_iter=10000,
            n_alphas=50,
            random_state=42,
        )
        self.model.fit(X_pca, y)

    def predict_one(self, X_row: np.ndarray) -> float:
        """Predict single observation using current model."""
        if self.model is None or self.scaler is None or self.pca is None:
            return 0.0
        X_scaled = self.scaler.transform(X_row.reshape(1, -1))
        X_pca = self.pca.transform(X_scaled)
        return float(self.model.predict(X_pca)[0])

    def walk_forward(
        self, features: pd.DataFrame, target: pd.Series
    ) -> pd.Series:
        """
        Walk-forward prediction loop with dynamic feature selection.

        For each training window, only uses feature columns that have
        >50% non-NaN data in that window (handles Polymarket's sparse panels).

        For each day t (starting after the initial training window):
        1. If refit is due, select valid features and train on [t - window : t]
        2. Predict return at t+1 using features at t

        Returns Series of predicted returns aligned with target dates.
        """
        n = len(features)
        predictions = pd.Series(index=features.index, dtype=float)
        predictions[:] = np.nan

        fit_count = 0
        start_idx = self.window
        self._active_cols = None  # columns used by current model

        log.info(f"  Walk-forward: {n - start_idx} prediction days, "
                 f"window={self.window}, refit every {self.refit_frequency} days")

        for t in range(start_idx, n):
            if (t - self._last_fit_idx) >= self.refit_frequency or self.model is None:
                train_start = max(0, t - self.window)
                train_slice = features.iloc[train_start:t]
                y_train_full = target.iloc[train_start:t]

                # Select columns with >50% real data in this window
                col_coverage = train_slice.notna().mean()
                valid_cols = col_coverage[col_coverage > 0.5].index.tolist()

                if len(valid_cols) < 2:
                    predictions.iloc[t] = 0.0
                    continue

                # Extract training data, fill remaining NaN with 0
                X_train = train_slice[valid_cols].fillna(0.0).values
                y_train = y_train_full.values

                # Drop rows where target is NaN
                valid_rows = np.isfinite(y_train)
                X_train = X_train[valid_rows]
                y_train = y_train[valid_rows]

                if len(X_train) < 30 or np.std(y_train) < 1e-10:
                    predictions.iloc[t] = 0.0
                    continue

                try:
                    self._fit(X_train, y_train)
                    self._active_cols = valid_cols
                    self._last_fit_idx = t
                    fit_count += 1
                except Exception as e:
                    log.debug(f"  Fit failed at index {t}: {e}")
                    predictions.iloc[t] = 0.0
                    continue

            # Predict using active columns
            if self._active_cols is None:
                predictions.iloc[t] = 0.0
                continue

            try:
                x_row = features.iloc[t][self._active_cols].fillna(0.0).values
                predictions.iloc[t] = self.predict_one(x_row)
            except Exception:
                predictions.iloc[t] = 0.0

        log.info(f"  Model re-fitted {fit_count} times")
        valid_preds = predictions.dropna()
        non_zero = (valid_preds.abs() > 1e-10).sum()
        log.info(f"  Generated {len(valid_preds)} predictions ({non_zero} non-zero)")
        return predictions


# ---------------------------------------------------------------------------
# 6. SIGNAL CONSTRUCTION & POSITION SIZING
# ---------------------------------------------------------------------------

def construct_signals(
    predictions: pd.Series,
    target: pd.Series,
    threshold: float = 0.0015,
    kelly_fraction: float = 0.25,
    ic_window: int = 21,
    leverage_cap: float = 2.0,
) -> pd.DataFrame:
    """
    Convert predicted returns into trading signals with position sizing.

    - Long if predicted > +threshold, short if < -threshold, flat otherwise
    - Quarter-Kelly sizing scaled by rolling 21-day IC
    - Capped at leverage_cap gross leverage

    Returns DataFrame with columns: 'signal', 'position_size', 'raw_position'.
    """
    log.info("Constructing signals...")

    signals = pd.DataFrame(index=predictions.index)
    signals["prediction"] = predictions
    signals["actual"] = target

    # Direction signal
    signals["signal"] = 0.0
    signals.loc[predictions > threshold, "signal"] = 1.0
    signals.loc[predictions < -threshold, "signal"] = -1.0

    # Rolling IC (Spearman rank correlation between prediction and actual)
    ic_values = pd.Series(index=predictions.index, dtype=float)
    pred_vals = predictions.values
    actual_vals = target.values
    for i in range(ic_window, len(predictions)):
        p = pred_vals[i - ic_window:i]
        a = actual_vals[i - ic_window:i]
        mask = np.isfinite(p) & np.isfinite(a)
        if mask.sum() >= 10:
            try:
                ic_values.iloc[i] = stats.spearmanr(p[mask], a[mask])[0]
            except Exception:
                ic_values.iloc[i] = 0.0
        else:
            ic_values.iloc[i] = 0.0

    signals["rolling_ic"] = ic_values

    # Position sizing: quarter-Kelly scaled by prediction magnitude
    # When rolling IC is available and positive, scale further by IC for conviction
    # Otherwise use base sizing to avoid zero positions during IC warm-up
    ic_positive = signals["rolling_ic"].clip(lower=0)
    pred_magnitude = predictions.abs() / threshold  # normalized strength

    # Base position: kelly * direction * magnitude
    base_position = kelly_fraction * pred_magnitude * signals["signal"]

    # IC boost: when IC is positive and reliable, scale up (cap at 2x base)
    ic_multiplier = (0.5 + ic_positive.clip(upper=0.5)).fillna(0.5)
    raw_position = base_position * ic_multiplier * 2

    signals["raw_position"] = raw_position
    signals["position_size"] = raw_position.clip(lower=-leverage_cap, upper=leverage_cap)
    signals["position_size"] = signals["position_size"].fillna(0.0)

    n_long = (signals["signal"] > 0).sum()
    n_short = (signals["signal"] < 0).sum()
    n_flat = (signals["signal"] == 0).sum()
    log.info(f"  Signals: {n_long} long, {n_short} short, {n_flat} flat days")
    log.info(f"  Avg position size: {signals['position_size'].abs().mean():.3f}")

    return signals


# ---------------------------------------------------------------------------
# 7. BACKTEST ENGINE
# ---------------------------------------------------------------------------

def run_backtest(
    signals: pd.DataFrame,
    jedi_data: pd.DataFrame,
    bench_data: pd.DataFrame,
    dd_half: float = -0.05,
    dd_flat: float = -0.10,
) -> pd.DataFrame:
    """
    Daily backtest loop with drawdown stop enforcement.

    - Apply position sizes to JEDI daily log returns
    - Enforce drawdown stops: halve at dd_half, flatten at dd_flat
    - Track equity curve and PnL

    Returns DataFrame with equity curve, returns, positions, drawdown.
    """
    log.info("Running backtest engine...")

    common_dates = signals.index.intersection(jedi_data.index).intersection(bench_data.index)
    common_dates = common_dates.sort_values()

    results = pd.DataFrame(index=common_dates)
    results["jedi_return"] = jedi_data.loc[common_dates, "jedi_log_return"]
    results["bench_return"] = bench_data.loc[common_dates, "bench_log_return"]
    results["raw_position"] = signals.loc[common_dates, "position_size"]

    equity = np.ones(len(common_dates))
    positions = np.zeros(len(common_dates))
    strat_returns = np.zeros(len(common_dates))
    peak_equity = 1.0
    drawdown_state = "normal"

    raw_pos = results["raw_position"].values
    jedi_ret = results["jedi_return"].values

    for t in range(len(common_dates)):
        if t > 0:
            current_dd = (equity[t - 1] - peak_equity) / peak_equity
        else:
            current_dd = 0.0

        if current_dd <= dd_flat:
            drawdown_state = "flat"
            positions[t] = 0.0
        elif current_dd <= dd_half:
            drawdown_state = "half"
            positions[t] = raw_pos[t] * 0.5
        else:
            if drawdown_state == "flat" and current_dd > dd_flat * 0.5:
                drawdown_state = "half"
                positions[t] = raw_pos[t] * 0.5
            elif drawdown_state == "half" and current_dd > dd_half * 0.5:
                drawdown_state = "normal"
                positions[t] = raw_pos[t]
            elif drawdown_state == "normal":
                positions[t] = raw_pos[t]
            else:
                if drawdown_state == "half":
                    positions[t] = raw_pos[t] * 0.5
                else:
                    positions[t] = 0.0

        strat_returns[t] = positions[t] * jedi_ret[t]

        if t == 0:
            equity[t] = 1.0 + strat_returns[t]
        else:
            equity[t] = equity[t - 1] * (1.0 + strat_returns[t])

        peak_equity = max(peak_equity, equity[t])

    results["position"] = positions
    results["strategy_return"] = strat_returns
    results["equity"] = equity
    results["bench_equity"] = (1.0 + results["bench_return"]).cumprod()

    rolling_peak = pd.Series(equity, index=common_dates).cummax()
    results["drawdown"] = (pd.Series(equity, index=common_dates) - rolling_peak) / rolling_peak

    bench_peak = results["bench_equity"].cummax()
    results["bench_drawdown"] = (results["bench_equity"] - bench_peak) / bench_peak

    log.info(f"  Backtest complete: {len(results)} trading days")
    return results


# ---------------------------------------------------------------------------
# 8. PERFORMANCE ANALYTICS
# ---------------------------------------------------------------------------

def compute_performance(results: pd.DataFrame) -> dict:
    """
    Compute comprehensive performance metrics for strategy and benchmark.
    """
    strat_ret = results["strategy_return"].dropna()
    bench_ret = results["bench_return"].dropna()

    def _metrics(returns: pd.Series, equity_col: str) -> dict:
        n_days = len(returns)
        if n_days < 10:
            return {k: 0.0 for k in [
                "total_return", "cagr", "sharpe", "sortino",
                "max_drawdown", "calmar", "win_rate", "profit_factor"
            ]}

        total_ret = results[equity_col].iloc[-1] / results[equity_col].iloc[0] - 1
        n_years = n_days / 252
        cagr = (1 + total_ret) ** (1 / n_years) - 1 if n_years > 0 else 0

        daily_mean = returns.mean()
        daily_std = returns.std()
        sharpe = (daily_mean / daily_std) * np.sqrt(252) if daily_std > 0 else 0

        downside = returns[returns < 0]
        downside_std = downside.std() if len(downside) > 0 else daily_std
        sortino = (daily_mean / downside_std) * np.sqrt(252) if downside_std > 0 else 0

        equity = results[equity_col]
        peak = equity.cummax()
        dd = (equity - peak) / peak
        max_dd = dd.min()

        calmar = cagr / abs(max_dd) if max_dd != 0 else 0

        winning_days = (returns > 0).sum()
        total_trading_days = (returns != 0).sum()
        win_rate = winning_days / total_trading_days if total_trading_days > 0 else 0

        gross_profit = returns[returns > 0].sum()
        gross_loss = abs(returns[returns < 0].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        return {
            "total_return": total_ret,
            "cagr": cagr,
            "sharpe": sharpe,
            "sortino": sortino,
            "max_drawdown": max_dd,
            "calmar": calmar,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
        }

    strat_metrics = _metrics(strat_ret, "equity")
    bench_metrics = _metrics(bench_ret, "bench_equity")

    return {"strategy": strat_metrics, "benchmark": bench_metrics}


def print_performance(perf: dict):
    """Pretty-print performance summary to console."""
    print("\n" + "=" * 72)
    print("  JEDI STRATEGY BACKTEST — PERFORMANCE SUMMARY")
    print("=" * 72)

    headers = ["Metric", "JEDI Strategy", "ITA Benchmark"]
    rows = [
        ("Total Return", f"{perf['strategy']['total_return']:+.2%}", f"{perf['benchmark']['total_return']:+.2%}"),
        ("CAGR", f"{perf['strategy']['cagr']:+.2%}", f"{perf['benchmark']['cagr']:+.2%}"),
        ("Sharpe Ratio", f"{perf['strategy']['sharpe']:.3f}", f"{perf['benchmark']['sharpe']:.3f}"),
        ("Sortino Ratio", f"{perf['strategy']['sortino']:.3f}", f"{perf['benchmark']['sortino']:.3f}"),
        ("Max Drawdown", f"{perf['strategy']['max_drawdown']:.2%}", f"{perf['benchmark']['max_drawdown']:.2%}"),
        ("Calmar Ratio", f"{perf['strategy']['calmar']:.3f}", f"{perf['benchmark']['calmar']:.3f}"),
        ("Win Rate", f"{perf['strategy']['win_rate']:.1%}", f"{perf['benchmark']['win_rate']:.1%}"),
        ("Profit Factor", f"{perf['strategy']['profit_factor']:.3f}", f"{perf['benchmark']['profit_factor']:.3f}"),
    ]

    col_widths = [20, 20, 20]
    header_line = "  ".join(h.ljust(w) for h, w in zip(headers, col_widths))
    print(f"\n  {header_line}")
    print("  " + "-" * sum(col_widths + [4]))

    for label, strat_val, bench_val in rows:
        line = "  ".join(v.ljust(w) for v, w in zip([label, strat_val, bench_val], col_widths))
        print(f"  {line}")

    print("\n" + "=" * 72)


# ---------------------------------------------------------------------------
# 9. PLOTS
# ---------------------------------------------------------------------------

def save_plots(results: pd.DataFrame, output_dir: Path):
    """Generate and save 3 PNG charts to output directory."""
    log.info(f"Saving plots to {output_dir}/")
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig_size = (14, 7)
    dpi = 150

    # --- Plot 1: Equity Curve vs ITA Benchmark ---
    fig, ax = plt.subplots(figsize=fig_size, dpi=dpi)

    ax.plot(results.index, results["equity"], linewidth=1.8,
            color="#1f77b4", label="JEDI Strategy")
    ax.plot(results.index, results["bench_equity"], linewidth=1.5,
            color="#ff7f0e", alpha=0.8, label="ITA Benchmark", linestyle="--")

    ax.set_title("JEDI Strategy — Equity Curve vs ITA Benchmark",
                 fontsize=16, fontweight="bold", pad=15)
    ax.set_ylabel("Portfolio Value (Starting = $1.00)", fontsize=12)
    ax.set_xlabel("")
    ax.legend(fontsize=12, loc="upper left")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    fig.autofmt_xdate(rotation=30)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_dir / "equity_curve.png", bbox_inches="tight")
    plt.close(fig)
    log.info("  Saved equity_curve.png")

    # --- Plot 2: Drawdown Chart ---
    fig, ax = plt.subplots(figsize=fig_size, dpi=dpi)

    ax.fill_between(results.index, results["drawdown"] * 100, 0,
                    alpha=0.5, color="#d62728", label="JEDI Strategy Drawdown")
    ax.plot(results.index, results["bench_drawdown"] * 100,
            linewidth=1.2, color="#ff7f0e", alpha=0.7,
            label="ITA Benchmark Drawdown", linestyle="--")

    ax.axhline(y=CONFIG["drawdown_half_stop"] * 100, color="#e377c2",
               linestyle=":", linewidth=1.5, alpha=0.8, label="-5% Half Stop")
    ax.axhline(y=CONFIG["drawdown_flat_stop"] * 100, color="#7f7f7f",
               linestyle=":", linewidth=1.5, alpha=0.8, label="-10% Flat Stop")

    ax.set_title("JEDI Strategy — Drawdown Analysis",
                 fontsize=16, fontweight="bold", pad=15)
    ax.set_ylabel("Drawdown (%)", fontsize=12)
    ax.set_xlabel("")
    ax.legend(fontsize=11, loc="lower left")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    fig.autofmt_xdate(rotation=30)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_dir / "drawdown.png", bbox_inches="tight")
    plt.close(fig)
    log.info("  Saved drawdown.png")

    # --- Plot 3: Rolling 63-Day Sharpe Comparison ---
    fig, ax = plt.subplots(figsize=fig_size, dpi=dpi)

    roll_window = 63

    strat_rolling_sharpe = (
        results["strategy_return"].rolling(roll_window).mean()
        / results["strategy_return"].rolling(roll_window).std()
    ) * np.sqrt(252)

    bench_rolling_sharpe = (
        results["bench_return"].rolling(roll_window).mean()
        / results["bench_return"].rolling(roll_window).std()
    ) * np.sqrt(252)

    ax.plot(results.index, strat_rolling_sharpe, linewidth=1.5,
            color="#1f77b4", label="JEDI Strategy", alpha=0.9)
    ax.plot(results.index, bench_rolling_sharpe, linewidth=1.5,
            color="#ff7f0e", label="ITA Benchmark", alpha=0.7, linestyle="--")
    ax.axhline(y=0, color="black", linewidth=0.8, alpha=0.5)

    ax.set_title("JEDI Strategy — Rolling 63-Day Sharpe Ratio",
                 fontsize=16, fontweight="bold", pad=15)
    ax.set_ylabel("Annualized Sharpe Ratio", fontsize=12)
    ax.set_xlabel("")
    ax.legend(fontsize=12, loc="upper right")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    fig.autofmt_xdate(rotation=30)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_dir / "rolling_sharpe.png", bbox_inches="tight")
    plt.close(fig)
    log.info("  Saved rolling_sharpe.png")


# ---------------------------------------------------------------------------
# 10. MAIN ENTRYPOINT
# ---------------------------------------------------------------------------

def main():
    """Orchestrate the full JEDI strategy backtest pipeline."""
    print("\n" + "=" * 72)
    print("  JEDI — Political Beta Factor Model for U.S. Defense Equities")
    print("  TAMID Quant @ NYU — Strategy Backtest")
    print("  Data Source: Polymarket (public API, no key required)")
    print("=" * 72 + "\n")

    t_start = time.time()
    config = CONFIG
    output_dir = config["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Step 1: Fetch data ----
    log.info("STEP 1/6: Fetching market data...")
    jedi_data = fetch_jedi_index(
        config["jedi_tickers"], config["start_date"], config["end_date"]
    )
    bench_data = fetch_benchmark(
        config["benchmark_ticker"], config["start_date"], config["end_date"]
    )
    prob_data = fetch_polymarket_probabilities(config)

    # ---- Step 2: Feature engineering ----
    log.info("STEP 2/6: Engineering features...")
    features, target = engineer_features(
        prob_data, jedi_data, config["ffill_limit"]
    )

    # ---- Step 3: Walk-forward model ----
    # Auto-adjust rolling window if dataset is too short
    rolling_window = config["rolling_window"]
    if len(features) < rolling_window + 50:
        rolling_window = max(60, len(features) // 3)
        log.info(f"  Auto-adjusted rolling window: {config['rolling_window']} → {rolling_window} "
                 f"(dataset has {len(features)} observations)")

    log.info("STEP 3/6: Running walk-forward PCA + Elastic Net model...")
    model = JEDIFactorModel(
        window=rolling_window,
        refit_frequency=config["refit_frequency"],
        pca_variance=config["pca_variance_threshold"],
        l1_ratios=config["elastic_net_l1_ratios"],
        cv_splits=config["ts_cv_splits"],
    )
    predictions = model.walk_forward(features, target)

    # ---- Step 4: Signal construction ----
    log.info("STEP 4/6: Constructing trading signals...")
    signals = construct_signals(
        predictions, target,
        threshold=config["signal_threshold"],
        kelly_fraction=config["kelly_fraction"],
        ic_window=config["ic_rolling_window"],
        leverage_cap=config["leverage_cap"],
    )

    # ---- Step 5: Backtest ----
    log.info("STEP 5/6: Running backtest engine...")
    results = run_backtest(
        signals, jedi_data, bench_data,
        dd_half=config["drawdown_half_stop"],
        dd_flat=config["drawdown_flat_stop"],
    )

    # ---- Step 6: Analytics & plots ----
    log.info("STEP 6/6: Computing performance analytics & generating plots...")
    perf = compute_performance(results)
    print_performance(perf)
    save_plots(results, output_dir)

    results_path = output_dir / "backtest_results.csv"
    results.to_csv(results_path)
    log.info(f"  Saved backtest results to {results_path}")

    elapsed = time.time() - t_start
    print(f"\n  Backtest completed in {elapsed:.1f}s")
    print(f"  Output saved to: {output_dir}/")
    print(f"  Files: equity_curve.png, drawdown.png, rolling_sharpe.png, backtest_results.csv\n")


if __name__ == "__main__":
    main()
