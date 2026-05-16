"""
Shared OHLCV data loading — replaces the duplicated _download() and
_synthetic_ohlcv() functions that appear identically in
volume_profile_strategy.py, vah_val_flip_long.py, and
staircase_breakout_strategy.py.

Usage:
    from utils.data_loader import load_ohlcv

    df = load_ohlcv("QQQ", start="2020-01-01", end="2025-01-01")
    # Returns a DataFrame with columns: Open High Low Close Volume
    # Falls back to synthetic GBM data if yfinance is unavailable.
"""

from __future__ import annotations
import numpy as np
import pandas as pd


# ── Synthetic data ────────────────────────────────────────────────────────────

def synthetic_ohlcv(
    start: str,
    end: str,
    freq: str = "D",
    s0: float = 400.0,
    mu: float = 0.10,
    sigma: float = 0.18,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate synthetic OHLCV via geometric Brownian motion.

    Parameters
    ----------
    start, end : ISO date strings
    freq       : pandas frequency string ("D", "4h", "1h", etc.)
    s0         : starting price
    mu         : annual drift
    sigma      : annual volatility
    seed       : random seed for reproducibility

    Returns
    -------
    DataFrame with DatetimeIndex and columns: Open High Low Close Volume
    """
    rng = np.random.default_rng(seed)

    # Infer the number of bars per year for scaling mu/sigma
    freq_hours = pd.tseries.frequencies.to_offset(freq).nanos / 3_600_000_000_000
    bars_per_year = 8_760 / freq_hours  # approximate

    dates  = pd.date_range(start, end, freq=freq)
    n      = len(dates)
    dt     = 1 / bars_per_year
    log_r  = (mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * rng.standard_normal(n)
    close  = s0 * np.exp(np.cumsum(log_r))

    noise      = sigma * np.sqrt(dt)
    high_mult  = np.exp( abs(rng.standard_normal(n)) * noise)
    low_mult   = np.exp(-abs(rng.standard_normal(n)) * noise)
    open_price = np.roll(close, 1)
    open_price[0] = s0

    df = pd.DataFrame({
        "Open"  : open_price,
        "High"  : close * high_mult,
        "Low"   : close * low_mult,
        "Close" : close,
        "Volume": rng.integers(500_000, 5_000_000, size=n).astype(float),
    }, index=dates)

    # Ensure OHLC relationship is valid
    df["High"] = df[["Open", "High", "Close"]].max(axis=1)
    df["Low"]  = df[["Open", "Low",  "Close"]].min(axis=1)
    return df


# ── yfinance download ─────────────────────────────────────────────────────────

def _try_download(ticker: str, start: str, end: str, interval: str = "1d") -> pd.DataFrame | None:
    """Attempt a yfinance download; return None if the library is blocked or fails."""
    try:
        import yfinance as yf
        raw = yf.download(ticker, start=start, end=end, interval=interval,
                          auto_adjust=True, progress=False)
        if raw.empty:
            return None
        # Flatten multi-level columns that yfinance sometimes returns
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        raw = raw[["Open", "High", "Low", "Close", "Volume"]].dropna()
        return raw
    except Exception as exc:
        print(f"[data_loader] yfinance unavailable ({exc}) — using synthetic data")
        return None


# ── CSV loader ────────────────────────────────────────────────────────────────

def _from_csv(path: str) -> pd.DataFrame:
    """
    Load OHLCV from a CSV file.

    Expects columns: Date (or index), Open, High, Low, Close, Volume.
    Handles 'Adj Close' by renaming it to 'Close'.
    """
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    if "Adj Close" in df.columns and "Close" not in df.columns:
        df = df.rename(columns={"Adj Close": "Close"})
    required = {"Open", "High", "Low", "Close", "Volume"}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing columns: {missing}")
    return df[list(required)].dropna()


# ── Public interface ──────────────────────────────────────────────────────────

def load_ohlcv(
    ticker: str,
    start: str,
    end: str,
    interval: str = "1d",
    csv_path: str | None = None,
    synthetic_kwargs: dict | None = None,
) -> pd.DataFrame:
    """
    Load OHLCV data from (in order of preference):
      1. A local CSV file if csv_path is provided
      2. yfinance if available in the environment
      3. Synthetic GBM data as a fallback

    Parameters
    ----------
    ticker           : e.g. "QQQ", "NQ=F"
    start, end       : ISO date strings "YYYY-MM-DD"
    interval         : yfinance interval string ("1d", "1h", etc.)
    csv_path         : optional path to a local OHLCV CSV
    synthetic_kwargs : extra keyword args forwarded to synthetic_ohlcv()

    Returns
    -------
    DataFrame with columns: Open High Low Close Volume
    """
    if csv_path:
        print(f"[data_loader] Loading from CSV: {csv_path}")
        return _from_csv(csv_path)

    df = _try_download(ticker, start, end, interval)
    if df is not None:
        print(f"[data_loader] Downloaded {len(df)} bars for {ticker} from yfinance")
        return df

    # Synthetic fallback
    freq = "D" if interval in ("1d", "1D") else interval.lower().replace("h", "h")
    kwargs = {"freq": freq, **(synthetic_kwargs or {})}
    print(f"[data_loader] Using synthetic GBM data (freq={freq}) — metrics are not meaningful")
    return synthetic_ohlcv(start, end, **kwargs)
