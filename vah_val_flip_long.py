#!/usr/bin/env python3
"""
Strategy 1: VAH→VAL Flip Long Only
====================================

WHAT IS THIS?
─────────────
This script backtests a trading strategy on US stock/futures data.  When you
run it, it will ask you a series of questions (ticker, timeframe, date range,
etc.), then simulate the strategy over that history, print performance stats,
and save a chart image.

CORE CONCEPT (Volume Profile Flip)
────────────────────────────────────
Imagine a price axis running vertically.  A "Volume Profile" is a bar chart
laid on its side showing how much trading happened at each price level over a
given period.  Three levels come out of it:

  VAH  — Value Area High   (top of the zone where 70% of volume traded)
  POC  — Point of Control  (the single price with the MOST volume)
  VAL  — Value Area Low    (bottom of that 70% zone)

The "flip" pattern:
  - Period 1:  VAL ──── POC ──── VAH              (where value was)
  - Period 2:              VAL ──── POC ──── VAH   (value STEPPED UP)
                           ↑
           This new VAL is right where the old VAH was.
           Old resistance just became new support → buy the pullback there.

ENTRY / EXIT LOGIC
───────────────────
  Entry : price pulls back down to the new VAL while we're above the EMA
  Exits : three-stage scale-out
    · 50 % of position sold at POC         (lock in first profit)
    · 70 % of remainder sold at VAH        (lock in more)
    · Rest held to VAH + extension target  (let the winner run)
  Stop  : just below VAL (if support breaks, exit immediately)

TIMEFRAMES SUPPORTED
─────────────────────
  1m  5m  15m  30m   ← intraday  (yfinance history limits apply)
  1h  4h              ← intraday/swing
  1d  1wk             ← daily/weekly

HOW TO RUN
───────────
  python vah_val_flip_long.py                        ← interactive prompts
  python vah_val_flip_long.py --no-prompt            ← use DEFAULT_CONFIG silently
  python vah_val_flip_long.py --ticker NQ=F --interval 4h --lookback_days 300
  python vah_val_flip_long.py --help                 ← all CLI options

INSTALL DEPENDENCIES
─────────────────────
  pip install yfinance pandas numpy matplotlib
"""

import argparse
import sys
import warnings

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec

warnings.filterwarnings("ignore")

# ── Optional: yfinance for live market data ────────────────────────────────────
# Without it the script uses synthetic (randomly generated) price data instead.
# Install with:  pip install yfinance
try:
    import yfinance as yf
    _YF_AVAILABLE = True
except ImportError:
    _YF_AVAILABLE = False


# ══════════════════════════════════════════════════════════════════════════════
#  TIMEFRAME REGISTRY
#  ─────────────────
#  Each entry describes one bar size the strategy can run on.
#
#  ⚙ TO ADD A NEW TIMEFRAME: copy an existing row and adjust the four values:
#      "yf"          → the interval string that yfinance accepts
#      "max_days"    → how many calendar days back Yahoo Finance will give you
#      "bars_per_day"→ roughly how many bars fit in one trading day
#                       (used to estimate how many bars to generate synthetically)
#      "resample"    → set to a pandas offset string if yfinance doesn't offer
#                       this bar size natively (e.g. "4h" resamples from 1H data)
#                       set to None if yfinance serves it directly
#
#  ⚠ yfinance history limits (hard limits enforced by Yahoo):
#      1m            → last 7 days only
#      5m / 15m / 30m → last 60 days
#      60m / 1h       → last 730 days (~2 years)
#      1d and above   → no limit
# ══════════════════════════════════════════════════════════════════════════════
TIMEFRAME_MAP = {
    # key    yf str    max hist  bars/day  resample?
    "1m":  {"yf": "1m",  "max_days": 7,    "bars_per_day": 390, "resample": None},
    "5m":  {"yf": "5m",  "max_days": 60,   "bars_per_day": 78,  "resample": None},
    "15m": {"yf": "15m", "max_days": 60,   "bars_per_day": 26,  "resample": None},
    "30m": {"yf": "30m", "max_days": 60,   "bars_per_day": 13,  "resample": None},
    "1h":  {"yf": "1h",  "max_days": 730,  "bars_per_day": 6,   "resample": None},
    # 4H: yfinance has no native 4H feed, so we download 1H and resample up
    "4h":  {"yf": "1h",  "max_days": 730,  "bars_per_day": 2,   "resample": "4h"},
    "1d":  {"yf": "1d",  "max_days": 3650, "bars_per_day": 1,   "resample": None},
    "1wk": {"yf": "1wk", "max_days": 3650, "bars_per_day": 0.2, "resample": None},
}

# Human-readable descriptions shown in the interactive timeframe menu.
# ⚙ Change the description text if you want different wording in the prompt.
TIMEFRAME_LABELS = {
    "1m":  "1 Minute   — intraday scalping          (max 7 days history)",
    "5m":  "5 Minutes  — intraday short-term        (max 60 days history)",
    "15m": "15 Minutes — intraday medium-term       (max 60 days history)",
    "30m": "30 Minutes — intraday swing             (max 60 days history)",
    "1h":  "1 Hour     — swing trading              (max 730 days history)",
    "4h":  "4 Hours    — position/swing  [resampled from 1H]  (max 730 days)",
    "1d":  "Daily      — position trading           (unlimited history)",
    "1wk": "Weekly     — long-term investing        (unlimited history)",
}

# Ordered list for the numbered menu (displayed top to bottom, finest to coarsest)
# ⚙ Reorder this list to change which timeframe appears first in the menu.
TIMEFRAME_ORDER = ["1m", "5m", "15m", "30m", "1h", "4h", "1d", "1wk"]

# Sensible volume-profile lookback bar counts for each timeframe.
# A "lookback" is the number of bars we include in one volume-profile calculation.
# We want each window to represent roughly one to a few "sessions" of activity.
#
# ⚙ CHANGE THESE if your signals feel too slow (reduce) or too noisy (increase).
#   Rule of thumb: think "how many bars = one trading day?" then multiply by 2-4.
#   Examples:  1H  → 6 bars/day × 3 days = ~18   (we use 16)
#              15m → 26 bars/day × 1.5 days = ~39  (we use 32)
LOOKBACK_DEFAULTS = {
    "1m":  120,   # ≈ 20 min of market time (fast, many signals)
    "5m":  48,    # ≈ 4 hours of market time
    "15m": 32,    # ≈ 1.5 trading days
    "30m": 20,    # ≈ 2 trading days
    "1h":  16,    # ≈ 3 trading days
    "4h":  8,     # ≈ 1.5 trading weeks
    "1d":  20,    # ≈ 4 trading weeks (one month)
    "1wk": 12,    # ≈ 3 months
}


# ══════════════════════════════════════════════════════════════════════════════
#  DEFAULT CONFIGURATION
#  ──────────────────────
#  These are the fallback values used when you run with --no-prompt.
#  In interactive mode the prompts will override them.
#  Each parameter has an explanation and a "try changing it" suggestion below.
# ══════════════════════════════════════════════════════════════════════════════
DEFAULT_CONFIG = {

    # ── DATA ─────────────────────────────────────────────────────────────────
    # ⚙ ticker: any Yahoo Finance symbol
    #   Stocks:  "SPY", "QQQ", "AAPL", "TSLA", "NVDA"
    #   Futures: "NQ=F" (Nasdaq), "ES=F" (S&P 500), "GC=F" (Gold)
    #   ETFs:    "TQQQ" (3× Nasdaq), "SOXL" (3× Semiconductors)
    "ticker": "SPY",

    # ⚙ interval: the bar size to use — see TIMEFRAME_MAP above for all options
    #   Intraday (uses more recent data, smaller time windows):
    #     "1m"  → each bar = 1 minute  (needs frequent re-run, only 7 days)
    #     "5m"  → each bar = 5 minutes
    #     "15m" → each bar = 15 minutes
    #     "30m" → each bar = 30 minutes
    #     "1h"  → each bar = 1 hour    (good balance for swing traders)
    #     "4h"  → each bar = 4 hours   (resampled from 1H)
    #   Swing/Position (uses longer history, bigger picture):
    #     "1d"  → each bar = 1 day     (most common for beginners)
    #     "1wk" → each bar = 1 week
    "interval": "1d",

    # ⚙ start / end: the date range to test over
    #   Format must be "YYYY-MM-DD".
    #   The max lookback depends on the interval (see TIMEFRAME_MAP comments).
    "start": "2022-01-01",
    "end":   "2025-01-01",

    # ─────────────────────────────────────────────────────────────────────────
    # ── VOLUME PROFILE ────────────────────────────────────────────────────────
    # ─────────────────────────────────────────────────────────────────────────

    # ⚙ lookback: number of bars used to calculate each volume profile window.
    #   Think of this as "how much recent history shapes each VAH/VAL/POC level".
    #   · Smaller (e.g. 10): the levels react quickly to recent price action,
    #     more signals but they may be less reliable
    #   · Larger (e.g. 40): levels are based on more history, slower to update,
    #     fewer but potentially stronger signals
    #   If left at 0 the script auto-selects based on the timeframe.
    "lookback": 0,   # 0 = auto-select from LOOKBACK_DEFAULTS

    # ⚙ vp_bins: how many price buckets divide the volume histogram
    #   · 50  → coarse, fast to compute, good for quick testing
    #   · 100 → the default, good balance of detail vs speed
    #   · 200 → very fine-grained, slower to compute, most precise levels
    #   You rarely need to change this unless you notice levels look "steppy".
    "vp_bins": 100,

    # ─────────────────────────────────────────────────────────────────────────
    # ── CAPITAL & RISK ────────────────────────────────────────────────────────
    # ─────────────────────────────────────────────────────────────────────────

    # ⚙ initial_capital: your starting account size in US dollars
    #   The backtest scales all profits/losses relative to this amount.
    #   You can set it to whatever your real account size is.
    "initial_capital": 10_000,

    # ⚙ risk_pct: the fraction of your current account you risk on each trade
    #   0.01 = 1%   (conservative — your account survives a long losing streak)
    #   0.02 = 2%   (standard "professional" rule — default here)
    #   0.05 = 5%   (aggressive — can grow fast but also wipe out faster)
    #   Beginners: start with 1% or less until you trust the strategy.
    "risk_pct": 0.02,

    # ⚙ max_position_pct: caps how much of your account goes into one trade
    #   Even if the position-size formula says "put 80% in", this clamps it.
    #   0.20 = 20%  (conservative, never over-concentrated)
    #   0.40 = 40%  (default — allows meaningful position sizes)
    #   0.80 = 80%  (very aggressive, not recommended without testing)
    "max_position_pct": 0.40,

    # ─────────────────────────────────────────────────────────────────────────
    # ── ENTRY FILTERS ────────────────────────────────────────────────────────
    # ─────────────────────────────────────────────────────────────────────────

    # ⚙ ema_window: the lookback period for the Exponential Moving Average
    #   The EMA is a trend filter — we only take long trades when price is
    #   ABOVE this moving average (i.e. we're in an uptrend).
    #
    #   · 20  → fast/responsive, catches trends early, but more whipsaws
    #   · 50  → the classic medium-term trend filter (default)
    #   · 200 → the classic long-term filter, very strict — few trades but
    #            only in very strong uptrends
    #
    #   For intraday timeframes (1m/5m/15m) consider smaller values (10–20)
    #   because trends on short bars change faster.
    "ema_window": 50,

    # ⚙ flip_tolerance: how closely the new VAL must match the old VAH
    #   to count as a "flip" (expressed as a fraction, not percent).
    #   0.01 = 1%  → very strict, few flips detected, highest quality signals
    #   0.03 = 3%  → the default, moderate
    #   0.06 = 6%  → lenient, many more flips detected, more signals but noisier
    #
    #   For volatile instruments (NQ=F, TQQQ) a larger tolerance is usually needed
    #   because value areas shift more aggressively.
    "flip_tolerance": 0.03,

    # ⚙ entry_buffer: how far ABOVE VAL we still allow an entry
    #   When price pulls back, it may not touch the exact VAL line.
    #   This buffer lets us enter if price is within X% above VAL.
    #   0.002 = 0.2%  → very tight, only enter right at VAL
    #   0.005 = 0.5%  → default, a bit of room
    #   0.010 = 1.0%  → generous, catches pullbacks that stop short of VAL
    "entry_buffer": 0.005,

    # ─────────────────────────────────────────────────────────────────────────
    # ── STOP LOSS ─────────────────────────────────────────────────────────────
    # ─────────────────────────────────────────────────────────────────────────

    # ⚙ stop_pct: how far below VAL to place the stop-loss (as a fraction)
    #   0.005 = 0.5%  → very tight stop, gets stopped out often
    #   0.015 = 1.5%  → default, gives price a little room to breathe
    #   0.030 = 3.0%  → wide stop, rarely stopped out but bigger losses when hit
    #
    #   There is a tradeoff: tighter stops = smaller losses per trade, but more
    #   of them.  Wider stops = fewer but larger losses.
    #   Aim for stop to sit below a meaningful support level, not just a % in the air.
    "stop_pct": 0.015,

    # ─────────────────────────────────────────────────────────────────────────
    # ── EXITS — SCALING OUT IN THREE STAGES ───────────────────────────────────
    # ─────────────────────────────────────────────────────────────────────────
    # The strategy sells the position in pieces rather than all at once.
    # This locks in gains gradually while letting part of the trade run further.
    #
    # Stage 1: Sell at POC (the high-volume price — a natural magnet)
    # Stage 2: Sell more at VAH (top of the value area — natural resistance)
    # Stage 3: Sell the last piece at the extension target (measured move)

    # ⚙ poc_exit_frac: what fraction of the position to sell at the POC
    #   0.33 = sell a third — conservative, holds more for later targets
    #   0.50 = sell half   — the default, balanced
    #   0.75 = sell three-quarters — more conservative, locks in most profit early
    "poc_exit_frac": 0.50,

    # ⚙ vah_exit_frac: what fraction of the REMAINING position to sell at VAH
    #   (applied after the POC exit, so it's a fraction of what's left)
    #   0.50 = half of remainder
    #   0.70 = most of remainder (default)
    #   1.00 = sell everything at VAH, don't hold for the extension
    "vah_exit_frac": 0.70,

    # ⚙ ext_mult: multiplier for the extension target beyond VAH
    #   The extension = VAH + ext_mult × (VAH − POC)
    #   In other words: if VAH is 100 and POC is 95, the range is 5.
    #   ext_mult = 1.0  → target = 100 + 1×5 = 105  (one range above VAH)
    #   ext_mult = 1.5  → target = 100 + 1.5×5 = 107.5
    #   ext_mult = 2.0  → very ambitious target, rarely hit
    #   If you find the extension is rarely reached, try reducing this.
    "ext_mult": 1.0,
}


# ══════════════════════════════════════════════════════════════════════════════
#  DATA LAYER
#  ──────────
#  Functions that fetch or generate price data.
#  You should not need to change anything in this section unless you want to
#  swap in a different data source (e.g. a broker API).
# ══════════════════════════════════════════════════════════════════════════════

def _resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """
    Collapse finer bars into coarser bars (e.g. 1H → 4H).

    Why we need this: yfinance has no native 4H feed, so we download 1H data
    and combine every four consecutive 1H candles into one 4H candle using:
      Open   = the first bar's open  (where the 4H session started)
      High   = the highest high      (peak during the 4H session)
      Low    = the lowest low        (trough during the 4H session)
      Close  = the last bar's close  (where the 4H session ended)
      Volume = sum of all bars       (total activity in the period)

    ⚙ The `rule` parameter is a pandas offset string.
      "4h"  → group into 4-hour blocks
      "2h"  → group into 2-hour blocks
      "1d"  → group into daily blocks
    """
    agg = {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}
    resampled = df.resample(rule, label="right", closed="right").agg(agg)
    resampled.dropna(subset=["Close"], inplace=True)
    return resampled


def _synthetic_ohlcv(ticker: str, n_bars: int, bars_per_day: float,
                     interval: str) -> pd.DataFrame:
    """
    Generate realistic fake OHLCV data for testing when live data is unavailable.

    Uses two well-known statistical models:
      · Geometric Brownian Motion (GBM) — the standard academic model for
        how stock prices move: random up/down steps with a slight upward drift.
      · GARCH-like volatility clustering — volatility is not constant; calm
        periods are followed by volatile ones (and vice versa).  This mimics
        how real markets behave.

    ⚙ Things you could adjust in here (advanced):
      S0        → the starting price level (default 18_000 for NQ-like data)
      mu        → the upward drift per bar (0.0004 ≈ small bullish bias)
      sigma_base→ the base volatility (0.013 = about 1.3% per bar)
    """
    rng = np.random.default_rng(abs(hash(ticker)) % (2**31))

    # ── Starting price level ──────────────────────────────────────────────
    # ⚙ Change S0 to match the price range of your instrument.
    #   18,000 ≈ Nasdaq 100 futures.  Use 400 for SPY, 100 for most stocks.
    S0 = 18_000.0

    # ── Volatility model (GARCH-inspired) ─────────────────────────────────
    # Each bar's vol is 91% of the previous bar's vol plus a small random jolt.
    # This creates realistic "calm → volatile → calm" cycles.
    sigma_base = 0.013    # ⚙ base volatility per bar (try 0.008 for calmer data)
    vols = np.full(n_bars, sigma_base)
    for i in range(1, n_bars):
        vols[i] = 0.91 * vols[i-1] + 0.09 * sigma_base * abs(rng.standard_normal()) + 0.001

    # ── Price path ─────────────────────────────────────────────────────────
    mu = 0.0004    # ⚙ drift per bar; 0 = random walk, >0 = bullish bias
    closes = S0 * np.exp(np.cumsum(rng.standard_normal(n_bars) * vols + mu))

    hl     = closes * vols * 2.5   # high-low range proportional to volatility
    highs  = closes + hl * rng.uniform(0.3, 0.7, n_bars)
    lows   = closes - hl * rng.uniform(0.3, 0.7, n_bars)
    opens  = np.roll(closes, 1); opens[0] = S0
    vols_v = rng.integers(1_000_000, 5_000_000, n_bars)

    # Map our interval key to a pandas date frequency string
    freq_map = {
        "1m": "1min", "5m": "5min", "15m": "15min", "30m": "30min",
        "1h": "1h",   "4h": "4h",   "1d": "B",       "1wk": "W",
    }
    freq  = freq_map.get(interval, "B")
    dates = pd.date_range(end=pd.Timestamp.now().normalize(), periods=n_bars, freq=freq)

    df = pd.DataFrame(
        {"Open": opens, "High": highs, "Low": lows, "Close": closes, "Volume": vols_v},
        index=dates,
    )
    df.index.name = "Date"
    return df


def _download(ticker: str, start: str, end: str, interval: str) -> pd.DataFrame:
    """
    Fetch OHLCV data from Yahoo Finance, resampling if needed, or fall back
    to synthetic data if yfinance is unavailable or returns too few bars.
    """
    tf           = TIMEFRAME_MAP.get(interval, TIMEFRAME_MAP["1d"])
    yf_interval  = tf["yf"]
    resample_to  = tf["resample"]
    bars_per_day = tf["bars_per_day"]

    if _YF_AVAILABLE:
        try:
            print(f"  Downloading {ticker} [{interval}] from {start} to {end} …")
            raw = yf.download(
                ticker, start=start, end=end,
                interval=yf_interval,
                auto_adjust=True,   # adjust for stock splits and dividends
                progress=False,
            )
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.droplevel(1)
            raw.dropna(inplace=True)

            if len(raw) > 20:
                if resample_to:
                    raw = _resample_ohlcv(raw, resample_to)
                    print(f"  Resampled to {resample_to}: {len(raw)} bars.")
                else:
                    print(f"  Downloaded {len(raw)} bars.")
                return raw
            else:
                print(f"  Too few bars ({len(raw)}).  "
                      f"The {interval} feed only goes back {tf['max_days']} days.")
        except Exception as e:
            print(f"  Yahoo Finance error ({e}) — using synthetic data.")

    print(f"  Generating synthetic OHLCV for {ticker} [{interval}] …")
    try:
        day_span = (pd.Timestamp(end) - pd.Timestamp(start)).days
    except Exception:
        day_span = 365
    n_bars = max(int(day_span * bars_per_day), 200)
    df = _synthetic_ohlcv(ticker, n_bars, bars_per_day, interval)
    if resample_to:
        df = _resample_ohlcv(df, resample_to)
    return df


# ══════════════════════════════════════════════════════════════════════════════
#  VOLUME PROFILE CALCULATOR
#  ──────────────────────────
#  Given a slice of OHLCV data, returns (POC, VAL, VAH) and the raw histogram.
#
#  HOW IT WORKS:
#  1. Divide the total price range (lowest Low to highest High) into N equal
#     "buckets" (controlled by cfg["vp_bins"]).
#  2. For each candle, spread its volume evenly across every bucket that its
#     high-low range overlaps.  A candle from 100–110 spreads volume across
#     every bucket between 100 and 110 proportionally.
#  3. POC = the bucket with the MOST accumulated volume.
#  4. Grow a window outward from the POC (always expanding toward whichever
#     side adds more volume) until 70% of total volume is included.
#     The window edges become VAL (bottom) and VAH (top).
#
#  You do not need to change this function.
# ══════════════════════════════════════════════════════════════════════════════

def compute_vp(ohlcv: pd.DataFrame, bins: int = 100):
    """
    Returns: (poc, val, vah, price_midpoints_array, volume_per_bin_array)
    """
    lo = float(ohlcv["Low"].min())
    hi = float(ohlcv["High"].max())

    if hi <= lo + 1e-8:
        # Flat/no-range data — all three levels collapse to the midpoint
        mid = (hi + lo) / 2
        return mid, mid, mid, np.array([mid]), np.array([1.0])

    # Build the histogram buckets
    edges = np.linspace(lo, hi, bins + 1)   # N+1 edges → N buckets
    mids  = (edges[:-1] + edges[1:]) / 2    # centre price of each bucket
    vol   = np.zeros(bins)                   # volume per bucket

    for k in range(len(ohlcv)):
        bar_hi = float(ohlcv["High"].iloc[k])
        bar_lo = float(ohlcv["Low"].iloc[k])
        bar_v  = float(ohlcv["Volume"].iloc[k])
        rng    = bar_hi - bar_lo

        if rng < 1e-10:
            # Zero-range (doji) bar: all volume goes to one bucket
            idx = min(int((bar_lo - lo) / (hi - lo) * bins), bins - 1)
            vol[idx] += bar_v
        else:
            # Distribute volume proportionally across overlapping buckets.
            # overlap[i] = how many price units of bucket i sit inside [bar_lo, bar_hi]
            overlap = np.maximum(
                0.0,
                np.minimum(edges[1:], bar_hi) - np.maximum(edges[:-1], bar_lo),
            )
            vol += bar_v * overlap / rng

    # POC: bucket with the highest volume
    poc_idx = int(np.argmax(vol))
    poc     = mids[poc_idx]

    # Expand outward from POC to capture 70% of total volume
    target    = vol.sum() * 0.70   # ⚙ 0.70 = the standard "70% value area"
    li = hi_i = poc_idx            # expand these indices outward
    acc       = vol[poc_idx]

    while acc < target:
        # Which direction adds more volume — up or down?
        add_lo = vol[li   - 1] if li   > 0        else -1.0
        add_hi = vol[hi_i + 1] if hi_i < bins - 1 else -1.0
        if add_lo < 0 and add_hi < 0:
            break
        if add_hi >= add_lo:
            hi_i += 1; acc += vol[hi_i]
        else:
            li   -= 1; acc += vol[li]

    return poc, mids[li], mids[hi_i], mids, vol


# ══════════════════════════════════════════════════════════════════════════════
#  BACKTEST ENGINE
#  ────────────────
#  Simulates trading the strategy bar by bar through historical data.
#  At each bar we only "know" what happened up to that point — no cheating
#  by looking at future prices (no lookahead bias).
#
#  You should not need to change this function to tune the strategy.
#  Tune the parameters in DEFAULT_CONFIG or via the interactive prompts.
# ══════════════════════════════════════════════════════════════════════════════

def run_backtest(cfg: dict):
    """
    Simulate the VAH→VAL flip strategy and return all trade/equity data.

    Returns
    -------
    raw        : the raw OHLCV DataFrame
    trades_df  : one row per completed trade (entry, exit, pnl, type)
    equity_df  : portfolio value at every bar
    levels_df  : VAH/POC/VAL/EMA at every bar (for the chart)
    entry_df   : entry timestamps and prices (for chart markers)
    exit_df    : exit timestamps, prices, and exit type (for chart markers)
    flip_df    : bars where a flip was detected (for chart markers)
    """
    raw      = _download(cfg["ticker"], cfg["start"], cfg["end"], cfg["interval"])
    lookback = cfg["lookback"]
    bins     = cfg["vp_bins"]
    tol      = cfg["flip_tolerance"]
    buf      = cfg["entry_buffer"]
    stop_p   = cfg["stop_pct"]
    ema_win  = cfg["ema_window"]
    capital  = float(cfg["initial_capital"])

    # ── Trend filter: Exponential Moving Average ──────────────────────────
    # ewm(span=N) gives about the same weight distribution as a simple N-period EMA.
    # adjust=False means the formula uses a recursive calculation (standard for EMA).
    ema = raw["Close"].ewm(span=ema_win, adjust=False).mean()

    # ── Trade tracking variables ──────────────────────────────────────────
    # These hold the state of the currently open position (if any).
    in_pos     = False   # True while we hold an open position
    shares     = 0.0     # units (shares/contracts) currently held
    avg_px     = 0.0     # average entry price
    cost       = 0.0     # cash tied up (shares × avg_px, i.e. what we "spent")
    entry_stop = 0.0     # stop-loss price for the current trade
    entry_poc  = 0.0     # first partial-exit target (POC)
    entry_vah  = 0.0     # second partial-exit target (VAH)
    entry_ext  = 0.0     # final exit target (extension)
    poc_done   = False   # have we hit the POC target yet?
    vah_done   = False   # have we hit the VAH target yet?

    equity_records = []
    level_records  = []
    trade_records  = []
    entry_marks    = []
    exit_marks     = []
    flip_marks     = []

    # We need 2× lookback bars of history before we can compute TWO consecutive
    # volume profiles (current period vs previous period) for the flip check.
    min_i = max(lookback * 2, ema_win + 5)

    for i in range(min_i, len(raw)):
        date  = raw.index[i]
        close = float(raw["Close"].iloc[i])
        high  = float(raw["High"].iloc[i])
        low   = float(raw["Low"].iloc[i])

        # ── Volume profiles for this bar ──────────────────────────────────
        # Current period: the most recent `lookback` bars
        poc_c, val_c, vah_c, _, _ = compute_vp(raw.iloc[i - lookback : i], bins)
        # Previous period: the `lookback` bars just before the current period
        poc_p, val_p, vah_p, _, _ = compute_vp(raw.iloc[i - lookback*2 : i - lookback], bins)

        # ── Flip check ────────────────────────────────────────────────────
        # The "flip" is confirmed when both conditions are true:
        #   a) The value area has moved up (current VAL > previous VAL)
        #   b) The new VAL is close to (or above) the old VAH
        #      — meaning the old ceiling is now the new floor
        val_above_prev_val = val_c > val_p
        near_prev_vah = (
            (vah_p > 0 and abs(val_c - vah_p) / vah_p < tol)
            or val_c >= vah_p
        )
        flip    = val_above_prev_val and near_prev_vah
        uptrend = close > float(ema.iloc[i])   # trend filter: price must be above EMA

        # ── Level calculations ────────────────────────────────────────────
        # Stop-loss: just below VAL — if price falls here, the flip level failed
        stop_lvl = val_c * (1 - stop_p)
        # Extension target: one measured move above VAH
        ext_lvl  = vah_c + cfg["ext_mult"] * (vah_c - poc_c)

        level_records.append({
            "date": date, "poc": poc_c, "val": val_c, "vah": vah_c,
            "vah_prev": vah_p, "flip": flip, "ema": float(ema.iloc[i]),
        })
        if flip:
            flip_marks.append({"date": date, "level": val_c})

        # ── EXIT CHECKS (evaluated first so we don't enter AND exit same bar) ─
        if in_pos:
            if low <= entry_stop:
                # Stop-loss triggered: the support level broke down
                pnl     = (entry_stop - avg_px) * shares
                capital += cost + pnl
                trade_records.append({"date": date, "type": "stop",
                                      "entry": avg_px, "exit": entry_stop, "pnl": pnl})
                exit_marks.append({"date": date, "price": entry_stop, "type": "stop"})
                in_pos = False; shares = cost = 0.0
                poc_done = vah_done = False

            elif not poc_done and high >= entry_poc:
                # Stage 1 exit: sell `poc_exit_frac` of position at POC
                # POC is the highest-volume price — a strong gravitational magnet
                cs       = shares * cfg["poc_exit_frac"]
                pnl_p    = (entry_poc - avg_px) * cs
                capital += cs * avg_px + pnl_p
                cost    -= cs * avg_px
                shares  -= cs
                poc_done = True
                exit_marks.append({"date": date, "price": entry_poc, "type": "poc"})

            elif poc_done and not vah_done and high >= entry_vah:
                # Stage 2 exit: sell `vah_exit_frac` of remaining position at VAH
                # VAH is the top of the value area — natural resistance
                cs       = shares * cfg["vah_exit_frac"]
                pnl_p    = (entry_vah - avg_px) * cs
                capital += cs * avg_px + pnl_p
                cost    -= cs * avg_px
                shares  -= cs
                vah_done = True
                exit_marks.append({"date": date, "price": entry_vah, "type": "vah"})

            elif vah_done and shares > 0 and high >= entry_ext:
                # Stage 3 exit: sell the remaining "runner" at the extension target
                pnl     = (entry_ext - avg_px) * shares
                capital += cost + pnl
                trade_records.append({"date": date, "type": "target",
                                      "entry": avg_px, "exit": entry_ext, "pnl": pnl})
                exit_marks.append({"date": date, "price": entry_ext, "type": "ext"})
                in_pos = False; shares = cost = 0.0
                poc_done = vah_done = False

        # ── ENTRY CHECK ───────────────────────────────────────────────────
        # Buy when ALL four conditions are met:
        #   1. No open position (only one trade at a time)
        #   2. A VAH→VAL flip is active on this bar
        #   3. Price is above the EMA (we're in an uptrend)
        #   4. This bar's LOW touched the VAL zone (the pullback arrived)
        entry_px = val_c * (1 + buf)   # VAL + a small buffer above it
        if (
            not in_pos
            and flip
            and uptrend
            and low  <= entry_px      # bar dipped into the entry zone
            and close > stop_lvl      # but didn't already break through the stop
        ):
            # ── Position sizing: fixed-fraction risk ───────────────────────
            # We risk exactly `risk_pct` of current capital.
            # The number of units = (dollars at risk) / (distance from entry to stop)
            fill_px     = min(close, entry_px)        # simulate limit-order fill
            risk_cash   = capital * cfg["risk_pct"]
            risk_per_sh = max(fill_px - stop_lvl, 1e-6)
            n_sh        = risk_cash / risk_per_sh
            # Cap the order so we never deploy more than max_position_pct of capital
            order_cash  = min(n_sh * fill_px, capital * cfg["max_position_pct"])

            if order_cash >= 50 and capital >= order_cash:
                shares     = n_sh
                avg_px     = fill_px
                cost       = order_cash
                entry_stop = stop_lvl
                entry_poc  = poc_c
                entry_vah  = vah_c
                entry_ext  = ext_lvl
                poc_done   = vah_done = False
                in_pos     = True
                capital   -= order_cash
                entry_marks.append({"date": date, "price": fill_px})

        # Record portfolio value = cash + mark-to-market value of open position
        equity_records.append({"date": date, "equity": capital + shares * close})

    # Close any still-open position at the last bar's price (mark to market)
    if in_pos and shares > 0:
        lp  = float(raw["Close"].iloc[-1])
        pnl = (lp - avg_px) * shares
        capital += cost + pnl
        trade_records.append({"date": raw.index[-1], "type": "expired",
                               "entry": avg_px, "exit": lp, "pnl": pnl})

    equity_df = pd.DataFrame(equity_records).set_index("date")
    levels_df = pd.DataFrame(level_records).set_index("date")
    trades_df = (pd.DataFrame(trade_records)
                 if trade_records else pd.DataFrame(columns=["pnl", "type"]))
    entry_df  = pd.DataFrame(entry_marks) if entry_marks else pd.DataFrame()
    exit_df   = pd.DataFrame(exit_marks)  if exit_marks  else pd.DataFrame()
    flip_df   = pd.DataFrame(flip_marks)  if flip_marks  else pd.DataFrame()

    return raw, trades_df, equity_df, levels_df, entry_df, exit_df, flip_df


# ══════════════════════════════════════════════════════════════════════════════
#  PERFORMANCE STATISTICS
#  ────────────────────────
#  Summarises how the backtest performed with standard finance metrics.
#  You do not need to change this section.
#
#  GLOSSARY:
#  Total Return    → (final_equity − start_equity) / start_equity × 100
#  Annualised Ret  → the equivalent yearly return if the test ran for exactly 1 year
#  Max Drawdown    → the worst peak-to-trough drop in equity during the test
#  Sharpe Ratio    → return per unit of risk; above 1.0 is good, above 2.0 is great
#  Win Rate        → percentage of closed trades that made money
# ══════════════════════════════════════════════════════════════════════════════

def compute_stats(equity_df: pd.DataFrame, initial_capital: float,
                  trades_df: pd.DataFrame) -> dict:
    final   = float(equity_df["equity"].iloc[-1])
    total_r = (final - initial_capital) / initial_capital * 100
    n_days  = max((equity_df.index[-1] - equity_df.index[0]).days, 1)
    ann_r   = ((final / initial_capital) ** (365 / n_days) - 1) * 100

    roll_max = equity_df["equity"].cummax()
    max_dd   = float(((equity_df["equity"] - roll_max) / roll_max * 100).min())

    daily_r = equity_df["equity"].pct_change().dropna()
    sharpe  = daily_r.mean() / daily_r.std() * (252 ** 0.5) if daily_r.std() > 0 else 0.0

    if not trades_df.empty and "pnl" in trades_df.columns:
        wins     = int((trades_df["pnl"] > 0).sum())
        total_t  = len(trades_df)
        win_rate = wins / total_t * 100 if total_t else 0.0
        avg_win  = float(trades_df.loc[trades_df["pnl"] > 0,  "pnl"].mean() or 0)
        avg_loss = float(trades_df.loc[trades_df["pnl"] <= 0, "pnl"].mean() or 0)
    else:
        total_t = wins = 0
        win_rate = avg_win = avg_loss = 0.0

    return {
        "Total Return":  f"{total_r:+.1f}%",
        "Ann. Return":   f"{ann_r:+.1f}%",
        "Max Drawdown":  f"{max_dd:.1f}%",
        "Sharpe":        f"{sharpe:.2f}",
        "Trades":        str(total_t),
        "Win Rate":      f"{win_rate:.0f}%",
        "Avg Win":       f"${avg_win:,.0f}",
        "Avg Loss":      f"${avg_loss:,.0f}",
        "Final Equity":  f"${final:,.0f}",
    }


# ══════════════════════════════════════════════════════════════════════════════
#  CHART
#  ──────
#  Produces a 4-panel figure and saves it as a PNG file.
#
#  ⚙ Things you can change here:
#    · `out` parameter in plot_results() → the output filename
#    · Colour constants (BG, BLUE, RED, …) → visual theme
#    · figsize=(20, 15) → make the chart bigger/smaller
#    · height_ratios=[3, 1, 1, 0.55] → relative heights of the 4 panels
# ══════════════════════════════════════════════════════════════════════════════

# ⚙ COLOUR PALETTE — change these hex codes to restyle the whole chart
BG, GRID, FG   = "#0d1117", "#21262d", "#e6edf3"   # background, grid, foreground text
BLUE, RED, ORG = "#58a6ff", "#f85149", "#ffa657"   # price line, stop/loss, POC/warning
GRN,  PRP, YLW = "#3fb950", "#bc8cff", "#e3b341"   # profit/VAH, extension, flip zone/EMA


def _style(ax):
    """Apply dark-theme styling to an axes panel."""
    ax.set_facecolor(BG)
    ax.tick_params(colors=FG, labelsize=8)
    for s in ax.spines.values():
        s.set_edgecolor(GRID)
    ax.xaxis.label.set_color(FG)
    ax.yaxis.label.set_color(FG)
    ax.title.set_color(FG)
    ax.grid(color=GRID, alpha=0.6, lw=0.5)


def plot_results(raw, trades_df, equity_df, levels_df, entry_df, exit_df,
                 flip_df, cfg: dict, stats: dict, out: str = "flip_long_results.png"):
    """
    4-panel chart:
      Panel 1 — Price with VAH/POC/VAL lines, flip zones, entry/exit markers
      Panel 2 — Equity curve
      Panel 3 — Drawdown (how far below the peak equity fell at each point)
      Panel 4 — Stats summary table
    """
    # ⚙ figsize=(width, height) in inches — increase for a larger image
    fig = plt.figure(figsize=(20, 15), facecolor=BG)
    # ⚙ height_ratios controls how tall each panel is relative to the others
    gs  = GridSpec(4, 1, figure=fig, height_ratios=[3, 1, 1, 0.55], hspace=0.42)
    ax_p, ax_eq, ax_dd, ax_st = [fig.add_subplot(gs[i]) for i in range(4)]
    for ax in (ax_p, ax_eq, ax_dd, ax_st):
        _style(ax)

    # ── Panel 1: Price ────────────────────────────────────────────────────
    sl = raw.loc[levels_df.index]
    ax_p.plot(sl.index, sl["Close"], color=BLUE, lw=1.0, label="Price")
    ax_p.plot(levels_df.index, levels_df["ema"],
              color=YLW, lw=0.9, ls="--", alpha=0.7,
              label=f"EMA({cfg['ema_window']}) — trend filter")

    # Faint green band showing the value area (VAL to VAH) at each point in time
    ax_p.fill_between(levels_df.index, levels_df["val"], levels_df["vah"],
                      alpha=0.07, color=GRN)
    ax_p.plot(levels_df.index, levels_df["val"], color=RED, lw=0.8, ls="--",
              alpha=0.9, label="VAL")
    ax_p.plot(levels_df.index, levels_df["poc"], color=ORG, lw=0.8, ls="-",
              alpha=0.9, label="POC")
    ax_p.plot(levels_df.index, levels_df["vah"], color=GRN, lw=0.8, ls="--",
              alpha=0.9, label="VAH")

    # Yellow bands mark bars where a flip was detected
    for _, r in levels_df[levels_df["flip"]].iterrows():
        ax_p.axhspan(r["val"] * 0.998, r["val"] * 1.002, alpha=0.18, color=YLW)

    if not entry_df.empty:
        ax_p.scatter(entry_df["date"], entry_df["price"],
                     marker="^", s=80, color=GRN, zorder=5,
                     edgecolors="white", lw=0.4, label="Long entry")
    if not exit_df.empty:
        ec = {"stop": RED, "poc": ORG, "vah": GRN, "ext": PRP}
        for _, r in exit_df.iterrows():
            ax_p.scatter(r["date"], r["price"], marker="D", s=55, zorder=5,
                         color=ec.get(r["type"], FG), edgecolors="white", lw=0.4)

    ax_p.legend(
        handles=[
            mpatches.Patch(color=BLUE, label="Price"),
            mpatches.Patch(color=YLW,  label=f"EMA({cfg['ema_window']}) trend filter"),
            mpatches.Patch(color=RED,  label="VAL"), mpatches.Patch(color=ORG, label="POC"),
            mpatches.Patch(color=GRN,  label="VAH"),
            mpatches.Patch(color=YLW,  alpha=0.4, label="Flip zone"),
            mpatches.Patch(color=GRN,  label="▲ Long entry"),
            mpatches.Patch(color=RED,  label="◆ Stop"), mpatches.Patch(color=PRP, label="◆ Extension"),
        ],
        loc="upper left", fontsize=7, facecolor=BG, labelcolor=FG, framealpha=0.8, ncol=3,
    )
    ax_p.set_title(
        f"{cfg['ticker']}  [{cfg['interval']}]  ·  VAH→VAL Flip Long  ·  "
        f"{cfg['start']} → {cfg['end']}",
        fontsize=11, fontweight="bold",
    )
    ax_p.set_ylabel("Price ($)")

    # ── Panel 2: Equity curve ─────────────────────────────────────────────
    eq = equity_df["equity"]
    ax_eq.plot(eq.index, eq, color=GRN, lw=1.5)
    ax_eq.axhline(cfg["initial_capital"], color=FG, lw=0.7, ls="--", alpha=0.3)
    ax_eq.fill_between(eq.index, cfg["initial_capital"], eq,
                       where=eq >= cfg["initial_capital"], color=GRN, alpha=0.12)
    ax_eq.fill_between(eq.index, cfg["initial_capital"], eq,
                       where=eq <  cfg["initial_capital"], color=RED, alpha=0.12)
    ax_eq.set_title("Equity Curve", fontsize=10)
    ax_eq.set_ylabel("Equity ($)")

    # ── Panel 3: Drawdown ─────────────────────────────────────────────────
    # Drawdown % = how far below the running all-time-high equity fell
    rm  = eq.cummax()
    ddp = (eq - rm) / rm * 100
    ax_dd.fill_between(ddp.index, ddp, 0, color=RED, alpha=0.55)
    ax_dd.plot(ddp.index, ddp, color=RED, lw=0.8)
    ax_dd.set_title("Drawdown (%)", fontsize=10)
    ax_dd.set_ylabel("DD %")

    # ── Panel 4: Stats table ──────────────────────────────────────────────
    ax_st.axis("off")
    tbl = ax_st.table(cellText=[list(stats.values())], colLabels=list(stats.keys()),
                      cellLoc="center", loc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(9); tbl.scale(1, 2.2)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_facecolor("#161b22" if r == 0 else BG)
        cell.set_edgecolor(GRID); cell.set_text_props(color=FG)

    plt.suptitle(
        "Strategy 1: VAH→VAL Flip Long  —  Buy the level that was once resistance, now confirmed support",
        fontsize=11, color=FG, y=1.005, fontweight="bold",
    )
    # ⚙ Change dpi=150 to dpi=300 for a higher-resolution image file
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
    print(f"  Chart saved → {out}")
    plt.show()


# ══════════════════════════════════════════════════════════════════════════════
#  INTERACTIVE SETUP PROMPT
#  ─────────────────────────
#  When you run the script without arguments this function walks you through
#  all settings one question at a time.  Just press Enter to accept the
#  default shown in [brackets].
#
#  ⚙ To change what value is shown as the default in each prompt, edit the
#    DEFAULT_CONFIG dictionary near the top of this file.
# ══════════════════════════════════════════════════════════════════════════════

def _ask(prompt: str, default, cast=str, valid=None, required=False):
    """
    Show a prompt, wait for input, and return the validated answer.

    Parameters
    ----------
    prompt   : the text shown to the user
    default  : value returned if the user presses Enter with no input.
               Ignored when required=True — the user MUST type something.
    cast     : type to convert the raw string to (str / int / float)
    valid    : if given, input must be one of these values
    required : if True, pressing Enter without typing is not allowed
    """
    while True:
        try:
            raw = input(prompt).strip()
        except EOFError:
            # stdin was closed (e.g. piped input ran out) — use the default
            print(f"  (no input — using default: {default})")
            return default

        if raw == "":
            if required:
                print(f"  ✗  This field is required. Please type a value.")
                continue
            return default

        try:
            value = cast(raw)
        except (ValueError, TypeError):
            print(f"  ✗  Please enter a valid {cast.__name__}.")
            continue

        if valid is not None and value not in valid:
            print(f"  ✗  Enter a number between {min(valid)} and {max(valid)}.")
            continue

        return value


def prompt_config() -> dict:
    """
    Interactive setup wizard.  Returns a complete config dict.

    ⚙ TO ADD A NEW QUESTION: call _ask() with your prompt and add the result
      to the `cfg` dict returned at the bottom.  Then use the key in run_backtest().
    """
    cfg = DEFAULT_CONFIG.copy()

    print()
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║   VAH → VAL Flip Long   ·   Interactive Setup               ║")
    print("║   Type your answer and press Enter.                         ║")
    print("║   Fields marked [default] accept Enter to keep that value.  ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print()

    # ── Ticker ────────────────────────────────────────────────────────────
    # ⚙ Any Yahoo Finance symbol works here.
    #   US stocks: AAPL, MSFT, NVDA, TSLA, SPY, QQQ
    #   Futures  : NQ=F (Nasdaq), ES=F (S&P 500), GC=F (Gold), CL=F (Oil)
    #   If yfinance can't find it you'll get synthetic data instead.
    print("  Ticker symbol — any Yahoo Finance symbol.")
    print("  Examples: SPY  QQQ  AAPL  NQ=F  ES=F  GC=F")
    cfg["ticker"] = _ask(f"  Ticker [{cfg['ticker']}]: ",
                         default=cfg["ticker"]).upper()
    print()

    # ── Timeframe ─────────────────────────────────────────────────────────
    # ⚙ The timeframe controls the bar/candle size the strategy runs on.
    #   Intraday (1m – 4h): more bars, faster signals, limited history window.
    #   Daily / Weekly    : fewer bars, bigger-picture signals, any date range.
    #
    #   Tip: start with 1d (daily) to understand the strategy, then explore
    #   intraday timeframes once you're comfortable with the results.
    print("  ┌────┬──────┬───────────────────────────────────────────────────────┐")
    print("  │  # │  TF  │ Description                                           │")
    print("  ├────┼──────┼───────────────────────────────────────────────────────┤")
    for idx, key in enumerate(TIMEFRAME_ORDER, start=1):
        label = TIMEFRAME_LABELS[key]
        marker = " ◀ default" if key == cfg["interval"] else ""
        print(f"  │ {idx:2d} │ {key:4s} │ {label:<49}{marker}")
    print("  └────┴──────┴───────────────────────────────────────────────────────┘")
    print()

    # required=True means the user MUST type a number — pressing Enter alone
    # is rejected and the prompt repeats.  The default is only used if stdin
    # closes unexpectedly (e.g. piped input).
    default_tf_idx = TIMEFRAME_ORDER.index(cfg["interval"]) + 1
    tf_idx = _ask(
        f"  Type a number (1–{len(TIMEFRAME_ORDER)}) and press Enter: ",
        default=default_tf_idx, cast=int,
        valid=list(range(1, len(TIMEFRAME_ORDER) + 1)),
        required=True,
    )
    cfg["interval"] = TIMEFRAME_ORDER[tf_idx - 1]
    tf_info = TIMEFRAME_MAP[cfg["interval"]]
    print(f"  ✓  Timeframe: {cfg['interval']}  (max history: {tf_info['max_days']} days)")
    print()

    # ── History length ────────────────────────────────────────────────────
    # ⚙ How many calendar days of data to download.
    #   More data → more trades in the backtest → more reliable stats.
    #   But intraday feeds have hard limits (see the timeframe table above).
    #   If you request more than the limit yfinance silently returns less.
    max_days     = tf_info["max_days"]
    default_days = min(365, max_days)
    print(f"  Days of history to fetch.  Max for {cfg['interval']}: {max_days} days.")
    days = _ask(f"  History in days [{default_days}]: ",
                default=default_days, cast=int)
    days = min(days, max_days)   # silently clamp to the feed limit

    end_dt   = pd.Timestamp.now().normalize()
    start_dt = end_dt - pd.Timedelta(days=days)
    cfg["start"] = start_dt.strftime("%Y-%m-%d")
    cfg["end"]   = end_dt.strftime("%Y-%m-%d")
    print(f"  → Date range: {cfg['start']} → {cfg['end']}")
    print()

    # ── Starting capital ──────────────────────────────────────────────────
    # ⚙ Set this to your real account size for realistic position-sizing results.
    #   The strategy risks a fixed % of capital per trade, so all $ amounts
    #   in the results scale proportionally with this number.
    print("  Starting capital in USD (set to your real account size).")
    cfg["initial_capital"] = _ask(
        f"  Capital [${cfg['initial_capital']:,.0f}]: ",
        default=cfg["initial_capital"], cast=float,
    )
    print()

    # ── Risk per trade ────────────────────────────────────────────────────
    # ⚙ The fraction of your current account you're willing to lose on one trade.
    #   0.01 = 1%  →  conservative, survives long losing streaks (recommended for beginners)
    #   0.02 = 2%  →  the industry "standard"
    #   0.05 = 5%  →  aggressive, can blow up quickly
    print("  Risk per trade as a decimal (0.01 = 1%, 0.02 = 2%).")
    print("  Beginners: use 0.01 until you've studied the results carefully.")
    cfg["risk_pct"] = _ask(
        f"  Risk per trade [{cfg['risk_pct']}]: ",
        default=cfg["risk_pct"], cast=float,
    )
    print()

    # ── Advanced parameters ───────────────────────────────────────────────
    # ⚙ By default we skip these and use the values from DEFAULT_CONFIG.
    #   Type 'y' at the prompt to configure them one by one.
    print("  Advanced parameters (EMA window, flip tolerance, stop %, exits).")
    print("  Press Enter to use the defaults from DEFAULT_CONFIG, or type 'y' to configure.")
    adv = _ask("  Configure advanced parameters? [n]: ", default="n").lower()

    if adv == "y":
        print()
        # ── EMA window ────────────────────────────────────────────────────
        # ⚙ Trend filter.  Larger = only trade in strong sustained uptrends.
        #   Smaller = trade sooner in uptrends but more false signals.
        print("  EMA window (trend filter).  50 = daily standard, 20 = faster.")
        cfg["ema_window"] = _ask(
            f"  EMA window [{cfg['ema_window']}]: ",
            default=cfg["ema_window"], cast=int,
        )

        # ── Flip tolerance ────────────────────────────────────────────────
        # ⚙ How close the new VAL must be to the old VAH to count as a flip.
        #   0.01 = very strict (few signals), 0.05 = lenient (many signals).
        print("  Flip tolerance as a decimal (0.03 = 3%).  More = more signals but noisier.")
        cfg["flip_tolerance"] = _ask(
            f"  Flip tolerance [{cfg['flip_tolerance']}]: ",
            default=cfg["flip_tolerance"], cast=float,
        )

        # ── Stop % ────────────────────────────────────────────────────────
        # ⚙ How far below VAL to place the stop-loss.
        #   Tighter (0.005) = small losses but stopped out more often.
        #   Wider   (0.03)  = fewer stop-outs but bigger losses when hit.
        print("  Stop % below VAL as a decimal (0.015 = 1.5%).")
        cfg["stop_pct"] = _ask(
            f"  Stop % [{cfg['stop_pct']}]: ",
            default=cfg["stop_pct"], cast=float,
        )

        # ── POC exit fraction ─────────────────────────────────────────────
        # ⚙ What fraction of the position to sell when price reaches the POC.
        #   0.50 = sell half there, keep the rest running.
        print("  Fraction to sell at POC (0.50 = 50%).")
        cfg["poc_exit_frac"] = _ask(
            f"  POC exit fraction [{cfg['poc_exit_frac']}]: ",
            default=cfg["poc_exit_frac"], cast=float,
        )

        # ── VAH exit fraction ─────────────────────────────────────────────
        # ⚙ What fraction of the REMAINING position to sell at VAH.
        #   Applied after the POC exit, so 0.70 = 70% of whatever is left.
        print("  Fraction of remainder to sell at VAH (0.70 = 70% of what's left).")
        cfg["vah_exit_frac"] = _ask(
            f"  VAH exit fraction [{cfg['vah_exit_frac']}]: ",
            default=cfg["vah_exit_frac"], cast=float,
        )

        # ── Extension multiplier ──────────────────────────────────────────
        # ⚙ The final target = VAH + ext_mult × (VAH − POC).
        #   1.0 = one measured move above VAH (the default).
        #   1.5 = 1.5× the measured move (more ambitious, hit less often).
        print("  Extension multiplier.  Target = VAH + N × (VAH − POC).  1.0 is standard.")
        cfg["ext_mult"] = _ask(
            f"  Extension multiplier [{cfg['ext_mult']}]: ",
            default=cfg["ext_mult"], cast=float,
        )

    # ── Auto-select VP lookback window ────────────────────────────────────
    # Uses LOOKBACK_DEFAULTS unless the user already set one via CLI.
    if cfg.get("lookback", 0) == 0:
        cfg["lookback"] = LOOKBACK_DEFAULTS.get(cfg["interval"], 20)

    # ── Output filename ───────────────────────────────────────────────────
    # ⚙ Change the default filename below if you want a different save path.
    default_out = f"flip_{cfg['ticker'].lower()}_{cfg['interval']}.png"
    print(f"  Output chart filename.")
    cfg["_out"] = _ask(f"  Save chart as [{default_out}]: ", default=default_out)

    print()
    print("  ─ Running backtest… ──────────────────────────────────────────")
    return cfg


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def main():
    # ── CLI argument parser (for power users / scripting) ─────────────────
    # If you pass no arguments the script switches to interactive prompt mode.
    # Any argument you DO pass skips that particular prompt.
    p = argparse.ArgumentParser(
        description="VAH→VAL Flip Long Strategy — backtester",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Run with no arguments for the interactive prompt (recommended for beginners).

CLI examples (bypasses prompts):
  python vah_val_flip_long.py --no-prompt
  python vah_val_flip_long.py --ticker NQ=F --interval 4h --lookback_days 300
  python vah_val_flip_long.py --ticker SPY  --interval 1d --start 2020-01-01 --end 2025-01-01

Supported intervals:  1m  5m  15m  30m  1h  4h  1d  1wk
""",
    )
    p.add_argument("--ticker",        default=None, help="Yahoo Finance ticker, e.g. SPY NQ=F AAPL")
    p.add_argument("--interval",      default=None, choices=list(TIMEFRAME_MAP.keys()),
                   help="Bar timeframe (see supported intervals above)")
    p.add_argument("--start",         default=None, help="Start date YYYY-MM-DD")
    p.add_argument("--end",           default=None, help="End date YYYY-MM-DD (default: today)")
    p.add_argument("--lookback_days", type=int, default=None,
                   help="Calendar days of history (alternative to --start/--end)")
    p.add_argument("--capital",       type=float, default=None, help="Starting capital in USD")
    p.add_argument("--lookback",      type=int,   default=None,
                   help="Bars per volume-profile window (0 = auto)")
    p.add_argument("--risk",          type=float, default=None,
                   help="Risk per trade as a decimal, e.g. 0.02 for 2%%")
    p.add_argument("--out",           default=None, help="Chart output filename")
    # ⚙ Add --no-prompt to skip the interactive wizard and use DEFAULT_CONFIG + any CLI flags
    p.add_argument("--no-prompt",     action="store_true",
                   help="Skip interactive prompts and use DEFAULT_CONFIG + any flags above")
    args = p.parse_args()

    # ── Decide: interactive or silent? ────────────────────────────────────
    # Interactive mode runs when the user typed no arguments at all.
    # Passing even one flag (like --ticker) or --no-prompt skips the wizard.
    no_args_given = len(sys.argv) == 1
    run_interactive = no_args_given and not args.no_prompt

    if run_interactive:
        # Full interactive setup wizard
        cfg = prompt_config()
        out = cfg.pop("_out", "flip_long_results.png")
    else:
        # Silent mode: start from DEFAULT_CONFIG and apply any CLI overrides
        cfg = DEFAULT_CONFIG.copy()
        if args.ticker:   cfg["ticker"]          = args.ticker.upper()
        if args.interval: cfg["interval"]         = args.interval
        if args.capital:  cfg["initial_capital"]  = args.capital
        if args.risk:     cfg["risk_pct"]         = args.risk

        # Resolve date range
        if args.lookback_days:
            end_dt   = pd.Timestamp.now().normalize()
            start_dt = end_dt - pd.Timedelta(days=args.lookback_days)
            cfg["start"] = start_dt.strftime("%Y-%m-%d")
            cfg["end"]   = end_dt.strftime("%Y-%m-%d")
        else:
            if args.start: cfg["start"] = args.start
            if args.end:   cfg["end"]   = args.end

        # Resolve VP lookback window
        cfg["lookback"] = (args.lookback if args.lookback
                           else LOOKBACK_DEFAULTS.get(cfg["interval"], 20))

        out = args.out or f"flip_{cfg['ticker'].lower()}_{cfg['interval']}.png"

        # Warn if date range exceeds the feed's history limit
        tf_info  = TIMEFRAME_MAP.get(cfg["interval"], TIMEFRAME_MAP["1d"])
        req_days = (pd.Timestamp(cfg["end"]) - pd.Timestamp(cfg["start"])).days
        if req_days > tf_info["max_days"]:
            print(f"  ⚠  {cfg['interval']} data is only available for "
                  f"the last {tf_info['max_days']} days "
                  f"(you requested {req_days} days).")

    # ── Print run configuration ────────────────────────────────────────────
    print()
    print("  Settings")
    print("  ─────────────────────────────────────────────")
    print(f"  Ticker       : {cfg['ticker']}")
    print(f"  Interval     : {cfg['interval']}")
    print(f"  Period       : {cfg['start']} → {cfg['end']}")
    print(f"  Capital      : ${cfg['initial_capital']:,.0f}")
    print(f"  VP lookback  : {cfg['lookback']} bars")
    print(f"  EMA filter   : {cfg['ema_window']}-bar")
    print(f"  Flip tol.    : {cfg['flip_tolerance']*100:.1f}%")
    print(f"  Risk/trade   : {cfg['risk_pct']*100:.1f}%")
    print(f"  Stop         : {cfg['stop_pct']*100:.1f}% below VAL")
    print(f"  Output       : {out}")
    print()

    # ── Run ────────────────────────────────────────────────────────────────
    raw, trades_df, equity_df, levels_df, entry_df, exit_df, flip_df = run_backtest(cfg)
    stats = compute_stats(equity_df, cfg["initial_capital"], trades_df)

    # ── Print results ──────────────────────────────────────────────────────
    print()
    print("  Results")
    print("  ─────────────────────────────────────────────")
    for k, v in stats.items():
        print(f"  {k:<20} {v}")
    print()

    if not trades_df.empty and "type" in trades_df.columns:
        print("  Exit breakdown:")
        print(trades_df["type"].value_counts().to_string(header=False))
    print()

    # ── Chart ──────────────────────────────────────────────────────────────
    plot_results(raw, trades_df, equity_df, levels_df, entry_df, exit_df,
                 flip_df, cfg, stats, out=out)


if __name__ == "__main__":
    main()
