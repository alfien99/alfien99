#!/usr/bin/env python3
"""
Strategy 1: VAH→VAL Flip Long Only
====================================
Core idea (plain English):
  - Every period, we compute a "Volume Profile" — a histogram showing WHERE
    most trading happened (by volume) on the price axis.
  - The top of the 70% value area is called VAH (Value Area High).
  - The bottom is VAL (Value Area Low).  The peak-volume price is POC.
  - When the market steps UP so that the NEW period's VAL aligns with the
    OLD period's VAH, that level has "flipped" from resistance → support.
  - We buy the pullback to that flipped level, expecting buyers to defend it
    just as they did when they first pushed price above it.

Entry rules:
  1. Compute rolling VP for the past `lookback` bars → get VAH(t)
  2. Compute rolling VP for the past `lookback` bars ending one period back → VAH(t-1)
  3. If |VAL(t) - VAH(t-1)| / VAH(t-1) < flip_tolerance → a flip has occurred
  4. Enter long when close pulls back into the VAL zone (close <= VAL(t) * (1 + entry_buffer))
  5. Only enter if price is above the 50-bar EMA (trend filter — avoids buying dips in downtrends)

Exits (scaled — selling in pieces to lock in gains step by step):
  - 50% of position at POC  (the highest-volume price — strong magnet)
  - 70% of remainder at VAH  (top of value area — natural resistance)
  - Rest at VAH + 1× (VAH - POC) extension  (the "stretch" target)
  - Stop: below VAL × (1 - stop_pct)  (if it falls through support, we're wrong)

Supported timeframes (--interval flag):
  1m, 5m, 15m, 30m, 1h, 4h, 1d, 1wk
  Note: yfinance caps intraday history (1m → 7 days, 5m/15m/30m → 60 days, 1h/4h → 730 days).
  4H bars are built by resampling 1H data — yfinance has no native 4H feed.

Usage:
  python vah_val_flip_long.py
  python vah_val_flip_long.py --ticker NQ=F --interval 4h --lookback_days 180
  python vah_val_flip_long.py --ticker SPY  --interval 1d --start 2022-01-01 --end 2025-01-01
"""

import argparse
import warnings

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec

warnings.filterwarnings("ignore")

# ── Optional dependency: yfinance for live market data ────────────────────────
# If not installed, the script falls back to realistic synthetic (fake) price data.
# Install with:  pip install yfinance
try:
    import yfinance as yf
    _YF_AVAILABLE = True
except ImportError:
    _YF_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────────────────
#  TIMEFRAME REGISTRY
#  Tells the downloader which yfinance interval to request, how many calendar
#  days of history are available, and how many bars make up one trading day.
#  "resample" means we pull a finer feed and aggregate up to the target bar size.
# ─────────────────────────────────────────────────────────────────────────────
TIMEFRAME_MAP = {
    #  key      yf interval  max history  approx bars/day  resample target
    "1m":  {"yf": "1m",  "max_days": 7,    "bars_per_day": 390, "resample": None},
    "5m":  {"yf": "5m",  "max_days": 60,   "bars_per_day": 78,  "resample": None},
    "15m": {"yf": "15m", "max_days": 60,   "bars_per_day": 26,  "resample": None},
    "30m": {"yf": "30m", "max_days": 60,   "bars_per_day": 13,  "resample": None},
    "1h":  {"yf": "1h",  "max_days": 730,  "bars_per_day": 6,   "resample": None},
    # yfinance has no 4H feed — download 1H and resample to 4-bar buckets
    "4h":  {"yf": "1h",  "max_days": 730,  "bars_per_day": 2,   "resample": "4h"},
    "1d":  {"yf": "1d",  "max_days": 3650, "bars_per_day": 1,   "resample": None},
    "1wk": {"yf": "1wk", "max_days": 3650, "bars_per_day": 0.2, "resample": None},
}


# ─────────────────────────────────────────────────────────────────────────────
#  DATA LAYER
# ─────────────────────────────────────────────────────────────────────────────

def _resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """
    Collapse finer bars into coarser bars.
    E.g. rule="4h" turns four 1H candles into one 4H candle.
    Aggregation rules:
      Open  → first bar's open   (where the 4H session started)
      High  → highest of the 4 highs  (the peak reached)
      Low   → lowest of the 4 lows    (the trough)
      Close → last bar's close        (where the 4H session ended)
      Volume → sum of all bars        (total activity)
    """
    agg = {
        "Open":   "first",
        "High":   "max",
        "Low":    "min",
        "Close":  "last",
        "Volume": "sum",
    }
    # label="right" means the timestamp on each resampled bar is its END time,
    # closed="right" means the interval includes that boundary.
    resampled = df.resample(rule, label="right", closed="right").agg(agg)
    # Drop incomplete periods at the end and empty rows
    resampled.dropna(subset=["Close"], inplace=True)
    return resampled


def _synthetic_ohlcv(ticker: str, n_bars: int, bars_per_day: float,
                     interval: str) -> pd.DataFrame:
    """
    Generate fake but realistic OHLCV data when live data isn't available.
    Uses geometric Brownian motion (the standard academic model for stock prices)
    with GARCH-like volatility clustering (quiet periods followed by volatile ones).

    Parameters
    ----------
    ticker      : used to seed the random number generator so the same ticker
                  always produces the same fake history
    n_bars      : how many bars to create
    bars_per_day: controls the date frequency label (e.g. 6 → hourly)
    interval    : the timeframe string, used to pick the right pandas freq label
    """
    rng = np.random.default_rng(abs(hash(ticker)) % (2**31))
    S0 = 18_000.0   # starting price

    # ── Volatility clustering (GARCH-inspired) ────────────────────────────
    # Each bar's volatility is mostly inherited from the previous bar (0.91 weight)
    # plus a small random shock — this gives realistic "calm then stormy" periods.
    sigma_base = 0.013
    vols = np.full(n_bars, sigma_base)
    for i in range(1, n_bars):
        vols[i] = 0.91 * vols[i-1] + 0.09 * sigma_base * abs(rng.standard_normal()) + 0.001

    # ── Price path (geometric Brownian motion) ────────────────────────────
    mu = 0.0004   # slight upward drift per bar
    closes  = S0 * np.exp(np.cumsum(rng.standard_normal(n_bars) * vols + mu))

    # ── High / Low around each close ──────────────────────────────────────
    hl      = closes * vols * 2.5   # typical high-low range per bar
    highs   = closes + hl * rng.uniform(0.3, 0.7, n_bars)
    lows    = closes - hl * rng.uniform(0.3, 0.7, n_bars)

    # Open of each bar = close of the previous bar (no gaps in synthetic data)
    opens   = np.roll(closes, 1)
    opens[0] = S0

    vols_v = rng.integers(1_000_000, 5_000_000, n_bars)

    # Pick the right pandas frequency string for the date index
    freq_map = {
        "1m": "1min", "5m": "5min", "15m": "15min", "30m": "30min",
        "1h": "1h",   "4h": "4h",   "1d": "B",      "1wk": "W",
    }
    freq = freq_map.get(interval, "B")   # "B" = business days

    dates = pd.date_range(end=pd.Timestamp.now().normalize(), periods=n_bars, freq=freq)
    df = pd.DataFrame(
        {"Open": opens, "High": highs, "Low": lows, "Close": closes, "Volume": vols_v},
        index=dates,
    )
    df.index.name = "Date"
    return df


def _download(ticker: str, start: str, end: str, interval: str) -> pd.DataFrame:
    """
    Fetch OHLCV data from Yahoo Finance, or fall back to synthetic data.

    Important yfinance history limits (as of 2025):
      1m  → only last 7 calendar days
      5m / 15m / 30m → only last 60 calendar days
      1h  → only last 730 calendar days (~2 years)
      1d and above → unlimited history
    If your start date is beyond the limit, yfinance silently returns fewer bars.
    """
    tf = TIMEFRAME_MAP.get(interval, TIMEFRAME_MAP["1d"])
    yf_interval = tf["yf"]           # the interval string yfinance understands
    resample_to = tf["resample"]     # e.g. "4h" if we need to aggregate later
    bars_per_day = tf["bars_per_day"]

    if _YF_AVAILABLE:
        try:
            print(f"  Downloading {ticker} [{interval}] from {start} to {end} …")
            raw = yf.download(
                ticker, start=start, end=end,
                interval=yf_interval,
                auto_adjust=True,   # adjusts prices for splits/dividends
                progress=False,
            )
            # yfinance sometimes returns a MultiIndex with the ticker as second level
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.droplevel(1)

            raw.dropna(inplace=True)

            if len(raw) > 20:
                # Resample if we requested a timeframe yfinance doesn't serve natively
                if resample_to:
                    raw = _resample_ohlcv(raw, resample_to)
                    print(f"  Resampled to {resample_to}: {len(raw)} bars.")
                else:
                    print(f"  Downloaded {len(raw)} bars.")
                return raw
            else:
                print(f"  Too few bars returned ({len(raw)}). "
                      f"Check that --start is within the {tf['max_days']}-day limit for {interval}.")
        except Exception as e:
            print(f"  Yahoo Finance error ({e}) — falling back to synthetic data.")

    # ── Synthetic fallback ─────────────────────────────────────────────────
    print(f"  Generating synthetic OHLCV for {ticker} [{interval}] …")
    # Estimate how many bars we need from the date range
    try:
        day_span = (pd.Timestamp(end) - pd.Timestamp(start)).days
    except Exception:
        day_span = 365
    n_bars = max(int(day_span * bars_per_day), 200)
    df = _synthetic_ohlcv(ticker, n_bars, bars_per_day, interval)
    if resample_to:
        df = _resample_ohlcv(df, resample_to)
    return df


# ─────────────────────────────────────────────────────────────────────────────
#  VOLUME PROFILE CALCULATION
#  A Volume Profile is a horizontal histogram that answers the question:
#  "At which price levels did the most trading volume occur during this period?"
#
#  Steps:
#  1. Divide the price range into `bins` equal buckets.
#  2. For each candle, distribute its volume across the buckets that overlap
#     the candle's high-low range (proportional to the overlap).
#  3. Find the POC = the bucket with the most volume.
#  4. Expand outward from the POC until we've captured 70% of total volume.
#     The outer edges of that 70% zone are VAL (bottom) and VAH (top).
# ─────────────────────────────────────────────────────────────────────────────

def compute_vp(ohlcv: pd.DataFrame, bins: int = 100):
    """
    Compute the Volume Profile for a slice of OHLCV data.

    Returns (poc, val, vah, price_mids, volume_per_bin):
      poc  : Point of Control — price with the highest traded volume
      val  : Value Area Low   — bottom of the 70% volume zone
      vah  : Value Area High  — top of the 70% volume zone
      mids : array of the centre price for each bin
      vol  : array of the volume assigned to each bin
    """
    lo = float(ohlcv["Low"].min())
    hi = float(ohlcv["High"].max())

    # Edge case: flat price (no range) — return the midpoint for all levels
    if hi <= lo + 1e-8:
        mid = (hi + lo) / 2
        return mid, mid, mid, np.array([mid]), np.array([1.0])

    # Create `bins` equal-width price buckets between the overall low and high
    edges = np.linspace(lo, hi, bins + 1)        # bin boundaries
    mids  = (edges[:-1] + edges[1:]) / 2         # centre of each bucket
    vol   = np.zeros(bins)                        # volume accumulator per bucket

    for k in range(len(ohlcv)):
        bar_hi = float(ohlcv["High"].iloc[k])
        bar_lo = float(ohlcv["Low"].iloc[k])
        bar_v  = float(ohlcv["Volume"].iloc[k])
        rng    = bar_hi - bar_lo

        if rng < 1e-10:
            # Doji / zero-range bar: assign all volume to its single bucket
            idx = min(int((bar_lo - lo) / (hi - lo) * bins), bins - 1)
            vol[idx] += bar_v
        else:
            # Spread volume proportionally across every bucket the bar overlaps.
            # overlap[i] = how many price units of bucket i are inside [bar_lo, bar_hi]
            overlap = np.maximum(
                0.0,
                np.minimum(edges[1:], bar_hi) - np.maximum(edges[:-1], bar_lo)
            )
            vol += bar_v * overlap / rng

    # ── POC: the price bucket with the most volume ─────────────────────────
    poc_idx = int(np.argmax(vol))
    poc     = mids[poc_idx]

    # ── Value Area: expand from POC until 70% of total volume is enclosed ──
    # The algorithm grows the window one step at a time, always picking the
    # side that adds more volume — this is the standard TPO / Market Profile rule.
    target = vol.sum() * 0.70
    li = hi_i = poc_idx   # lower index / higher index (grow outward from POC)
    acc = vol[poc_idx]    # volume accumulated so far

    while acc < target:
        # Volume available if we expand one step down vs one step up
        add_lo = vol[li   - 1] if li   > 0        else -1.0
        add_hi = vol[hi_i + 1] if hi_i < bins - 1 else -1.0

        if add_lo < 0 and add_hi < 0:
            break   # hit both edges — we're done

        # Expand toward whichever side has more volume (ties go upward)
        if add_hi >= add_lo:
            hi_i += 1; acc += vol[hi_i]
        else:
            li   -= 1; acc += vol[li]

    return poc, mids[li], mids[hi_i], mids, vol


# ─────────────────────────────────────────────────────────────────────────────
#  BACKTEST ENGINE
#  Walks through price bar by bar and simulates entries/exits.
#  This is a "bar-by-bar" simulation — we only know what happened up to bar i
#  when we make decisions at bar i.  No lookahead.
# ─────────────────────────────────────────────────────────────────────────────

def run_backtest(cfg: dict):
    """
    Run the VAH→VAL flip strategy on historical data described by `cfg`.

    The "flip" concept (visualise this as a staircase going up):
      Period t-1:  VAL────────POC────────VAH  (old value area)
      Period t  :                 VAL────POC────VAH  (new value area shifted up)
                                  ↑ new VAL is near old VAH → it "flipped"

    Returns several DataFrames used by the plotting and stats functions.
    """
    raw      = _download(cfg["ticker"], cfg["start"], cfg["end"], cfg["interval"])
    lookback = cfg["lookback"]   # number of bars per volume-profile window
    bins     = cfg["vp_bins"]
    tol      = cfg["flip_tolerance"]   # how close new VAL must be to old VAH
    buf      = cfg["entry_buffer"]     # how far above VAL we still allow entry
    stop_p   = cfg["stop_pct"]         # stop is X% below VAL
    ema_win  = cfg["ema_window"]       # trend filter — only trade above this EMA
    capital  = float(cfg["initial_capital"])

    # ── Exponential Moving Average (trend filter) ─────────────────────────
    # EMA gives more weight to recent bars.  We use it as a simple uptrend check:
    # if price > EMA → uptrend → we're allowed to buy.  If price < EMA → skip.
    ema = raw["Close"].ewm(span=ema_win, adjust=False).mean()

    # ── Trade-state variables ─────────────────────────────────────────────
    in_pos     = False   # are we currently holding a position?
    shares     = 0.0     # how many units we own
    avg_px     = 0.0     # average price we paid
    cost       = 0.0     # total cash tied up in the position (shares × avg_px)
    entry_stop = 0.0     # the stop-loss price level for this trade
    entry_poc  = 0.0     # the POC target level for the first partial exit
    entry_vah  = 0.0     # the VAH target level for the second partial exit
    entry_ext  = 0.0     # the extension target for the final exit
    poc_done   = False   # have we already taken profits at the POC?
    vah_done   = False   # have we already taken profits at the VAH?

    # ── Output containers ─────────────────────────────────────────────────
    equity_records = []   # running portfolio value at each bar
    level_records  = []   # VP levels at each bar (for plotting)
    trade_records  = []   # completed trades (for stats)
    entry_marks    = []   # dates/prices of entries (for chart markers)
    exit_marks     = []   # dates/prices of exits  (for chart markers)
    flip_marks     = []   # dates/prices where a flip was detected

    # We need at least 2× lookback bars of history before the first signal is valid
    min_i = max(lookback * 2, ema_win + 5)

    for i in range(min_i, len(raw)):
        date  = raw.index[i]
        close = float(raw["Close"].iloc[i])
        high  = float(raw["High"].iloc[i])
        low   = float(raw["Low"].iloc[i])

        # ── Compute Volume Profiles ───────────────────────────────────────
        # "Current" period: the most recent `lookback` bars ending now
        poc_c, val_c, vah_c, _, _ = compute_vp(raw.iloc[i - lookback : i], bins)
        # "Previous" period: the `lookback` bars before the current period
        poc_p, val_p, vah_p, _, _ = compute_vp(raw.iloc[i - lookback * 2 : i - lookback], bins)

        # ── Flip Detection ────────────────────────────────────────────────
        # A "flip" means the value area has stepped up:
        #   Condition 1: current VAL is higher than previous VAL (market moved up)
        #   Condition 2: current VAL is close to (or above) previous VAH
        #                (the new floor is where the old ceiling was)
        val_above_prev_val = val_c > val_p
        near_prev_vah = (
            (vah_p > 0 and abs(val_c - vah_p) / vah_p < tol)   # within tolerance %
            or val_c >= vah_p                                    # or already above it
        )
        flip = val_above_prev_val and near_prev_vah

        # Trend filter: only trade when price is above its EMA (we're in an uptrend)
        uptrend = close > float(ema.iloc[i])

        # ── Compute trade levels for this bar ─────────────────────────────
        # Stop-loss: placed just below the current VAL
        # If price falls here, the flip level failed and we exit to limit losses
        stop_lvl = val_c * (1 - stop_p)

        # Extension target: beyond VAH, sized as one times the VAH-to-POC distance
        # This is like a "measured move" — if price broke VAH, it could travel that far again
        ext_lvl = vah_c + cfg["ext_mult"] * (vah_c - poc_c)

        # Save levels for plotting
        level_records.append({
            "date": date, "poc": poc_c, "val": val_c, "vah": vah_c,
            "vah_prev": vah_p, "flip": flip, "ema": float(ema.iloc[i]),
        })
        if flip:
            flip_marks.append({"date": date, "level": val_c})

        # ── EXIT LOGIC (checked before entry to avoid same-bar in-and-out) ──
        if in_pos:
            if low <= entry_stop:
                # ── Stop-loss hit ─────────────────────────────────────────
                # The flip level broke down — take the loss and move on
                pnl     = (entry_stop - avg_px) * shares
                capital += cost + pnl   # return the cash (with loss deducted)
                trade_records.append({
                    "date": date, "type": "stop",
                    "entry": avg_px, "exit": entry_stop, "pnl": pnl,
                })
                exit_marks.append({"date": date, "price": entry_stop, "type": "stop"})
                in_pos = False; shares = cost = 0.0
                poc_done = vah_done = False

            elif not poc_done and high >= entry_poc:
                # ── First partial exit at POC (50% of position) ──────────
                # POC is the highest-volume price — strong magnet and natural target
                cs       = shares * cfg["poc_exit_frac"]
                pnl_p    = (entry_poc - avg_px) * cs
                capital += cs * avg_px + pnl_p   # return cash + profit
                cost    -= cs * avg_px            # reduce cost basis
                shares  -= cs
                poc_done = True
                exit_marks.append({"date": date, "price": entry_poc, "type": "poc"})

            elif poc_done and not vah_done and high >= entry_vah:
                # ── Second partial exit at VAH (70% of remaining) ────────
                # Top of the value area — where sellers tend to re-emerge
                cs       = shares * cfg["vah_exit_frac"]
                pnl_p    = (entry_vah - avg_px) * cs
                capital += cs * avg_px + pnl_p
                cost    -= cs * avg_px
                shares  -= cs
                vah_done = True
                exit_marks.append({"date": date, "price": entry_vah, "type": "vah"})

            elif vah_done and shares > 0 and high >= entry_ext:
                # ── Final exit at extension target (the "runner") ─────────
                # The remaining small piece is held for the full measured move
                pnl     = (entry_ext - avg_px) * shares
                capital += cost + pnl
                trade_records.append({
                    "date": date, "type": "target",
                    "entry": avg_px, "exit": entry_ext, "pnl": pnl,
                })
                exit_marks.append({"date": date, "price": entry_ext, "type": "ext"})
                in_pos = False; shares = cost = 0.0
                poc_done = vah_done = False

        # ── ENTRY LOGIC ───────────────────────────────────────────────────
        # All four conditions must be true:
        #   1. Not already in a trade
        #   2. A VAH→VAL flip is active
        #   3. We are in an uptrend (price > EMA)
        #   4. This bar's LOW touched the VAL zone (the pullback arrived)
        entry_px = val_c * (1 + buf)   # the limit order price — VAL + small buffer
        if (
            not in_pos
            and flip
            and uptrend
            and low  <= entry_px       # bar dipped into the VAL zone
            and close > stop_lvl       # but didn't blow through the stop already
        ):
            # ── Position sizing (fixed-fraction risk) ─────────────────────
            # We risk exactly `risk_pct` of current capital on every trade.
            # size = (dollars_at_risk) / (price_from_entry_to_stop)
            fill_px     = min(close, entry_px)   # simulate a limit order fill
            risk_cash   = capital * cfg["risk_pct"]
            risk_per_sh = max(fill_px - stop_lvl, 1e-6)
            n_sh        = risk_cash / risk_per_sh
            # Also cap position at `max_position_pct` of capital (diversification guard)
            order_cash  = min(n_sh * fill_px, capital * cfg["max_position_pct"])

            if order_cash >= 50 and capital >= order_cash:   # minimum viable trade
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

        # Track portfolio value (cash + mark-to-market value of open position)
        equity_records.append({"date": date, "equity": capital + shares * close})

    # ── Close any open position at end of data ("mark to market") ─────────
    if in_pos and shares > 0:
        lp  = float(raw["Close"].iloc[-1])
        pnl = (lp - avg_px) * shares
        capital += cost + pnl
        trade_records.append({
            "date": raw.index[-1], "type": "expired",
            "entry": avg_px, "exit": lp, "pnl": pnl,
        })

    # ── Package results into DataFrames ───────────────────────────────────
    equity_df = pd.DataFrame(equity_records).set_index("date")
    levels_df = pd.DataFrame(level_records).set_index("date")
    trades_df = (
        pd.DataFrame(trade_records)
        if trade_records
        else pd.DataFrame(columns=["pnl", "type"])
    )
    entry_df = pd.DataFrame(entry_marks) if entry_marks else pd.DataFrame()
    exit_df  = pd.DataFrame(exit_marks)  if exit_marks  else pd.DataFrame()
    flip_df  = pd.DataFrame(flip_marks)  if flip_marks  else pd.DataFrame()

    return raw, trades_df, equity_df, levels_df, entry_df, exit_df, flip_df


# ─────────────────────────────────────────────────────────────────────────────
#  PERFORMANCE STATISTICS
#  Summarise what the backtest achieved in a few key numbers.
# ─────────────────────────────────────────────────────────────────────────────

def compute_stats(equity_df: pd.DataFrame, initial_capital: float,
                  trades_df: pd.DataFrame) -> dict:
    """
    Calculate standard trading performance metrics.

    Total Return   : (final value − starting value) / starting value × 100
    Annualised Return : what the total return would be if held for exactly 1 year
    Max Drawdown   : the worst peak-to-trough loss during the test period
    Sharpe Ratio   : risk-adjusted return; >1 is decent, >2 is very good
    Win Rate       : % of closed trades that made money
    """
    final   = float(equity_df["equity"].iloc[-1])
    total_r = (final - initial_capital) / initial_capital * 100

    # Annualise: compound the total return to a per-year rate
    n_days  = max((equity_df.index[-1] - equity_df.index[0]).days, 1)
    ann_r   = ((final / initial_capital) ** (365 / n_days) - 1) * 100

    # Max drawdown: how far below the previous peak did equity fall?
    roll_max = equity_df["equity"].cummax()
    max_dd   = float(((equity_df["equity"] - roll_max) / roll_max * 100).min())

    # Sharpe: mean daily return divided by its standard deviation, scaled to yearly
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


# ─────────────────────────────────────────────────────────────────────────────
#  CHART
# ─────────────────────────────────────────────────────────────────────────────

# Dark-theme colour palette
BG, GRID, FG   = "#0d1117", "#21262d", "#e6edf3"
BLUE, RED, ORG = "#58a6ff", "#f85149", "#ffa657"
GRN,  PRP, YLW = "#3fb950", "#bc8cff", "#e3b341"


def _style(ax):
    """Apply dark-theme styling to a single Axes object."""
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
    Produce a 4-panel chart:
      Panel 1 (top, large): Price with VP levels, flip zones, and trade markers
      Panel 2 (middle):     Equity curve showing portfolio growth over time
      Panel 3 (lower):      Drawdown chart — how far below peak equity fell
      Panel 4 (bottom):     Stats table
    """
    fig = plt.figure(figsize=(20, 15), facecolor=BG)
    gs  = GridSpec(4, 1, figure=fig, height_ratios=[3, 1, 1, 0.55], hspace=0.42)
    ax_p, ax_eq, ax_dd, ax_st = [fig.add_subplot(gs[i]) for i in range(4)]
    for ax in (ax_p, ax_eq, ax_dd, ax_st):
        _style(ax)

    # ── Panel 1: Price chart ───────────────────────────────────────────────
    # Align the price series to the same dates we computed VP levels for
    sl = raw.loc[levels_df.index]
    ax_p.plot(sl.index, sl["Close"], color=BLUE, lw=1.0, label="Price")
    ax_p.plot(
        levels_df.index, levels_df["ema"],
        color=YLW, lw=0.9, ls="--", alpha=0.7,
        label=f"EMA({cfg['ema_window']}) — trend filter",
    )

    # Shade the value area (VAL to VAH) with a faint green band
    ax_p.fill_between(levels_df.index, levels_df["val"], levels_df["vah"],
                      alpha=0.07, color=GRN)
    ax_p.plot(levels_df.index, levels_df["val"], color=RED, lw=0.8, ls="--",
              alpha=0.9, label="VAL (Value Area Low)")
    ax_p.plot(levels_df.index, levels_df["poc"], color=ORG, lw=0.8, ls="-",
              alpha=0.9, label="POC (Point of Control)")
    ax_p.plot(levels_df.index, levels_df["vah"], color=GRN, lw=0.8, ls="--",
              alpha=0.9, label="VAH (Value Area High)")

    # Highlight each bar where a flip was detected with a yellow horizontal band
    for _, r in levels_df[levels_df["flip"]].iterrows():
        ax_p.axhspan(r["val"] * 0.998, r["val"] * 1.002, alpha=0.18, color=YLW)

    # Plot entry triangles (green △) and exit diamonds (coloured by exit type)
    if not entry_df.empty:
        ax_p.scatter(
            entry_df["date"], entry_df["price"],
            marker="^", s=80, color=GRN, zorder=5,
            edgecolors="white", lw=0.4, label="Long entry",
        )
    if not exit_df.empty:
        exit_colours = {"stop": RED, "poc": ORG, "vah": GRN, "ext": PRP}
        for _, r in exit_df.iterrows():
            ax_p.scatter(
                r["date"], r["price"],
                marker="D", s=55, zorder=5,
                color=exit_colours.get(r["type"], FG),
                edgecolors="white", lw=0.4,
            )

    ax_p.legend(
        handles=[
            mpatches.Patch(color=BLUE, label="Price"),
            mpatches.Patch(color=YLW,  label=f"EMA({cfg['ema_window']}) trend filter"),
            mpatches.Patch(color=RED,  label="VAL"),
            mpatches.Patch(color=ORG,  label="POC"),
            mpatches.Patch(color=GRN,  label="VAH"),
            mpatches.Patch(color=YLW,  alpha=0.4, label="Flip zone (VAH→VAL)"),
            mpatches.Patch(color=GRN,  label="▲ Entry at flipped VAL"),
            mpatches.Patch(color=RED,  label="◆ Stop exit"),
            mpatches.Patch(color=ORG,  label="◆ POC partial exit"),
            mpatches.Patch(color=PRP,  label="◆ Extension target"),
        ],
        loc="upper left", fontsize=7, facecolor=BG, labelcolor=FG,
        framealpha=0.8, ncol=3,
    )
    ax_p.set_title(
        f"{cfg['ticker']}  [{cfg['interval']}]  ·  VAH→VAL Flip Long  ·  "
        f"{cfg['start']} → {cfg['end']}",
        fontsize=11, fontweight="bold",
    )
    ax_p.set_ylabel("Price ($)")

    # ── Panel 2: Equity curve ──────────────────────────────────────────────
    eq = equity_df["equity"]
    ax_eq.plot(eq.index, eq, color=GRN, lw=1.5)
    ax_eq.axhline(cfg["initial_capital"], color=FG, lw=0.7, ls="--", alpha=0.3)
    # Green fill above starting capital = profit; red fill below = loss
    ax_eq.fill_between(eq.index, cfg["initial_capital"], eq,
                       where=eq >= cfg["initial_capital"], color=GRN, alpha=0.12)
    ax_eq.fill_between(eq.index, cfg["initial_capital"], eq,
                       where=eq <  cfg["initial_capital"], color=RED, alpha=0.12)
    ax_eq.set_title("Equity Curve", fontsize=10)
    ax_eq.set_ylabel("Equity ($)")

    # ── Panel 3: Drawdown ──────────────────────────────────────────────────
    # Drawdown = how far equity has fallen below its all-time high at each point
    rm  = eq.cummax()                        # running maximum (all-time high)
    ddp = (eq - rm) / rm * 100              # percentage below the peak
    ax_dd.fill_between(ddp.index, ddp, 0, color=RED, alpha=0.55)
    ax_dd.plot(ddp.index, ddp, color=RED, lw=0.8)
    ax_dd.set_title("Drawdown (%)", fontsize=10)
    ax_dd.set_ylabel("DD %")

    # ── Panel 4: Stats table ───────────────────────────────────────────────
    ax_st.axis("off")
    tbl = ax_st.table(
        cellText=[list(stats.values())],
        colLabels=list(stats.keys()),
        cellLoc="center", loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 2.2)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_facecolor("#161b22" if r == 0 else BG)
        cell.set_edgecolor(GRID)
        cell.set_text_props(color=FG)

    plt.suptitle(
        "Strategy 1: VAH→VAL Flip Long  —  Buy the level that was once resistance, now confirmed support",
        fontsize=11, color=FG, y=1.005, fontweight="bold",
    )
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
    print(f"  Chart saved → {out}")
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
#  DEFAULT CONFIGURATION
#  Change these values to tune the strategy.
#  You can also override most of them via command-line flags (see --help).
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_CONFIG = {
    # ── Data settings ────────────────────────────────────────────────────
    "ticker":   "SPY",
    "interval": "1d",          # timeframe: 1m 5m 15m 30m 1h 4h 1d 1wk
    "start":    "2022-01-01",
    "end":      "2025-01-01",

    # ── Volume Profile settings ───────────────────────────────────────────
    "lookback": 20,            # bars per VP window (adjust for your timeframe)
    "vp_bins":  100,           # price buckets in the histogram (more = finer detail)

    # ── Capital & risk ────────────────────────────────────────────────────
    "initial_capital":  10_000,
    "risk_pct":         0.02,  # risk 2% of portfolio per trade
    "max_position_pct": 0.40,  # never put more than 40% of capital in one trade

    # ── Strategy parameters ───────────────────────────────────────────────
    "ema_window":      50,     # trend filter: only trade when close > this EMA
    "flip_tolerance":  0.03,   # new VAL must be within 3% of old VAH to count as a flip
    "entry_buffer":    0.005,  # enter when price is up to 0.5% above VAL
    "stop_pct":        0.015,  # stop-loss is 1.5% below VAL

    # ── Exit fractions ────────────────────────────────────────────────────
    "poc_exit_frac": 0.50,     # sell 50% when price reaches POC
    "vah_exit_frac": 0.70,     # sell 70% of remainder when price reaches VAH
    "ext_mult":      1.0,      # extension target = VAH + 1× (VAH − POC)
}

# ── Sensible per-timeframe lookback defaults ──────────────────────────────────
# Shorter bars need more bars to represent the same "one session" of market activity
LOOKBACK_DEFAULTS = {
    "1m": 120, "5m": 48, "15m": 32, "30m": 20,
    "1h": 16,  "4h": 8,  "1d": 20,  "1wk": 12,
}


# ─────────────────────────────────────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def main():
    # ── Command-line argument parser ─────────────────────────────────────
    # Lets you run the script with custom settings without editing the file:
    #   python vah_val_flip_long.py --ticker QQQ --interval 1h --lookback_days 180
    p = argparse.ArgumentParser(
        description="VAH→VAL Flip Long Strategy backtester",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python vah_val_flip_long.py
  python vah_val_flip_long.py --interval 4h --lookback_days 400
  python vah_val_flip_long.py --ticker NQ=F --interval 1h --start 2023-06-01 --end 2024-12-01
  python vah_val_flip_long.py --ticker SPY  --interval 1d --start 2020-01-01 --end 2025-01-01

Supported intervals:
  1m  (last 7 days only)   5m  (last 60 days)   15m (last 60 days)
  30m (last 60 days)       1h  (last 730 days)   4h  (last 730 days, resampled)
  1d  (unlimited history)  1wk (unlimited history)
        """,
    )
    p.add_argument("--ticker",       default=None,
                   help="Yahoo Finance ticker symbol, e.g. SPY, NQ=F, AAPL")
    p.add_argument("--interval",     default=None,
                   choices=list(TIMEFRAME_MAP.keys()),
                   help="Bar timeframe (default: 1d)")
    p.add_argument("--start",        default=None,
                   help="Start date YYYY-MM-DD (must be within interval's history limit)")
    p.add_argument("--end",          default=None,
                   help="End date YYYY-MM-DD (default: today)")
    p.add_argument("--lookback_days", type=int, default=None,
                   help="Calendar days of history to fetch (overrides --start/--end)")
    p.add_argument("--capital",      type=float, default=None,
                   help="Starting capital in USD")
    p.add_argument("--lookback",     type=int,   default=None,
                   help="Bars per volume-profile window")
    p.add_argument("--out",          default="flip_long_results.png",
                   help="Output filename for the chart image")
    args = p.parse_args()

    # ── Build config: start from defaults then apply any CLI overrides ────
    cfg = DEFAULT_CONFIG.copy()
    if args.ticker:   cfg["ticker"]   = args.ticker.upper()
    if args.interval: cfg["interval"] = args.interval
    if args.capital:  cfg["initial_capital"] = args.capital

    # If the user gave a lookback_days, compute start/end from today
    if args.lookback_days:
        end_dt   = pd.Timestamp.now().normalize()
        start_dt = end_dt - pd.Timedelta(days=args.lookback_days)
        cfg["start"] = start_dt.strftime("%Y-%m-%d")
        cfg["end"]   = end_dt.strftime("%Y-%m-%d")
    else:
        if args.start: cfg["start"] = args.start
        if args.end:   cfg["end"]   = args.end

    # Auto-select a sensible VP lookback window if not explicitly provided
    if args.lookback:
        cfg["lookback"] = args.lookback
    else:
        cfg["lookback"] = LOOKBACK_DEFAULTS.get(cfg["interval"], 20)

    # Warn if the requested date range exceeds yfinance's limit for this interval
    tf_info   = TIMEFRAME_MAP.get(cfg["interval"], TIMEFRAME_MAP["1d"])
    max_days  = tf_info["max_days"]
    req_days  = (pd.Timestamp(cfg["end"]) - pd.Timestamp(cfg["start"])).days
    if req_days > max_days:
        print(f"  ⚠ WARNING: {cfg['interval']} data is only available for the last "
              f"{max_days} days, but you requested {req_days} days.")
        print(f"    yfinance will silently return less data than requested.")

    # ── Print run summary ─────────────────────────────────────────────────
    print("\nVAH→VAL Flip Long Strategy")
    print("=" * 50)
    print(f"  Ticker       : {cfg['ticker']}")
    print(f"  Interval     : {cfg['interval']}")
    print(f"  Period       : {cfg['start']} → {cfg['end']}")
    print(f"  Capital      : ${cfg['initial_capital']:,.0f}")
    print(f"  VP lookback  : {cfg['lookback']} bars")
    print(f"  Flip tol.    : {cfg['flip_tolerance']*100:.1f}%")
    print(f"  EMA filter   : {cfg['ema_window']}-bar")
    print(f"  Risk/trade   : {cfg['risk_pct']*100:.1f}%")
    print(f"  Stop         : {cfg['stop_pct']*100:.1f}% below VAL")

    # ── Run backtest ──────────────────────────────────────────────────────
    raw, trades_df, equity_df, levels_df, entry_df, exit_df, flip_df = run_backtest(cfg)
    stats = compute_stats(equity_df, cfg["initial_capital"], trades_df)

    # ── Print results ─────────────────────────────────────────────────────
    print("\n" + "=" * 50)
    for k, v in stats.items():
        print(f"  {k:<20} {v}")
    print("=" * 50)

    if not trades_df.empty and "type" in trades_df.columns:
        print("\n  Exit breakdown:")
        print(trades_df["type"].value_counts().to_string(header=False))
    print()

    # ── Plot ──────────────────────────────────────────────────────────────
    plot_results(
        raw, trades_df, equity_df, levels_df, entry_df, exit_df,
        flip_df, cfg, stats, out=args.out,
    )


if __name__ == "__main__":
    main()
