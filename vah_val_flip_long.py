#!/usr/bin/env python3
"""
VAH / VAL Flip Strategy  ·  Intraday
======================================
Long  : value area steps UP   → buy the pullback to flipped support
          new VAL ≈ old VAH  (what was resistance is now support)

Short : value area steps DOWN  → sell the rally to flipped resistance
          new VAH ≈ old VAL  (what was support is now resistance)
          Shorts use a tighter flip tolerance, require a falling EMA,
          and carry a closer stop — they are intentionally harder to trigger.

Usage:
  python vah_val_flip_long.py          ← interactive prompt
  python vah_val_flip_long.py --quick  ← skip prompt, run with defaults below
"""

import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates
import matplotlib.patches as mpatches

warnings.filterwarnings("ignore")

try:
    import yfinance as yf
    _HAS_YF = True
except ImportError:
    _HAS_YF = False


# ══════════════════════════════════════════════════════════════════════════════
#  INTRADAY TIMEFRAME MENU
#  ────────────────────────
#  ⚙ Add a row here to add a new timeframe option.
#  "yf"        → interval string Yahoo Finance accepts
#  "max_days"  → hard history limit enforced by Yahoo Finance
#  "lookback"  → bars per volume-profile window (≈ 2–3 intraday sessions)
#  "resample"  → pandas offset to aggregate into (None = yfinance serves it natively)
# ══════════════════════════════════════════════════════════════════════════════
TIMEFRAMES = {
    "1m":  {
        "yf": "1m",  "max_days": 7,   "lookback": 60,
        "resample": None,
        "desc": "1 min   · scalping / very fast signals   (max 7 days history)",
    },
    "5m":  {
        "yf": "5m",  "max_days": 60,  "lookback": 48,
        "resample": None,
        "desc": "5 min   · intraday momentum              (max 60 days history)",
    },
    "15m": {
        "yf": "15m", "max_days": 60,  "lookback": 26,
        "resample": None,
        "desc": "15 min  · intraday swing                 (max 60 days history)",
    },
    "30m": {
        "yf": "30m", "max_days": 60,  "lookback": 16,
        "resample": None,
        "desc": "30 min  · half-session swing             (max 60 days history)",
    },
    "1h":  {
        "yf": "1h",  "max_days": 730, "lookback": 12,
        "resample": None,
        "desc": "1 hr    · multi-day swing                (max 730 days history)",
    },
    "4h":  {
        "yf": "1h",  "max_days": 730, "lookback": 6,
        "resample": "4h",
        "desc": "4 hr    · position swing [resampled 1H]  (max 730 days history)",
    },
}
_TF_KEYS = list(TIMEFRAMES.keys())


# ══════════════════════════════════════════════════════════════════════════════
#  STRATEGY CONFIG
#  ────────────────
#  All tunable parameters live here. The interactive prompt sets ticker,
#  interval, and days_back. Everything else is changed directly in this dict.
# ══════════════════════════════════════════════════════════════════════════════
CFG = {

    # ── Symbol & data ─────────────────────────────────────────────────────────
    # ticker   : any Yahoo Finance symbol
    #            Futures  → NQ=F (Nasdaq), ES=F (S&P 500), GC=F (Gold), CL=F (Oil)
    #            Stocks   → AAPL, SPY, QQQ, TSLA, NVDA
    # days     : calendar days of history — must respect each timeframe's max_days
    "ticker": "NQ=F",
    "days":   30,

    # ── Volume profile ────────────────────────────────────────────────────────
    # vp_bins : number of price buckets in the histogram
    #           50 → fast, coarser resolution
    #           100 → balanced (good default)
    #           200 → very fine, slower to compute
    "vp_bins": 50,

    # ── EMA trend filter ──────────────────────────────────────────────────────
    # Long  trades only fire when close is ABOVE the EMA.
    # Short trades only fire when close is BELOW the EMA.
    # ⚙ Smaller → catches trends earlier, more signals, more noise.
    #   Larger  → fewer but cleaner trend trades only.
    "ema_period": 20,

    # ── Long entry ────────────────────────────────────────────────────────────
    # long_flip_tol  : new VAL must be within this % of old VAH to count as a flip
    #                  Raise it → more flips detected (noisier).
    #                  Lower it → fewer, cleaner flips only.
    # long_entry_buf : enter if price is within this % ABOVE VAL
    #                  (simulates a limit order that fills on a pullback near VAL)
    # long_stop_pct  : stop-loss placed this % BELOW VAL
    #                  If price falls through the flip level, exit and accept the loss.
    "long_flip_tol":  0.030,   # 3 %
    "long_entry_buf": 0.005,   # 0.5 %
    "long_stop_pct":  0.015,   # 1.5 %

    # ── Short entry ───────────────────────────────────────────────────────────
    # Shorts are harder to time so the rules are deliberately stricter:
    #
    # short_flip_tol  : tighter than long (2 % vs 3 %) — fewer shorts qualify.
    # short_entry_buf : enter if price is within this % BELOW VAH
    #                   (price must rally up to resistance before we sell)
    # short_stop_pct  : tighter stop ABOVE VAH (1 % vs 1.5 % for longs)
    #                   Shorts are cut faster to limit damage from short squeezes.
    # short_ema_slope : True  → EMA must also be FALLING to allow a short entry.
    #                           This is the extra strictness filter.
    #                   False → only require price < EMA (same as long filter).
    "short_flip_tol":  0.020,   # 2 %
    "short_entry_buf": 0.005,   # 0.5 %
    "short_stop_pct":  0.010,   # 1 %
    "short_ema_slope": True,    # require confirmed declining EMA for shorts

    # ── Exits (same logic for long and short, mirrored) ───────────────────────
    # Positions are scaled out in three stages:
    #   Stage 1 at POC     → sell/cover poc_exit_frac of position
    #   Stage 2 at VAH/VAL → sell/cover vah_exit_frac of what remains
    #   Stage 3 at ext     → sell/cover the rest (the "runner")
    #
    # ext_mult : size of the extension target relative to the VAH–POC range.
    #   1.0 → extension = VAH + 1 × (VAH − POC) for longs  (one measured move)
    #         extension = VAL − 1 × (POC − VAL) for shorts
    #   Increase to hold the runner longer; decrease if extension is rarely hit.
    "poc_exit_frac": 0.50,   # sell 50 % at POC
    "vah_exit_frac": 0.70,   # sell 70 % of remainder at VAH (long) / VAL (short)
    "ext_mult":      1.0,

    # ── Risk & capital ────────────────────────────────────────────────────────
    # risk_pct    : % of current capital risked on each trade.
    #               0.005 = 0.5 %   0.01 = 1 %   0.02 = 2 %
    #               Lower is more conservative. Start at 0.5–1 % when testing.
    # max_pos_pct : hard cap on position size as a fraction of capital.
    #               Prevents a very tight stop from creating a huge position.
    # capital     : starting account size in USD.
    "capital":     10_000,
    "risk_pct":    0.010,
    "max_pos_pct": 0.30,
}


# ══════════════════════════════════════════════════════════════════════════════
#  COLOURS  (dark theme)
#  ⚙ Change hex codes here to restyle all charts.
# ══════════════════════════════════════════════════════════════════════════════
_C = {
    "bg":     "#0d1117",
    "panel":  "#161b22",
    "border": "#30363d",
    "text":   "#e6edf3",
    "muted":  "#8b949e",
    "green":  "#3fb950",
    "red":    "#f85149",
    "orange": "#ffa657",
    "blue":   "#58a6ff",
    "yellow": "#e3b341",
    "purple": "#bc8cff",
}


# ══════════════════════════════════════════════════════════════════════════════
#  DATA
# ══════════════════════════════════════════════════════════════════════════════

def _resample(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Aggregate finer bars into coarser ones (e.g. 1H → 4H)."""
    agg = {"Open": "first", "High": "max", "Low": "min",
           "Close": "last", "Volume": "sum"}
    return (df.resample(rule, label="right", closed="right")
              .agg(agg)
              .dropna(subset=["Close"]))


def _synthetic(n: int, interval: str) -> pd.DataFrame:
    """Generate realistic fake OHLCV when live data is unavailable."""
    rng = np.random.default_rng(42)
    sigma, S0 = 0.012, 18_000.0
    v = np.full(n, sigma)
    for i in range(1, n):
        v[i] = 0.90 * v[i-1] + 0.10 * sigma * abs(rng.standard_normal()) + 0.001
    c  = S0 * np.exp(np.cumsum(rng.standard_normal(n) * v + 0.0003))
    hl = c * v * 2.2
    h  = c + hl * rng.uniform(0.3, 0.7, n)
    lo = c - hl * rng.uniform(0.3, 0.7, n)
    op = np.roll(c, 1); op[0] = S0
    vol = rng.integers(300_000, 2_000_000, n)
    freq = {"1m": "1min", "5m": "5min", "15m": "15min",
            "30m": "30min", "1h": "1h", "4h": "4h"}.get(interval, "5min")
    idx = pd.date_range(end=pd.Timestamp.now().normalize(), periods=n, freq=freq)
    return pd.DataFrame({"Open": op, "High": h, "Low": lo,
                         "Close": c, "Volume": vol}, index=idx)


def get_data(ticker: str, interval: str, days: int) -> pd.DataFrame:
    """Download from Yahoo Finance or fall back to synthetic data."""
    tf = TIMEFRAMES[interval]
    days = min(days, tf["max_days"])

    if _HAS_YF:
        end   = pd.Timestamp.now()
        start = end - pd.Timedelta(days=days)
        try:
            print(f"\n  Fetching {ticker} [{interval}]  "
                  f"{start.date()} → {end.date()} …")
            raw = yf.download(
                ticker,
                start=start.strftime("%Y-%m-%d"),
                end=end.strftime("%Y-%m-%d"),
                interval=tf["yf"],
                auto_adjust=True,
                progress=False,
            )
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.droplevel(1)
            raw.dropna(inplace=True)
            if len(raw) > 30:
                if tf["resample"]:
                    raw = _resample(raw, tf["resample"])
                print(f"  {len(raw)} bars loaded")
                return raw
            print(f"  Only {len(raw)} bars returned — using synthetic data")
        except Exception as e:
            print(f"  yfinance error ({e}) — using synthetic data")

    bars_est = max(int(days * tf["lookback"] * 2.5), 500)
    df = _synthetic(bars_est, interval)
    if tf["resample"]:
        df = _resample(df, tf["resample"])
    print(f"  Synthetic: {len(df)} {interval} bars")
    return df


# ══════════════════════════════════════════════════════════════════════════════
#  VOLUME PROFILE
#  ──────────────
#  Returns (poc, val, vah) for a chunk of OHLCV bars.
#    POC = price bucket with the most traded volume
#    VAL = bottom of the 70% value area (expands outward from POC)
#    VAH = top of the 70% value area
# ══════════════════════════════════════════════════════════════════════════════

def compute_vp(chunk: pd.DataFrame, bins: int) -> tuple:
    lo, hi = float(chunk["Low"].min()), float(chunk["High"].max())
    if hi - lo < 1e-8:
        m = (lo + hi) / 2
        return m, m, m

    edges = np.linspace(lo, hi, bins + 1)
    mids  = (edges[:-1] + edges[1:]) / 2
    vol   = np.zeros(bins)

    for k in range(len(chunk)):
        b_lo = float(chunk["Low"].iloc[k])
        b_hi = float(chunk["High"].iloc[k])
        b_v  = float(chunk["Volume"].iloc[k])
        span = b_hi - b_lo
        if span < 1e-10:
            idx = min(int((b_lo - lo) / (hi - lo) * bins), bins - 1)
            vol[idx] += b_v
        else:
            overlap = np.maximum(
                0.0,
                np.minimum(edges[1:], b_hi) - np.maximum(edges[:-1], b_lo),
            )
            vol += b_v * overlap / span

    poc_i  = int(np.argmax(vol))
    target = vol.sum() * 0.70
    li = hi_i = poc_i
    acc = vol[poc_i]
    while acc < target:
        al = vol[li   - 1] if li   > 0        else -1.0
        ah = vol[hi_i + 1] if hi_i < bins - 1 else -1.0
        if al < 0 and ah < 0:
            break
        if ah >= al:
            hi_i += 1; acc += vol[hi_i]
        else:
            li   -= 1; acc += vol[li]

    return mids[poc_i], mids[li], mids[hi_i]   # poc, val, vah


# ══════════════════════════════════════════════════════════════════════════════
#  BACKTEST
#  ─────────
#  Bar-by-bar simulation. Detects flips, enters positions, manages exits.
#  Returns trades, equity curve, VP levels (for the chart), entries, exits.
# ══════════════════════════════════════════════════════════════════════════════

def run_backtest(df: pd.DataFrame, cfg: dict, lookback: int):
    bins    = cfg["vp_bins"]
    ema_n   = cfg["ema_period"]
    capital = float(cfg["capital"])

    ema_series = df["Close"].ewm(span=ema_n, adjust=False).mean()

    # ── Open position state ───────────────────────────────────────────────────
    in_pos    = False
    direction = None      # "long" or "short"
    shares    = 0.0
    avg_px    = 0.0
    cost      = 0.0       # capital tied up (returned + pnl on exit)
    sl        = 0.0
    tp1 = tp2 = tp3 = 0.0
    tp1_done  = False
    tp2_done  = False

    trades  = []   # completed trade records
    equity  = []   # portfolio value at each bar
    levels  = []   # VP levels at each bar (for chart)
    entries = []   # entry markers  (for chart)
    exits   = []   # exit markers   (for chart)

    min_i = lookback * 2 + ema_n   # bars needed before first signal is possible

    for i in range(len(df)):
        bar   = df.iloc[i]
        close = float(bar["Close"])
        high  = float(bar["High"])
        low   = float(bar["Low"])
        dt    = df.index[i]

        # Compute VP levels only once we have enough history
        if i < min_i:
            equity.append(capital + shares * close)
            continue

        poc_c, val_c, vah_c = compute_vp(df.iloc[i - lookback     : i], bins)
        poc_p, val_p, vah_p = compute_vp(df.iloc[i - lookback * 2 : i - lookback], bins)
        ema_v = float(ema_series.iloc[i])

        levels.append({"dt": dt, "poc": poc_c, "val": val_c,
                       "vah": vah_c, "ema": ema_v})

        # ── EXIT LOGIC ────────────────────────────────────────────────────────
        if in_pos:
            is_long  = direction == "long"
            hit_sl   = (is_long and low  <= sl) or (not is_long and high >= sl)
            hit_tp1  = not tp1_done and (
                (is_long and high >= tp1) or (not is_long and low <= tp1))
            hit_tp2  = tp1_done and not tp2_done and (
                (is_long and high >= tp2) or (not is_long and low <= tp2))
            hit_tp3  = tp2_done and shares > 0 and (
                (is_long and high >= tp3) or (not is_long and low <= tp3))

            sign = 1.0 if is_long else -1.0   # profit direction multiplier

            if hit_sl:
                pnl     = (sl - avg_px) * shares * sign
                capital += cost + pnl
                trades.append({"dt": dt, "dir": direction, "type": "stop",
                               "entry": avg_px, "exit": sl, "pnl": pnl})
                exits.append({"dt": dt, "price": sl, "type": "stop"})
                in_pos = False; shares = cost = 0.0
                tp1_done = tp2_done = False

            elif hit_tp1:
                cs      = shares * cfg["poc_exit_frac"]
                pnl_p   = (tp1 - avg_px) * cs * sign
                capital += cs * avg_px + pnl_p
                cost    -= cs * avg_px
                shares  -= cs
                tp1_done = True
                exits.append({"dt": dt, "price": tp1, "type": "tp1"})

            elif hit_tp2:
                cs      = shares * cfg["vah_exit_frac"]
                pnl_p   = (tp2 - avg_px) * cs * sign
                capital += cs * avg_px + pnl_p
                cost    -= cs * avg_px
                shares  -= cs
                tp2_done = True
                exits.append({"dt": dt, "price": tp2, "type": "tp2"})

            elif hit_tp3:
                pnl     = (tp3 - avg_px) * shares * sign
                capital += cost + pnl
                trades.append({"dt": dt, "dir": direction, "type": "target",
                               "entry": avg_px, "exit": tp3, "pnl": pnl})
                exits.append({"dt": dt, "price": tp3, "type": "tp3"})
                in_pos = False; shares = cost = 0.0
                tp1_done = tp2_done = False

        # ── SIGNAL DETECTION & ENTRY ──────────────────────────────────────────
        if not in_pos:

            # ── LONG: value area stepped UP ──────────────────────────────────
            # new VAL is near old VAH → flipped from resistance to support
            # Enter when price pulls BACK DOWN into VAL after the break above.
            long_flip = (
                (val_c > val_p) and
                (vah_p > 0) and
                (abs(val_c - vah_p) / vah_p < cfg["long_flip_tol"]
                 or val_c >= vah_p)
            )
            entry_long = val_c * (1 + cfg["long_entry_buf"])
            stop_long  = val_c * (1 - cfg["long_stop_pct"])

            if (long_flip
                    and close > ema_v           # price is above EMA → uptrend
                    and low   <= entry_long     # this bar touched the VAL zone
                    and close >  stop_long):    # but didn't break through the stop
                fill      = min(close, entry_long)
                risk_cash = capital * cfg["risk_pct"]
                risk_pts  = max(fill - stop_long, 1e-6)
                n_sh      = risk_cash / risk_pts
                order     = min(n_sh * fill, capital * cfg["max_pos_pct"])
                if order >= 10 and capital >= order:
                    shares = n_sh; avg_px = fill; cost = order
                    sl     = stop_long
                    tp1    = poc_c
                    tp2    = vah_c
                    tp3    = vah_c + cfg["ext_mult"] * (vah_c - poc_c)
                    tp1_done = tp2_done = False
                    in_pos = True; direction = "long"; capital -= order
                    entries.append({"dt": dt, "price": fill, "dir": "long"})

            # ── SHORT: value area stepped DOWN ───────────────────────────────
            # new VAH is near old VAL → flipped from support to resistance.
            # Enter when price RALLIES BACK UP into VAH after the break below.
            #
            # Extra strictness vs long:
            #   1. Tighter flip tolerance (short_flip_tol < long_flip_tol)
            #   2. EMA must be FALLING (not just price below EMA)
            if not in_pos:
                ema_falling = (
                    float(ema_series.iloc[i])
                    < float(ema_series.iloc[max(0, i - ema_n)])
                )
                short_ok = (close < ema_v) and (
                    not cfg["short_ema_slope"] or ema_falling
                )
                short_flip = (
                    (vah_c < vah_p) and
                    (val_p > 0) and
                    (abs(vah_c - val_p) / val_p < cfg["short_flip_tol"]
                     or vah_c <= val_p)
                )
                entry_short = vah_c * (1 - cfg["short_entry_buf"])
                stop_short  = vah_c * (1 + cfg["short_stop_pct"])

                if (short_flip
                        and short_ok              # price below falling EMA
                        and high  >= entry_short  # this bar touched the VAH zone
                        and close <  stop_short): # but didn't blow through the stop
                    fill      = max(close, entry_short)
                    risk_cash = capital * cfg["risk_pct"]
                    risk_pts  = max(stop_short - fill, 1e-6)
                    n_sh      = risk_cash / risk_pts
                    order     = min(n_sh * fill, capital * cfg["max_pos_pct"])
                    if order >= 10 and capital >= order:
                        shares = n_sh; avg_px = fill; cost = order
                        sl     = stop_short
                        tp1    = poc_c
                        tp2    = val_c
                        tp3    = val_c - cfg["ext_mult"] * (poc_c - val_c)
                        tp1_done = tp2_done = False
                        in_pos = True; direction = "short"; capital -= order
                        entries.append({"dt": dt, "price": fill, "dir": "short"})

        equity.append(capital + shares * close)

    # Mark open position to market at final bar
    if in_pos and shares > 0:
        lp  = float(df["Close"].iloc[-1])
        pnl = (lp - avg_px) * shares * (1.0 if direction == "long" else -1.0)
        capital += cost + pnl
        trades.append({"dt": df.index[-1], "dir": direction, "type": "open",
                       "entry": avg_px, "exit": lp, "pnl": pnl})

    return trades, equity, levels, entries, exits


# ══════════════════════════════════════════════════════════════════════════════
#  STATISTICS
# ══════════════════════════════════════════════════════════════════════════════

def compute_stats(trades: list, equity: list, capital: float) -> dict:
    if not trades or not equity:
        return {}

    eq   = pd.Series(equity)
    final = float(eq.iloc[-1])
    ret   = (final - capital) / capital * 100

    peak  = eq.cummax()
    dd    = float(((eq - peak) / peak * 100).min())

    dr    = eq.pct_change().dropna()
    sharpe = dr.mean() / dr.std() * (252 ** 0.5) if dr.std() > 0 else 0.0

    t  = pd.DataFrame(trades)
    # Only count completed trades (stop + target), exclude still-open marks
    closed = t[t["type"].isin(["stop", "target"])] if "type" in t.columns else t

    def _side(direction):
        s  = closed[closed["dir"] == direction] if "dir" in closed.columns else pd.DataFrame()
        if s.empty:
            return {"trades": 0, "wins": 0, "wr": "—", "avg_win": "—", "avg_loss": "—"}
        wins = int((s["pnl"] > 0).sum())
        n    = len(s)
        return {
            "trades":   n,
            "wins":     wins,
            "wr":       f"{wins/n*100:.0f}%",
            "avg_win":  f"${s.loc[s.pnl>0,'pnl'].mean():,.0f}"  if wins   else "—",
            "avg_loss": f"${s.loc[s.pnl<=0,'pnl'].mean():,.0f}" if wins<n else "—",
        }

    return {
        "Return":      f"{ret:+.1f}%",
        "Max DD":      f"{dd:.1f}%",
        "Sharpe":      f"{sharpe:.2f}",
        "Long trades": _side("long")["trades"],
        "Long WR":     _side("long")["wr"],
        "Short trades":_side("short")["trades"],
        "Short WR":    _side("short")["wr"],
        "Final equity":f"${final:,.0f}",
    }


# ══════════════════════════════════════════════════════════════════════════════
#  CHART
# ══════════════════════════════════════════════════════════════════════════════

def _style_ax(ax):
    ax.set_facecolor(_C["panel"])
    ax.tick_params(colors=_C["muted"], labelsize=8)
    for sp in ax.spines.values():
        sp.set_edgecolor(_C["border"])
    ax.grid(color=_C["border"], alpha=0.5, lw=0.4)


def plot(df: pd.DataFrame, trades: list, equity: list,
         levels: list, entries: list, exits: list,
         cfg: dict, stats: dict, interval: str, out: str):
    """
    3-row chart:
      Row 0 — Price line + EMA + rolling VAH/POC/VAL + trade markers
      Row 1 — Volume bars
      Row 2 — Equity curve + drawdown
    ⚙ Change figsize or height_ratios to resize the panels.
    """
    plt.style.use("dark_background")
    fig = plt.figure(figsize=(20, 12), facecolor=_C["bg"])
    gs  = gridspec.GridSpec(
        3, 1, figure=fig,
        height_ratios=[4, 1, 1.5],
        hspace=0.10,
        left=0.05, right=0.97, top=0.93, bottom=0.06,
    )
    ax_p  = fig.add_subplot(gs[0])
    ax_v  = fig.add_subplot(gs[1], sharex=ax_p)
    ax_eq = fig.add_subplot(gs[2], sharex=ax_p)
    for ax in (ax_p, ax_v, ax_eq):
        _style_ax(ax)

    # ── Restrict to recent bars so intraday charts stay readable ──────────────
    # ⚙ Change 350 to show more or fewer bars on the price panel.
    SHOW_BARS = 350
    if len(df) > SHOW_BARS:
        plot_df = df.iloc[-SHOW_BARS:]
    else:
        plot_df = df

    lv_df = pd.DataFrame(levels).set_index("dt")
    lv_df = lv_df[lv_df.index >= plot_df.index[0]]

    # ── Price line ────────────────────────────────────────────────────────────
    ax_p.plot(plot_df.index, plot_df["Close"],
              color=_C["blue"], lw=1.0, zorder=2)
    if not lv_df.empty:
        ax_p.plot(lv_df.index, lv_df["ema"],
                  color=_C["yellow"], lw=0.9, ls="--", alpha=0.8, label="EMA")
        ax_p.plot(lv_df.index, lv_df["vah"],
                  color=_C["green"],  lw=0.8, ls="--", alpha=0.85, label="VAH")
        ax_p.plot(lv_df.index, lv_df["poc"],
                  color=_C["orange"], lw=0.8, ls="-",  alpha=0.85, label="POC")
        ax_p.plot(lv_df.index, lv_df["val"],
                  color=_C["red"],    lw=0.8, ls="--", alpha=0.85, label="VAL")
        # Shade value area
        ax_p.fill_between(lv_df.index, lv_df["val"], lv_df["vah"],
                          alpha=0.05, color=_C["blue"])

    # ── Trade markers ─────────────────────────────────────────────────────────
    for e in entries:
        if e["dt"] < plot_df.index[0]:
            continue
        color  = _C["green"] if e["dir"] == "long" else _C["red"]
        marker = "^" if e["dir"] == "long" else "v"
        ax_p.scatter(e["dt"], e["price"], marker=marker, s=90,
                     color=color, zorder=6, edgecolors=_C["bg"], lw=0.5)

    exit_colors = {
        "stop": _C["red"], "tp1": _C["orange"],
        "tp2": _C["green"], "tp3": _C["purple"],
    }
    for x in exits:
        if x["dt"] < plot_df.index[0]:
            continue
        ax_p.scatter(x["dt"], x["price"], marker="D", s=45, zorder=6,
                     color=exit_colors.get(x["type"], _C["text"]),
                     edgecolors=_C["bg"], lw=0.4)

    ax_p.set_xlim(plot_df.index[0], plot_df.index[-1])
    ax_p.set_ylim(plot_df["Low"].min() * 0.999,
                  plot_df["High"].max() * 1.001)

    ax_p.legend(handles=[
        mpatches.Patch(color=_C["blue"],   label="Price"),
        mpatches.Patch(color=_C["yellow"], label=f"EMA({cfg['ema_period']})"),
        mpatches.Patch(color=_C["green"],  label="VAH"),
        mpatches.Patch(color=_C["orange"], label="POC"),
        mpatches.Patch(color=_C["red"],    label="VAL"),
        mpatches.Patch(color=_C["green"],  label="▲ Long entry"),
        mpatches.Patch(color=_C["red"],    label="▼ Short entry"),
        mpatches.Patch(color=_C["red"],    label="◆ Stop"),
        mpatches.Patch(color=_C["purple"], label="◆ Extension exit"),
    ], loc="upper left", fontsize=7, facecolor=_C["panel"],
       labelcolor=_C["text"], framealpha=0.9, ncol=3)

    ax_p.set_title(
        f"{cfg['ticker']}  [{interval}]  ·  VAH/VAL Flip  ·  "
        f"Long ▲  /  Short ▼  (showing last {len(plot_df)} bars)",
        color=_C["text"], fontsize=11, fontweight="bold", pad=8,
    )
    ax_p.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    plt.setp(ax_p.get_xticklabels(), visible=False)

    # ── Volume bars ───────────────────────────────────────────────────────────
    bull = plot_df["Close"] >= plot_df["Open"]
    ax_v.bar(plot_df.index,
             plot_df["Volume"],
             color=np.where(bull, _C["green"], _C["red"]),
             alpha=0.6, width=0.6 / (len(plot_df) / 100))
    ax_v.set_ylabel("Volume", color=_C["muted"], fontsize=8)
    ax_v.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"{x/1e6:.1f}M"))
    plt.setp(ax_v.get_xticklabels(), visible=False)

    # ── Equity + drawdown ─────────────────────────────────────────────────────
    if equity:
        eq_s = pd.Series(equity, index=df.index[-len(equity):])
        ax_eq.plot(eq_s.index, eq_s, color=_C["green"], lw=1.5)
        ax_eq.axhline(cfg["capital"], color=_C["muted"], lw=0.7, ls="--", alpha=0.5)
        ax_eq.fill_between(eq_s.index, cfg["capital"], eq_s,
                           where=eq_s >= cfg["capital"],
                           color=_C["green"], alpha=0.10)
        ax_eq.fill_between(eq_s.index, cfg["capital"], eq_s,
                           where=eq_s < cfg["capital"],
                           color=_C["red"], alpha=0.15)

        # Overlay drawdown as a faint red fill on the right y-axis
        ax_dd = ax_eq.twinx()
        peak  = eq_s.cummax()
        dd    = (eq_s - peak) / peak * 100
        ax_dd.fill_between(eq_s.index, dd, 0, color=_C["red"], alpha=0.18)
        ax_dd.plot(eq_s.index, dd, color=_C["red"], lw=0.6, alpha=0.6)
        ax_dd.set_ylabel("DD %", color=_C["red"], fontsize=7)
        ax_dd.tick_params(colors=_C["red"], labelsize=7)
        ax_dd.set_ylim(dd.min() * 1.5, 5)
        for sp in ax_dd.spines.values():
            sp.set_edgecolor(_C["border"])

    ax_eq.set_ylabel("Equity", color=_C["muted"], fontsize=8)
    ax_eq.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax_eq.xaxis.set_major_formatter(mdates.DateFormatter("%d %b %H:%M"))
    ax_eq.tick_params(axis="x", rotation=20, labelsize=7)

    # ── Stats text block ──────────────────────────────────────────────────────
    if stats:
        lines = [f"{k}: {v}" for k, v in stats.items()]
        ax_eq.text(
            0.99, 0.97, "\n".join(lines),
            transform=ax_eq.transAxes,
            color=_C["text"], fontsize=7.5, fontfamily="monospace",
            va="top", ha="right",
            bbox=dict(facecolor=_C["panel"], edgecolor=_C["border"],
                      alpha=0.9, pad=5),
        )

    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=_C["bg"])
    print(f"\n  Chart saved → {out}")
    plt.show()


# ══════════════════════════════════════════════════════════════════════════════
#  INTERACTIVE PROMPT
# ══════════════════════════════════════════════════════════════════════════════

def _input(prompt: str, default=None, cast=str, choices=None, required=False):
    """Prompt the user, validate, and return the answer (or default on Enter)."""
    while True:
        try:
            raw = input(prompt).strip()
        except EOFError:
            return default
        if not raw:
            if required:
                print("  ✗  Required — please type a value.")
                continue
            return default
        try:
            val = cast(raw)
        except (ValueError, TypeError):
            print(f"  ✗  Expected a {cast.__name__}.")
            continue
        if choices is not None and val not in choices:
            print(f"  ✗  Enter a number between {min(choices)} and {max(choices)}.")
            continue
        return val


def prompt_settings(cfg: dict) -> tuple[str, str, int]:
    """
    Ask the user for ticker, timeframe, and days of history.
    Returns (ticker, interval, days).
    """
    print()
    print("╔══════════════════════════════════════════════════════╗")
    print("║   VAH / VAL Flip  ·  Intraday Strategy Setup        ║")
    print("╚══════════════════════════════════════════════════════╝")
    print()

    # ── Ticker ────────────────────────────────────────────────────────────────
    print("  Ticker  (Yahoo Finance symbol)")
    print("  Futures: NQ=F  ES=F  GC=F  CL=F")
    print("  Stocks : AAPL  SPY   QQQ   TSLA")
    ticker = _input(f"  → [{cfg['ticker']}]: ", default=cfg["ticker"]).upper()
    print()

    # ── Timeframe ─────────────────────────────────────────────────────────────
    print("  Timeframe")
    print(f"  {'#':>2}   {'TF':<5}  Description")
    print(f"  {'─'*60}")
    for i, key in enumerate(_TF_KEYS, 1):
        default_tag = " ◀" if key == "5m" else ""
        print(f"  {i:>2}   {key:<5}  {TIMEFRAMES[key]['desc']}{default_tag}")
    print()

    # User MUST type a number — pressing Enter alone is rejected
    n = _input(
        f"  → type 1–{len(_TF_KEYS)} and press Enter: ",
        default=None, cast=int,
        choices=list(range(1, len(_TF_KEYS) + 1)),
        required=True,
    )
    interval = _TF_KEYS[n - 1]
    tf       = TIMEFRAMES[interval]
    print(f"  ✓  {interval}  —  {tf['desc']}")
    print()

    # ── Days of history ───────────────────────────────────────────────────────
    max_d   = tf["max_days"]
    default_d = min(cfg["days"], max_d)
    print(f"  Days of history  (max for {interval}: {max_d})")
    days = _input(
        f"  → [{default_d}]: ",
        default=default_d, cast=int,
    )
    days = max(5, min(days, max_d))
    print(f"  ✓  {days} days")
    print()

    return ticker, interval, days


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    quick = "--quick" in sys.argv   # skip prompt, run with CFG defaults

    if quick:
        ticker   = CFG["ticker"]
        interval = "5m"
        days     = CFG["days"]
        print(f"\n  Quick mode — {ticker} [{interval}]  {days} days")
    else:
        ticker, interval, days = prompt_settings(CFG)

    tf       = TIMEFRAMES[interval]
    lookback = tf["lookback"]

    # Override CFG with prompted values
    cfg          = CFG.copy()
    cfg["ticker"] = ticker
    cfg["days"]   = days

    # Fetch data
    df = get_data(ticker, interval, days)
    if df.empty or len(df) < lookback * 2 + cfg["ema_period"] + 5:
        print("  Not enough data to run the strategy. Try a longer date range.")
        return

    # Run backtest
    print(f"  Running backtest  ({len(df)} bars, lookback={lookback}) …")
    trades, equity, levels, entries, exits = run_backtest(df, cfg, lookback)

    # Stats
    stats = compute_stats(trades, equity, cfg["capital"])
    print()
    print("  ── Results " + "─" * 40)
    for k, v in stats.items():
        print(f"  {k:<18} {v}")

    if trades:
        t = pd.DataFrame(trades)
        if "dir" in t.columns and "type" in t.columns:
            print()
            print("  ── Exit breakdown by direction " + "─" * 20)
            for d in ("long", "short"):
                sub = t[(t["dir"] == d) & (t["type"].isin(["stop", "target"]))]
                if not sub.empty:
                    print(f"  {d.capitalize()}: "
                          + sub["type"].value_counts().to_string(index=True, header=False))

    # Chart
    out = f"flip_{ticker.lower().replace('=', '')}_{interval}.png"
    plot(df, trades, equity, levels, entries, exits, cfg, stats, interval, out)


if __name__ == "__main__":
    main()
