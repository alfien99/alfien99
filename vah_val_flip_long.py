#!/usr/bin/env python3
"""
Strategy 1: VAH→VAL Flip Long Only

Pattern: When a prior period's VAH becomes the current period's VAL,
that level is a high-conviction support zone. Price broke above it once
(showing buyers dominated), then the market re-anchors its value area there.
A pullback to that flipped level is a long entry.

Entry rules:
  1. Compute rolling VP for the past `lookback` bars → get VAL(t), VAH(t)
  2. Compute rolling VP for the prior `lookback` bars → VAL(t-1), VAH(t-1)
  3. If VAL(t) > VAL(t-1) AND VAL(t) is near/above VAH(t-1) → a flip occurred
  4. Enter long when bar's LOW touches the VAL zone (limit order at VAL*(1+buf))
  5. Only enter if price is above the lagged EMA (look-ahead-free trend filter)
  6. flip_consumed flag blocks re-entry on the same flip regime after stop-out;
     resets only after ≥3 consecutive bars where flip is False

Section 1 fixes applied:
  - EMA look-ahead bias removed: uses ema.shift(1) so bar-i filter uses bar-(i-1) EMA
  - Flip-consumed guard: suppresses re-entry on same regime after stop-out
  - sell_stop() for stop exits: 0.15% slippage vs 0.05% for limit targets
  - Synthetic data runs are loudly labelled everywhere

Exits:
  - 50% at POC  (sell_lim)
  - 70% of remainder at VAH  (sell_lim)
  - Rest at VAH + 1× (VAH - POC) extension  (sell_lim)
  - Stop: below VAL × (1 - stop_pct)  (sell_stop — wider slippage)

Usage:
  python vah_val_flip_long.py
  python vah_val_flip_long.py --ticker QQQ --start 2020-01-01 --end 2025-01-01
"""

import argparse
import warnings

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec

warnings.filterwarnings("ignore")

try:
    import yfinance as yf
    _YF_AVAILABLE = True
except ImportError:
    _YF_AVAILABLE = False


# ── Transaction costs ──────────────────────────────────────────────────────────
COMMISSION = 0.0005   # 0.05% per side
SLIP_LIM   = 0.0005   # 0.05% slippage on limit fills (target exits, entries)
SLIP_STOP  = 0.0015   # 0.15% slippage on stop-loss exits (wider spread under stress)

def buy_lim(p):   return p * (1 + SLIP_LIM  + COMMISSION)
def sell_lim(p):  return p * (1 - SLIP_LIM  - COMMISSION)
def sell_stop(p): return p * (1 - SLIP_STOP - COMMISSION)


# ── Data ───────────────────────────────────────────────────────────────────────

def _synthetic_ohlcv(ticker, start, end):
    rng   = np.random.default_rng(abs(hash(ticker)) % (2**31))
    dates = pd.bdate_range(start, end)
    n     = len(dates)
    mu, sigma, S0 = 0.0004, 0.013, 18_000.0
    vols = np.full(n, sigma)
    for i in range(1, n):
        vols[i] = 0.91 * vols[i-1] + 0.09 * sigma * abs(rng.standard_normal()) + 0.001
    closes  = S0 * np.exp(np.cumsum(rng.standard_normal(n) * vols + mu))
    hl      = closes * vols * 2.5
    highs   = closes + hl * rng.uniform(0.3, 0.7, n)
    lows    = closes - hl * rng.uniform(0.3, 0.7, n)
    opens   = np.roll(closes, 1); opens[0] = S0
    vols_v  = rng.integers(1_000_000, 5_000_000, n)
    df = pd.DataFrame({"Open": opens, "High": highs, "Low": lows,
                       "Close": closes, "Volume": vols_v}, index=dates)
    df.index.name = "Date"
    return df


def _download(ticker, start, end):
    """Return (df, is_synthetic). is_synthetic=True when yfinance is unavailable."""
    if _YF_AVAILABLE:
        try:
            raw = yf.download(ticker, start=start, end=end,
                              auto_adjust=True, progress=False)
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.droplevel(1)
            raw.dropna(inplace=True)
            if len(raw) > 20:
                print(f"  Downloaded {len(raw)} bars.")
                return raw, False
        except Exception as e:
            print(f"  Yahoo Finance error ({e}) — using synthetic data.")
    print(f"  Generating synthetic OHLCV for {ticker} …")
    return _synthetic_ohlcv(ticker, start, end), True


# ── Volume profile ─────────────────────────────────────────────────────────────

def compute_vp(ohlcv, bins=100):
    lo, hi = float(ohlcv["Low"].min()), float(ohlcv["High"].max())
    if hi <= lo + 1e-8:
        mid = (hi + lo) / 2
        return mid, mid, mid, np.array([mid]), np.array([1.0])
    edges = np.linspace(lo, hi, bins + 1)
    mids  = (edges[:-1] + edges[1:]) / 2
    vol   = np.zeros(bins)
    for k in range(len(ohlcv)):
        rng = float(ohlcv["High"].iloc[k]) - float(ohlcv["Low"].iloc[k])
        v   = float(ohlcv["Volume"].iloc[k])
        if rng < 1e-10:
            idx = min(int((float(ohlcv["Low"].iloc[k]) - lo) / (hi - lo) * bins), bins - 1)
            vol[idx] += v
        else:
            overlap = np.maximum(0.0,
                np.minimum(edges[1:], float(ohlcv["High"].iloc[k]))
                - np.maximum(edges[:-1], float(ohlcv["Low"].iloc[k])))
            vol += v * overlap / rng
    poc_idx = int(np.argmax(vol))
    poc     = mids[poc_idx]
    target  = vol.sum() * 0.70
    li = hi_i = poc_idx
    acc = vol[poc_idx]
    while acc < target:
        al = vol[li - 1]   if li   > 0        else -1.0
        ah = vol[hi_i + 1] if hi_i < bins - 1 else -1.0
        if al < 0 and ah < 0:
            break
        if al >= ah:
            li -= 1; acc += vol[li]
        else:
            hi_i += 1; acc += vol[hi_i]
    return poc, mids[li], mids[hi_i], mids, vol


# ── Backtest ───────────────────────────────────────────────────────────────────

def run_backtest(cfg):
    raw, is_synthetic = _download(cfg["ticker"], cfg["start"], cfg["end"])
    lookback = cfg["lookback"]
    bins     = cfg["vp_bins"]
    tol      = cfg["flip_tolerance"]
    buf      = cfg["entry_buffer"]
    stop_p   = cfg["stop_pct"]
    ema_win  = cfg["ema_window"]
    capital  = float(cfg["initial_capital"])

    # ── Fix 1: EMA look-ahead bias ────────────────────────────────────────────
    # The unshifted EMA at bar i is computed using bar i's close, so checking
    # close[i] > ema[i] is contaminated.  Shifting by 1 means bar i's trend
    # filter uses the EMA value that was known BEFORE bar i opened.
    ema_raw    = raw["Close"].ewm(span=ema_win, adjust=False).mean()
    ema_lagged = ema_raw.shift(1)   # look-ahead-free

    min_i = max(lookback * 2, ema_win + 5)

    # Print before/after comparison on three evenly-spaced sample bars
    step = max((len(raw) - min_i) // 3, 1)
    samples = [min_i, min_i + step, min_i + 2 * step]
    print("\n  EMA look-ahead fix — before vs. after (3 sample bars):")
    print(f"  {'Date':<12} {'Close':>10} {'EMA(orig)':>12} {'EMA(lag)':>12}"
          f" {'Trend-orig':>10} {'Trend-fix':>10}")
    print("  " + "-" * 68)
    for idx in samples:
        if idx >= len(raw):
            break
        d  = raw.index[idx].strftime("%Y-%m-%d")
        c  = float(raw["Close"].iloc[idx])
        e0 = float(ema_raw.iloc[idx])
        e1 = float(ema_lagged.iloc[idx]) if not np.isnan(ema_lagged.iloc[idx]) else float("nan")
        t0 = "UP" if c > e0 else "DOWN"
        t1 = "UP" if (not np.isnan(e1) and c > e1) else "DOWN"
        changed = " ← changed" if t0 != t1 else ""
        print(f"  {d:<12} {c:>10.2f} {e0:>12.2f} {e1:>12.2f} {t0:>10} {t1:>10}{changed}")
    print()

    # ── Position state ────────────────────────────────────────────────────────
    in_pos     = False
    shares     = 0.0
    avg_px     = 0.0
    cost       = 0.0
    entry_stop = 0.0
    entry_poc  = 0.0
    entry_vah  = 0.0
    entry_ext  = 0.0
    poc_done   = False
    vah_done   = False

    # ── Fix 2: Flip-consumed guard ────────────────────────────────────────────
    # Once a trade is opened on a flip signal, further entries are suppressed
    # until flip has been False for at least 3 consecutive bars.  This prevents
    # re-entering the same stale regime after a stop-out.
    flip_consumed       = False
    flip_false_streak   = 0
    suppressed_reentries = 0

    # ── Fix 3: Stop slippage tracking ─────────────────────────────────────────
    stop_exit_records = []   # (planned_price, realized_price, shares_exited)

    equity_records = []
    level_records  = []
    trade_records  = []
    entry_marks    = []
    exit_marks     = []
    flip_marks     = []

    for i in range(min_i, len(raw)):
        date  = raw.index[i]
        close = float(raw["Close"].iloc[i])
        high  = float(raw["High"].iloc[i])
        low   = float(raw["Low"].iloc[i])

        poc_c, val_c, vah_c, _, _ = compute_vp(raw.iloc[i - lookback: i], bins)
        poc_p, val_p, vah_p, _, _ = compute_vp(raw.iloc[i - lookback * 2: i - lookback], bins)

        val_above_prev_val = val_c > val_p
        near_prev_vah = (vah_p > 0 and abs(val_c - vah_p) / vah_p < tol) or val_c >= vah_p
        flip = val_above_prev_val and near_prev_vah

        # Fix 1: use lagged EMA value (no look-ahead)
        ema_val = ema_lagged.iloc[i]
        uptrend = (not np.isnan(ema_val)) and close > float(ema_val)

        stop_lvl = val_c * (1 - stop_p)
        ext_lvl  = vah_c + cfg["ext_mult"] * (vah_c - poc_c)

        level_records.append({"date": date, "poc": poc_c, "val": val_c,
                               "vah": vah_c, "vah_prev": vah_p,
                               "flip": flip, "ema": float(ema_lagged.iloc[i])
                               if not np.isnan(ema_lagged.iloc[i]) else float(ema_raw.iloc[i])})
        if flip:
            flip_marks.append({"date": date, "level": val_c})

        # Fix 2: update flip_false_streak and reset flip_consumed
        if not flip:
            flip_false_streak += 1
            if flip_false_streak >= 3:
                flip_consumed = False        # new flip opportunity is clean
        else:
            flip_false_streak = 0            # streak broken; need 3 new False bars to reset

        # ── Exits ─────────────────────────────────────────────────────────────
        if in_pos:
            # Fix 3: stop exit uses sell_stop (wider slippage under stress)
            if low <= entry_stop:
                realized = sell_stop(entry_stop)
                stop_exit_records.append((entry_stop, realized, shares))
                pnl = (realized - avg_px) * shares
                capital += cost + pnl
                trade_records.append({"date": date, "type": "stop",
                                      "entry": avg_px, "exit": realized, "pnl": pnl})
                exit_marks.append({"date": date, "price": realized, "type": "stop"})
                in_pos = False; shares = cost = 0.0
                poc_done = vah_done = False

            elif not poc_done and high >= entry_poc:
                cs       = shares * cfg["poc_exit_frac"]
                realized = sell_lim(entry_poc)
                pnl_p    = (realized - avg_px) * cs
                capital += cs * avg_px + pnl_p
                cost    -= cs * avg_px
                shares  -= cs
                poc_done = True
                exit_marks.append({"date": date, "price": realized, "type": "poc"})

            elif poc_done and not vah_done and high >= entry_vah:
                cs       = shares * cfg["vah_exit_frac"]
                realized = sell_lim(entry_vah)
                pnl_p    = (realized - avg_px) * cs
                capital += cs * avg_px + pnl_p
                cost    -= cs * avg_px
                shares  -= cs
                vah_done = True
                exit_marks.append({"date": date, "price": realized, "type": "vah"})

            elif vah_done and shares > 0 and high >= entry_ext:
                realized = sell_lim(entry_ext)
                pnl      = (realized - avg_px) * shares
                capital += cost + pnl
                trade_records.append({"date": date, "type": "target",
                                      "entry": avg_px, "exit": realized, "pnl": pnl})
                exit_marks.append({"date": date, "price": realized, "type": "ext"})
                in_pos = False; shares = cost = 0.0
                poc_done = vah_done = False

        # ── Entry ─────────────────────────────────────────────────────────────
        # Limit order at VAL*(1+buf); filled when bar's low touches that level.
        entry_limit = val_c * (1 + buf)
        # Fix 2: also gate on flip_consumed to block same-regime re-entries
        can_enter = not in_pos and flip and not flip_consumed and uptrend and low <= entry_limit and close > stop_lvl
        suppressed = not in_pos and flip and flip_consumed and uptrend and low <= entry_limit and close > stop_lvl

        if suppressed:
            suppressed_reentries += 1

        if can_enter:
            raw_fill   = min(close, entry_limit)
            fill_px    = buy_lim(raw_fill)          # Fix 3: entry cost includes commission+slippage
            risk_per_sh = max(fill_px - sell_stop(stop_lvl), 1e-6)  # round-trip risk to stop
            risk_cash  = capital * cfg["risk_pct"]
            n_sh       = risk_cash / risk_per_sh
            order_cash = min(n_sh * fill_px, capital * cfg["max_position_pct"])

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
                flip_consumed = True             # Fix 2: mark this flip as consumed
                capital   -= order_cash
                entry_marks.append({"date": date, "price": fill_px})

        equity_records.append({"date": date, "equity": capital + shares * close})

    # Close any open position at final bar
    if in_pos and shares > 0:
        lp  = float(raw["Close"].iloc[-1])
        realized = sell_lim(lp)
        pnl = (realized - avg_px) * shares
        capital += cost + pnl
        trade_records.append({"date": raw.index[-1], "type": "expired",
                               "entry": avg_px, "exit": realized, "pnl": pnl})

    # ── Fix 3: stop slippage report ───────────────────────────────────────────
    n_stops = len(stop_exit_records)
    print(f"  Stop exits     : {n_stops}")
    if n_stops > 0:
        planned_pct  = (SLIP_LIM  + COMMISSION) * 100
        realized_pct = (SLIP_STOP + COMMISSION) * 100
        extra_costs  = sum((sell_lim(p) - sell_stop(p)) * s
                           for (p, _, s) in stop_exit_records)
        print(f"  Planned slippage (sell_lim) : {planned_pct:.3f}% per stop exit")
        print(f"  Realized slippage (sell_stop): {realized_pct:.3f}% per stop exit")
        print(f"  Extra cost from stop slippage: ${extra_costs:,.2f}")
    else:
        print(f"  (no stop exits — slippage comparison not applicable)")

    # ── Fix 2: suppressed re-entry report ─────────────────────────────────────
    print(f"  Suppressed re-entries (flip_consumed guard): {suppressed_reentries}")

    equity_df = pd.DataFrame(equity_records).set_index("date")
    levels_df = pd.DataFrame(level_records).set_index("date")
    trades_df = (pd.DataFrame(trade_records)
                 if trade_records else pd.DataFrame(columns=["pnl", "type"]))
    entry_df  = pd.DataFrame(entry_marks) if entry_marks  else pd.DataFrame()
    exit_df   = pd.DataFrame(exit_marks)  if exit_marks   else pd.DataFrame()
    flip_df   = pd.DataFrame(flip_marks)  if flip_marks   else pd.DataFrame()
    return (raw, trades_df, equity_df, levels_df, entry_df, exit_df, flip_df,
            is_synthetic, suppressed_reentries)


# ── Stats ──────────────────────────────────────────────────────────────────────

def compute_stats(equity_df, initial_capital, trades_df):
    final   = float(equity_df["equity"].iloc[-1])
    total_r = (final - initial_capital) / initial_capital * 100
    n_days  = max((equity_df.index[-1] - equity_df.index[0]).days, 1)
    ann_r   = ((final / initial_capital) ** (365 / n_days) - 1) * 100
    roll_max = equity_df["equity"].cummax()
    max_dd   = float(((equity_df["equity"] - roll_max) / roll_max * 100).min())
    daily_r  = equity_df["equity"].pct_change().dropna()
    sharpe   = daily_r.mean() / daily_r.std() * (252 ** 0.5) if daily_r.std() > 0 else 0.0
    down     = daily_r[daily_r < 0]
    sortino  = daily_r.mean() / down.std() * (252 ** 0.5) if len(down) > 1 and down.std() > 0 else 0.0
    if not trades_df.empty and "pnl" in trades_df.columns:
        wins     = int((trades_df["pnl"] > 0).sum())
        total_t  = len(trades_df)
        win_rate = wins / total_t * 100 if total_t else 0.0
        avg_win  = float(trades_df.loc[trades_df["pnl"] > 0, "pnl"].mean() or 0)
        avg_loss = float(trades_df.loc[trades_df["pnl"] <= 0, "pnl"].mean() or 0)
    else:
        total_t = wins = 0; win_rate = avg_win = avg_loss = 0.0
    return {"Total Return": f"{total_r:+.1f}%", "Ann. Return": f"{ann_r:+.1f}%",
            "Sharpe": f"{sharpe:.2f}", "Sortino": f"{sortino:.2f}",
            "Max Drawdown": f"{max_dd:.1f}%",
            "Trades": str(total_t), "Win Rate": f"{win_rate:.0f}%",
            "Avg Win": f"${avg_win:,.0f}", "Avg Loss": f"${avg_loss:,.0f}",
            "Final Equity": f"${final:,.0f}"}


# ── Buy-and-hold benchmark ─────────────────────────────────────────────────────

def compute_bah(raw, initial_capital):
    """
    Simulate buy-and-hold over the full raw price series.
    One market buy on bar 0, one market sell on the last bar.
    Same commission + market slippage applied as the strategy uses.
    Returns (equity_series, stats_dict).
    """
    entry_px  = buy_lim(float(raw["Close"].iloc[0]))
    shares    = initial_capital / entry_px
    # Mark-to-market: equity = shares × close, final bar net of exit costs
    equity    = shares * raw["Close"].copy().astype(float)
    exit_px   = sell_lim(float(raw["Close"].iloc[-1]))
    equity.iloc[-1] = shares * exit_px

    n_days    = max((raw.index[-1] - raw.index[0]).days, 1)
    final     = float(equity.iloc[-1])
    total_r   = (final - initial_capital) / initial_capital * 100
    ann_r     = ((final / initial_capital) ** (365 / n_days) - 1) * 100
    roll_max  = equity.cummax()
    max_dd    = float(((equity - roll_max) / roll_max * 100).min())
    daily_r   = equity.pct_change().dropna()
    sharpe    = daily_r.mean() / daily_r.std() * 252**0.5 if daily_r.std() > 0 else 0.0
    down      = daily_r[daily_r < 0]
    sortino   = daily_r.mean() / down.std() * 252**0.5 if len(down) > 1 and down.std() > 0 else 0.0

    stats = {
        "Total Return": f"{total_r:+.1f}%",
        "Ann. Return":  f"{ann_r:+.1f}%",
        "Sharpe":       f"{sharpe:.2f}",
        "Sortino":      f"{sortino:.2f}",
        "Max Drawdown": f"{max_dd:.1f}%",
        "Trades":       "1",
        "Win Rate":     "100%" if total_r > 0 else "0%",
        "Avg Win":      f"${final - initial_capital:,.0f}" if total_r > 0 else "—",
        "Avg Loss":     "—" if total_r > 0 else f"${final - initial_capital:,.0f}",
        "Final Equity": f"${final:,.0f}",
        "_sharpe_raw":  sharpe,
    }
    return equity, stats


# ── Plot ───────────────────────────────────────────────────────────────────────

BG, GRID, FG   = "#0d1117", "#21262d", "#e6edf3"
BLUE, RED, ORG = "#58a6ff", "#f85149", "#ffa657"
GRN,  PRP, YLW = "#3fb950", "#bc8cff", "#e3b341"


def _style(ax):
    ax.set_facecolor(BG)
    ax.tick_params(colors=FG, labelsize=8)
    for s in ax.spines.values(): s.set_edgecolor(GRID)
    ax.xaxis.label.set_color(FG); ax.yaxis.label.set_color(FG)
    ax.title.set_color(FG); ax.grid(color=GRID, alpha=0.6, lw=0.5)


def plot_results(raw, trades_df, equity_df, levels_df, entry_df, exit_df,
                 flip_df, cfg, stats, bah_equity=None, bah_stats=None,
                 is_synthetic=False, out="flip_long_results.png"):
    synth_pfx = "[SYNTHETIC DATA — RESULTS NOT MEANINGFUL]\n" if is_synthetic else ""

    fig = plt.figure(figsize=(20, 16), facecolor=BG)
    gs  = GridSpec(4, 1, figure=fig, height_ratios=[3, 1, 1, 0.7], hspace=0.44)
    ax_p, ax_eq, ax_dd, ax_st = [fig.add_subplot(gs[i]) for i in range(4)]
    for ax in (ax_p, ax_eq, ax_dd, ax_st): _style(ax)

    sl = raw.loc[levels_df.index]
    ax_p.plot(sl.index, sl["Close"], color=BLUE, lw=1.0, label="Price")
    ax_p.plot(levels_df.index, levels_df["ema"],  color=YLW, lw=0.9, ls="--",
              alpha=0.7, label=f"EMA({cfg['ema_window']}) lagged")
    ax_p.fill_between(levels_df.index, levels_df["val"], levels_df["vah"],
                      alpha=0.07, color=GRN)
    ax_p.plot(levels_df.index, levels_df["val"], color=RED, lw=0.8, ls="--", alpha=0.9, label="VAL")
    ax_p.plot(levels_df.index, levels_df["poc"], color=ORG, lw=0.8, ls="-",  alpha=0.9, label="POC")
    ax_p.plot(levels_df.index, levels_df["vah"], color=GRN, lw=0.8, ls="--", alpha=0.9, label="VAH")

    for _, r in levels_df[levels_df["flip"]].iterrows():
        ax_p.axhspan(r["val"] * 0.998, r["val"] * 1.002, alpha=0.18, color=YLW)

    if not entry_df.empty:
        ax_p.scatter(entry_df["date"], entry_df["price"], marker="^", s=80,
                     color=GRN, zorder=5, edgecolors="white", lw=0.4, label="Long entry")
    if not exit_df.empty:
        ec = {"stop": RED, "poc": ORG, "vah": GRN, "ext": PRP}
        for _, r in exit_df.iterrows():
            ax_p.scatter(r["date"], r["price"], marker="D", s=55,
                         color=ec.get(r["type"], FG), zorder=5, edgecolors="white", lw=0.4)

    ax_p.legend(handles=[
        mpatches.Patch(color=BLUE, label="Price"),
        mpatches.Patch(color=YLW,  label=f"EMA({cfg['ema_window']}) — lagged, no look-ahead"),
        mpatches.Patch(color=RED,  label="VAL"),
        mpatches.Patch(color=ORG,  label="POC"),
        mpatches.Patch(color=GRN,  label="VAH"),
        mpatches.Patch(color=YLW,  alpha=0.4, label="Flip zone (VAH→VAL)"),
        mpatches.Patch(color=GRN,  label="▲ Entry at flipped VAL"),
        mpatches.Patch(color=RED,  label="◆ Stop exit (sell_stop 0.15% slip)"),
        mpatches.Patch(color=PRP,  label="◆ Extension target"),
    ], loc="upper left", fontsize=7, facecolor=BG, labelcolor=FG, framealpha=0.8, ncol=3)

    # Fix 4: synthetic prefix in chart title
    ax_p.set_title(
        f"{synth_pfx}{cfg['ticker']}  ·  VAH→VAL Flip Long  ·  {cfg['start']} → {cfg['end']}",
        fontsize=11, fontweight="bold",
        color=RED if is_synthetic else FG)
    ax_p.set_ylabel("Price ($)")

    eq = equity_df["equity"]
    ax_eq.plot(eq.index, eq, color=GRN, lw=1.8, label="Strategy", zorder=3)
    ax_eq.axhline(cfg["initial_capital"], color=FG, lw=0.7, ls="--", alpha=0.3)
    ax_eq.fill_between(eq.index, cfg["initial_capital"], eq,
                       where=eq >= cfg["initial_capital"], color=GRN, alpha=0.12)
    ax_eq.fill_between(eq.index, cfg["initial_capital"], eq,
                       where=eq <  cfg["initial_capital"], color=RED, alpha=0.12)

    # Section 2: buy-and-hold benchmark curve on same panel
    if bah_equity is not None:
        bah_scaled = bah_equity.reindex(eq.index, method="ffill")
        bah_scaled = bah_scaled * (cfg["initial_capital"] / float(bah_equity.iloc[0]))
        ax_eq.plot(bah_scaled.index, bah_scaled, color=BLUE, lw=1.2,
                   ls="--", alpha=0.8, label="Buy-and-Hold", zorder=2)

    ax_eq.legend(fontsize=8, facecolor=BG, labelcolor=FG, framealpha=0.8)
    ax_eq.set_title(f"{synth_pfx}Equity Curve — Strategy vs Buy-and-Hold", fontsize=10,
                    color=RED if is_synthetic else FG)
    ax_eq.set_ylabel("Equity ($)")

    rm  = eq.cummax()
    ddp = (eq - rm) / rm * 100
    ax_dd.fill_between(ddp.index, ddp, 0, color=RED, alpha=0.55)
    ax_dd.plot(ddp.index, ddp, color=RED, lw=0.8)
    ax_dd.set_title("Drawdown (%)", fontsize=10); ax_dd.set_ylabel("DD %")

    ax_st.axis("off")
    # Section 2: two-row table — Strategy and Buy-and-Hold side by side
    display_keys = ["Total Return", "Ann. Return", "Sharpe", "Sortino",
                    "Max Drawdown", "Trades", "Win Rate", "Final Equity"]
    col_labels = [""] + display_keys
    strat_row  = ["Strategy"] + [stats.get(k, "—") for k in display_keys]
    bah_row    = ["Buy-and-Hold"] + (
        [bah_stats.get(k, "—") for k in display_keys] if bah_stats else ["—"] * len(display_keys))
    tbl = ax_st.table(cellText=[strat_row, bah_row], colLabels=col_labels,
                      cellLoc="center", loc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(8.5); tbl.scale(1, 2.0)
    row_colors = {0: "#161b22", 1: BG, 2: "#0a1628"}
    for (r, c), cell in tbl.get_celld().items():
        cell.set_facecolor(row_colors.get(r, BG))
        cell.set_edgecolor(GRID)
        lbl_color = GRN if r == 1 else (BLUE if r == 2 else FG)
        cell.set_text_props(color=lbl_color)

    # Fix 4: synthetic prefix in suptitle
    base_title = ("Strategy 1: VAH→VAL Flip Long  —  "
                  "Buy the level that was once resistance, now confirmed support")
    plt.suptitle(f"{synth_pfx}{base_title}",
                 fontsize=11, color=RED if is_synthetic else FG,
                 y=1.005, fontweight="bold")
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
    print(f"  Chart saved → {out}")
    plt.show()


# ── Config & entry ─────────────────────────────────────────────────────────────

DEFAULT_CONFIG = {
    "ticker":           "SPY",
    "start":            "2022-01-01",
    "end":              "2025-01-01",
    "lookback":         20,
    "vp_bins":          100,
    "initial_capital":  10_000,
    "ema_window":       50,
    "flip_tolerance":   0.03,
    "entry_buffer":     0.005,
    "risk_pct":         0.02,
    "max_position_pct": 0.40,
    "stop_pct":         0.015,
    "poc_exit_frac":    0.50,
    "vah_exit_frac":    0.70,
    "ext_mult":         1.0,
}


def main():
    p = argparse.ArgumentParser(description="VAH→VAL Flip Long Strategy")
    p.add_argument("--ticker",   default=None)
    p.add_argument("--start",    default=None)
    p.add_argument("--end",      default=None)
    p.add_argument("--capital",  type=float, default=None)
    p.add_argument("--lookback", type=int,   default=None)
    p.add_argument("--out",      default="flip_long_results.png")
    args = p.parse_args()

    cfg = DEFAULT_CONFIG.copy()
    if args.ticker:   cfg["ticker"]          = args.ticker.upper()
    if args.start:    cfg["start"]           = args.start
    if args.end:      cfg["end"]             = args.end
    if args.capital:  cfg["initial_capital"] = args.capital
    if args.lookback: cfg["lookback"]        = args.lookback

    print("\nVAH→VAL Flip Long Strategy")
    print("=" * 50)
    print(f"  Ticker       : {cfg['ticker']}")
    print(f"  Period       : {cfg['start']} → {cfg['end']}")
    print(f"  Capital      : ${cfg['initial_capital']:,.0f}")
    print(f"  Flip tol.    : {cfg['flip_tolerance']*100:.1f}%")
    print(f"  EMA filter   : {cfg['ema_window']}-bar (lagged — no look-ahead)")
    print(f"  Risk/trade   : {cfg['risk_pct']*100:.1f}%")
    print(f"  Stop         : {cfg['stop_pct']*100:.1f}% below VAL  [sell_stop {SLIP_STOP*100:.2f}% slip]")
    print(f"  Target exits : sell_lim {SLIP_LIM*100:.2f}% slip  |  Commission: {COMMISSION*100:.2f}% per side")

    (raw, trades_df, equity_df, levels_df,
     entry_df, exit_df, flip_df,
     is_synthetic, suppressed) = run_backtest(cfg)

    stats = compute_stats(equity_df, cfg["initial_capital"], trades_df)

    # Section 2: buy-and-hold benchmark (same period, same costs)
    bah_equity, bah_stats = compute_bah(raw, cfg["initial_capital"])

    # Fix 4: loud synthetic warning in printed output
    if is_synthetic:
        synth_line = "!" * 60
        print(f"\n{synth_line}")
        print("  [SYNTHETIC DATA — RESULTS NOT MEANINGFUL]")
        print("  yfinance is blocked or unavailable.  The numbers below")
        print("  come from a random-walk simulation with no real edge.")
        print("  Download real data and re-run for meaningful results:")
        print(f"    python -c \"import yfinance as yf; "
              f"yf.download('{cfg['ticker']}', start='{cfg['start']}', "
              f"end='{cfg['end']}', auto_adjust=True).to_csv('{cfg['ticker'].lower()}.csv')\"")
        print(f"{synth_line}")

    tag = "  [SYNTHETIC]  " if is_synthetic else "  "
    print("\n" + "=" * 50)
    for k, v in stats.items():
        print(f"{tag}{k:<20} {v}")
    print("=" * 50)

    if not trades_df.empty and "type" in trades_df.columns:
        print("\n  Exit breakdown:")
        print(trades_df["type"].value_counts().to_string(header=False))

    # Section 2: side-by-side benchmark comparison table
    cmp_keys = ["Total Return", "Ann. Return", "Sharpe", "Sortino", "Max Drawdown", "Final Equity"]
    w = 22
    print("\n" + "─" * (w + 16 + 16))
    print(f"  {'Metric':<{w}} {'Strategy':>14} {'Buy-and-Hold':>14}")
    print("─" * (w + 16 + 16))
    for k in cmp_keys:
        sv = stats.get(k, "—")
        bv = bah_stats.get(k, "—")
        print(f"  {k:<{w}} {sv:>14} {bv:>14}")
    print("─" * (w + 16 + 16))

    # Explicit underperformance / outperformance verdict
    strat_sharpe = float(stats["Sharpe"])
    bah_sharpe   = bah_stats.get("_sharpe_raw", 0.0)
    print()
    if strat_sharpe < bah_sharpe:
        print(f"  ⚠  UNDERPERFORMS: Strategy Sharpe ({strat_sharpe:.2f}) < "
              f"Buy-and-Hold Sharpe ({bah_sharpe:.2f}).")
        print(f"     The strategy does not demonstrate edge over passive investing "
              f"in this {'synthetic' if is_synthetic else 'period'}.")
    else:
        print(f"  ✓  OUTPERFORMS: Strategy Sharpe ({strat_sharpe:.2f}) > "
              f"Buy-and-Hold Sharpe ({bah_sharpe:.2f}).")

    if is_synthetic:
        print("\n  *** SYNTHETIC RUN — do not use these results for trading decisions ***")

    print()

    plot_results(raw, trades_df, equity_df, levels_df, entry_df, exit_df,
                 flip_df, cfg, stats,
                 bah_equity=bah_equity, bah_stats=bah_stats,
                 is_synthetic=is_synthetic, out=args.out)


if __name__ == "__main__":
    main()
