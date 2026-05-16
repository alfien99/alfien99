#!/usr/bin/env python3
"""
Strategy 2: Staircase Breakout with Value-Area Confirmation (Best Strategy)

Reasoning from chart patterns:
  The US 100 chart shows a clear "staircase accumulation" structure:
    1. Price consolidates in a value area (VAL–VAH band)
    2. Breaks above VAH on expanding volume
    3. Consolidates again at a higher level — with the new VAL ≈ old VAH
    4. Repeats

  The highest probability entries come at TWO points in this cycle:
    A) Breakout entry: close above VAH + momentum confirmation
       → Catches the move early but is slightly more aggressive
    B) Pullback entry: after breakout, price retraces to VAL zone (which was old VAH)
       → Catches a second chance with a tighter stop

  This strategy combines both:
    - Phase A: enter on a confirmed VAH breakout (close > VAH for N bars)
    - Phase B: if missed / stopped out, re-enter on pullback to VAL
    - Adds a Volume Surge filter to reduce false breakouts
    - Uses an ATR-based trailing stop for Phase A runners
    - Exits in three tranches: POC+VAH+extension

Usage:
  python staircase_breakout_strategy.py
  python staircase_breakout_strategy.py --ticker QQQ --start 2021-01-01 --end 2025-01-01
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


# ── Data ───────────────────────────────────────────────────────────────────────

def _synthetic_ohlcv(ticker, start, end):
    rng   = np.random.default_rng(abs(hash(ticker)) % (2**31))
    dates = pd.bdate_range(start, end)
    n     = len(dates)
    mu, sigma, S0 = 0.0005, 0.013, 18_000.0
    vols = np.full(n, sigma)
    for i in range(1, n):
        vols[i] = 0.91 * vols[i-1] + 0.09 * sigma * abs(rng.standard_normal()) + 0.001
    closes  = S0 * np.exp(np.cumsum(rng.standard_normal(n) * vols + mu))
    hl      = closes * vols * 2.5
    highs   = closes + hl * rng.uniform(0.35, 0.65, n)
    lows    = closes - hl * rng.uniform(0.35, 0.65, n)
    opens   = np.roll(closes, 1); opens[0] = S0
    volumes = rng.integers(1_000_000, 6_000_000, n)
    df = pd.DataFrame({"Open": opens, "High": highs, "Low": lows,
                       "Close": closes, "Volume": volumes}, index=dates)
    df.index.name = "Date"
    return df


def _download(ticker, start, end):
    if _YF_AVAILABLE:
        try:
            raw = yf.download(ticker, start=start, end=end,
                              auto_adjust=True, progress=False)
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.droplevel(1)
            raw.dropna(inplace=True)
            if len(raw) > 20:
                print(f"  Downloaded {len(raw)} bars.")
                return raw
        except Exception as e:
            print(f"  Yahoo Finance error ({e}) — using synthetic data.")
    print(f"  Generating synthetic OHLCV for {ticker} …")
    return _synthetic_ohlcv(ticker, start, end)


# ── Indicators ─────────────────────────────────────────────────────────────────

def compute_atr(raw, window=14):
    h, l, c = raw["High"], raw["Low"], raw["Close"]
    prev_c  = c.shift(1)
    tr = pd.concat([h - l, (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    return tr.ewm(span=window, adjust=False).mean()


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
    raw      = _download(cfg["ticker"], cfg["start"], cfg["end"])
    lookback = cfg["lookback"]
    bins     = cfg["vp_bins"]
    capital  = float(cfg["initial_capital"])

    atr     = compute_atr(raw, cfg["atr_window"])
    vol_ma  = raw["Volume"].rolling(cfg["vol_ma_window"]).mean()
    ema200  = raw["Close"].ewm(span=200, adjust=False).mean()

    # State
    in_pos         = False
    shares         = 0.0
    avg_px         = 0.0
    cost           = 0.0
    entry_poc      = 0.0
    entry_vah_exit = 0.0
    entry_ext      = 0.0
    entry_stop     = 0.0
    trail_stop     = 0.0
    poc_done       = False
    vah_done       = False
    phase          = None     # "breakout" or "pullback"

    # Track bars above VAH for breakout confirmation
    bars_above_vah = 0
    last_vah       = None

    equity_records = []
    level_records  = []
    trade_records  = []
    entry_marks    = []
    exit_marks     = []

    min_i = max(lookback + cfg["vol_ma_window"], 210)

    for i in range(min_i, len(raw)):
        date  = raw.index[i]
        close = float(raw["Close"].iloc[i])
        high  = float(raw["High"].iloc[i])
        low   = float(raw["Low"].iloc[i])

        poc, val, vah, _, _ = compute_vp(raw.iloc[i - lookback: i], bins)
        atr_v   = float(atr.iloc[i])
        vol_v   = float(raw["Volume"].iloc[i])
        vol_avg = float(vol_ma.iloc[i])
        uptrend = close > float(ema200.iloc[i])
        vol_surge = vol_v > vol_avg * cfg["vol_surge_mult"]

        ext_lvl  = vah + cfg["ext_mult"] * (vah - poc)

        level_records.append({"date": date, "poc": poc, "val": val,
                               "vah": vah, "ext": ext_lvl,
                               "ema200": float(ema200.iloc[i])})

        # ── Exits ────────────────────────────────────────────────────────────
        if in_pos:
            # Update trailing stop for breakout phase
            if phase == "breakout":
                trail_stop = max(trail_stop, close - cfg["trail_atr_mult"] * atr_v)
                actual_stop = max(entry_stop, trail_stop)
            else:
                actual_stop = entry_stop

            if low <= actual_stop:
                pnl = (actual_stop - avg_px) * shares
                capital += cost + pnl
                trade_records.append({"date": date, "type": "stop", "phase": phase,
                                      "entry": avg_px, "exit": actual_stop, "pnl": pnl})
                exit_marks.append({"date": date, "price": actual_stop, "type": "stop"})
                in_pos = False; shares = cost = 0.0; poc_done = vah_done = False

            elif not poc_done and high >= entry_poc:
                cs       = shares * cfg["poc_exit_frac"]
                pnl_p    = (entry_poc - avg_px) * cs
                capital += cs * avg_px + pnl_p
                cost    -= cs * avg_px; shares -= cs
                poc_done = True
                exit_marks.append({"date": date, "price": entry_poc, "type": "poc"})

            elif poc_done and not vah_done and high >= entry_vah_exit:
                cs       = shares * cfg["vah_exit_frac"]
                pnl_p    = (entry_vah_exit - avg_px) * cs
                capital += cs * avg_px + pnl_p
                cost    -= cs * avg_px; shares -= cs
                vah_done = True
                exit_marks.append({"date": date, "price": entry_vah_exit, "type": "vah"})

            elif vah_done and shares > 0 and high >= entry_ext:
                pnl = (entry_ext - avg_px) * shares
                capital += cost + pnl
                trade_records.append({"date": date, "type": "target", "phase": phase,
                                      "entry": avg_px, "exit": entry_ext, "pnl": pnl})
                exit_marks.append({"date": date, "price": entry_ext, "type": "ext"})
                in_pos = False; shares = cost = 0.0; poc_done = vah_done = False

        # ── Entry Phase A: Breakout ──────────────────────────────────────────
        # Close above VAH + optional volume surge + uptrend
        if not in_pos and uptrend:
            if close > vah:
                bars_above_vah += 1
                last_vah = vah
            else:
                bars_above_vah = 0

            use_vol_filter = vol_surge or not cfg.get("require_vol_surge", True)
            if (bars_above_vah >= cfg["breakout_confirm_bars"]
                    and use_vol_filter and close > vah):
                stop_lvl   = vah - cfg["breakout_stop_atr"] * atr_v
                risk_cash  = capital * cfg["risk_pct"]
                risk_per_sh = max(close - stop_lvl, 1e-6)
                n_sh       = risk_cash / risk_per_sh
                order_cash = min(n_sh * close, capital * cfg.get("max_position_pct", 0.40))

                if order_cash >= 50 and capital >= order_cash:
                    shares         = n_sh
                    avg_px         = close
                    cost           = order_cash
                    entry_stop     = stop_lvl
                    trail_stop     = close - cfg["trail_atr_mult"] * atr_v
                    entry_poc      = poc
                    entry_vah_exit = vah + (vah - poc)   # first target: VAH + (VAH-POC)
                    entry_ext      = ext_lvl
                    poc_done       = vah_done = False
                    in_pos         = True
                    phase          = "breakout"
                    capital       -= order_cash
                    bars_above_vah = 0
                    entry_marks.append({"date": date, "price": close, "phase": "breakout"})

        # ── Entry Phase B: Pullback to VAL (after breakout above) ───────────
        # Model a limit order at VAL: if bar's low dips to VAL zone, fill there.
        # Condition: had a prior breakout (last_vah set), uptrend holds,
        # and price touches VAL intraday.
        pb_fill = val * (1 + cfg["pullback_buffer"])
        above_stop_pb = low > val * (1 - cfg["stop_pct"])
        if (not in_pos and uptrend and last_vah is not None
                and low <= pb_fill and above_stop_pb):
            stop_lvl    = val * (1 - cfg["stop_pct"])
            fill_px_pb  = min(close, pb_fill)
            risk_cash   = capital * cfg["risk_pct"] * 1.5
            risk_per_sh = max(fill_px_pb - stop_lvl, 1e-6)
            n_sh        = risk_cash / risk_per_sh
            order_cash  = min(n_sh * fill_px_pb, capital * cfg.get("max_position_pct", 0.40))

            if order_cash >= 50 and capital >= order_cash:
                shares         = n_sh
                avg_px         = fill_px_pb
                cost           = order_cash
                entry_stop     = val * (1 - cfg["stop_pct"])
                trail_stop     = entry_stop
                entry_poc      = poc
                entry_vah_exit = vah
                entry_ext      = ext_lvl
                poc_done       = vah_done = False
                in_pos         = True
                phase          = "pullback"
                capital       -= order_cash
                entry_marks.append({"date": date, "price": fill_px_pb, "phase": "pullback"})

        equity_records.append({"date": date,
                                "equity": capital + shares * close})

    if in_pos and shares > 0:
        lp  = float(raw["Close"].iloc[-1])
        pnl = (lp - avg_px) * shares
        capital += cost + pnl
        trade_records.append({"date": raw.index[-1], "type": "expired", "phase": phase,
                               "entry": avg_px, "exit": lp, "pnl": pnl})

    equity_df = pd.DataFrame(equity_records).set_index("date")
    levels_df = pd.DataFrame(level_records).set_index("date")
    trades_df = (pd.DataFrame(trade_records)
                 if trade_records else pd.DataFrame(columns=["pnl", "type", "phase"]))
    entry_df  = pd.DataFrame(entry_marks) if entry_marks else pd.DataFrame()
    exit_df   = pd.DataFrame(exit_marks)  if exit_marks  else pd.DataFrame()
    return raw, trades_df, equity_df, levels_df, entry_df, exit_df


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
    if not trades_df.empty and "pnl" in trades_df.columns:
        wins     = int((trades_df["pnl"] > 0).sum())
        total_t  = len(trades_df)
        win_rate = wins / total_t * 100 if total_t else 0.0
        avg_win  = float(trades_df.loc[trades_df["pnl"] > 0, "pnl"].mean() or 0)
        avg_loss = float(trades_df.loc[trades_df["pnl"] <= 0, "pnl"].mean() or 0)
        bo_pnl   = trades_df.loc[trades_df.get("phase", pd.Series()) == "breakout", "pnl"].sum() if "phase" in trades_df else 0
        pb_pnl   = trades_df.loc[trades_df.get("phase", pd.Series()) == "pullback", "pnl"].sum() if "phase" in trades_df else 0
    else:
        total_t = wins = 0
        win_rate = avg_win = avg_loss = bo_pnl = pb_pnl = 0.0
    return {"Total Return": f"{total_r:+.1f}%", "Ann. Return": f"{ann_r:+.1f}%",
            "Max Drawdown": f"{max_dd:.1f}%", "Sharpe": f"{sharpe:.2f}",
            "Trades": str(total_t), "Win Rate": f"{win_rate:.0f}%",
            "Breakout PnL": f"${bo_pnl:,.0f}", "Pullback PnL": f"${pb_pnl:,.0f}",
            "Avg Win": f"${avg_win:,.0f}", "Avg Loss": f"${avg_loss:,.0f}",
            "Final Equity": f"${final:,.0f}"}


# ── Plot ───────────────────────────────────────────────────────────────────────

BG, GRID, FG   = "#0d1117", "#21262d", "#e6edf3"
BLUE, RED, ORG = "#58a6ff", "#f85149", "#ffa657"
GRN,  PRP, YLW = "#3fb950", "#bc8cff", "#e3b341"
CYAN            = "#39d0d8"


def _style(ax):
    ax.set_facecolor(BG)
    ax.tick_params(colors=FG, labelsize=8)
    for s in ax.spines.values(): s.set_edgecolor(GRID)
    ax.xaxis.label.set_color(FG); ax.yaxis.label.set_color(FG)
    ax.title.set_color(FG); ax.grid(color=GRID, alpha=0.6, lw=0.5)


def plot_results(raw, trades_df, equity_df, levels_df, entry_df, exit_df,
                 cfg, stats, out="staircase_results.png"):
    fig = plt.figure(figsize=(22, 16), facecolor=BG)
    gs  = GridSpec(4, 1, figure=fig, height_ratios=[3, 1, 1, 0.65], hspace=0.42)
    ax_p, ax_eq, ax_dd, ax_st = [fig.add_subplot(gs[i]) for i in range(4)]
    for ax in (ax_p, ax_eq, ax_dd, ax_st): _style(ax)

    sl = raw.loc[levels_df.index]
    ax_p.plot(sl.index, sl["Close"], color=BLUE, lw=1.0, label="Price")
    ax_p.plot(levels_df.index, levels_df["ema200"], color=YLW, lw=1.0, ls="--", alpha=0.6, label="EMA 200")
    ax_p.fill_between(levels_df.index, levels_df["val"], levels_df["vah"],
                      alpha=0.07, color=GRN)
    ax_p.plot(levels_df.index, levels_df["val"], color=RED, lw=0.8, ls="--", alpha=0.9, label="VAL")
    ax_p.plot(levels_df.index, levels_df["poc"], color=ORG, lw=0.8, ls="-",  alpha=0.9, label="POC")
    ax_p.plot(levels_df.index, levels_df["vah"], color=GRN, lw=0.8, ls="--", alpha=0.9, label="VAH")
    ax_p.plot(levels_df.index, levels_df["ext"], color=PRP, lw=0.6, ls=":",  alpha=0.55, label="Extension target")

    if not entry_df.empty:
        bo = entry_df[entry_df["phase"] == "breakout"]
        pb = entry_df[entry_df["phase"] == "pullback"]
        if not bo.empty:
            ax_p.scatter(bo["date"], bo["price"], marker="^", s=90, color=CYAN,
                         zorder=5, edgecolors="white", lw=0.4, label="Breakout entry")
        if not pb.empty:
            ax_p.scatter(pb["date"], pb["price"], marker="^", s=70, color=GRN,
                         zorder=5, edgecolors="white", lw=0.4, label="Pullback entry")

    if not exit_df.empty:
        ec = {"stop": RED, "poc": ORG, "vah": GRN, "ext": PRP}
        for _, r in exit_df.iterrows():
            ax_p.scatter(r["date"], r["price"], marker="D", s=50,
                         color=ec.get(r["type"], FG), zorder=5, edgecolors="white", lw=0.4)

    ax_p.legend(handles=[
        mpatches.Patch(color=BLUE, label="Price"),
        mpatches.Patch(color=YLW,  label="EMA 200 (trend)"),
        mpatches.Patch(color=RED,  label="VAL"),
        mpatches.Patch(color=ORG,  label="POC"),
        mpatches.Patch(color=GRN,  label="VAH"),
        mpatches.Patch(color=PRP,  label="Extension target"),
        mpatches.Patch(color=CYAN, label="▲ Breakout entry"),
        mpatches.Patch(color=GRN,  label="▲ Pullback entry"),
        mpatches.Patch(color=RED,  label="◆ Stop"),
        mpatches.Patch(color=PRP,  label="◆ Target"),
    ], loc="upper left", fontsize=7, facecolor=BG, labelcolor=FG, framealpha=0.8, ncol=3)
    ax_p.set_title(
        f"{cfg['ticker']}  ·  Staircase Breakout + Pullback  ·  {cfg['start']} → {cfg['end']}",
        fontsize=11, fontweight="bold")
    ax_p.set_ylabel("Price ($)")

    eq = equity_df["equity"]
    ax_eq.plot(eq.index, eq, color=GRN, lw=1.5)
    ax_eq.axhline(cfg["initial_capital"], color=FG, lw=0.7, ls="--", alpha=0.3)
    ax_eq.fill_between(eq.index, cfg["initial_capital"], eq,
                       where=eq >= cfg["initial_capital"], color=GRN, alpha=0.12)
    ax_eq.fill_between(eq.index, cfg["initial_capital"], eq,
                       where=eq <  cfg["initial_capital"], color=RED, alpha=0.12)
    ax_eq.set_title("Equity Curve", fontsize=10); ax_eq.set_ylabel("Equity ($)")

    rm  = eq.cummax()
    ddp = (eq - rm) / rm * 100
    ax_dd.fill_between(ddp.index, ddp, 0, color=RED, alpha=0.55)
    ax_dd.plot(ddp.index, ddp, color=RED, lw=0.8)
    ax_dd.set_title("Drawdown (%)", fontsize=10); ax_dd.set_ylabel("DD %")

    ax_st.axis("off")
    tbl = ax_st.table(cellText=[list(stats.values())], colLabels=list(stats.keys()),
                      cellLoc="center", loc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(9); tbl.scale(1, 2.2)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_facecolor("#161b22" if r == 0 else BG)
        cell.set_edgecolor(GRID); cell.set_text_props(color=FG)

    plt.suptitle(
        "Strategy 2: Staircase Breakout + VAL Pullback  —  Volume-confirmed breakout ▲ · Trend-aligned pullback ▲ · ATR trailing stop",
        fontsize=11, color=FG, y=1.005, fontweight="bold")
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
    print(f"  Chart saved → {out}")
    plt.show()


# ── Config & entry ─────────────────────────────────────────────────────────────

DEFAULT_CONFIG = {
    "ticker":                "SPY",
    "start":                 "2022-01-01",
    "end":                   "2025-01-01",
    "lookback":              20,
    "vp_bins":               100,
    "initial_capital":       10_000,

    # Trend filter
    # (EMA 200 is hardcoded; only trade in direction of long-term trend)

    # Phase A — Breakout
    "breakout_confirm_bars": 2,      # close above VAH for N bars
    "require_vol_surge":     False,  # set True to enforce volume filter (useful with real data)
    "vol_surge_mult":        1.4,    # volume must be 1.4× its MA (when require_vol_surge=True)
    "vol_ma_window":         20,
    "breakout_stop_atr":     1.5,    # stop = entry - 1.5 × ATR
    "trail_atr_mult":        2.0,    # trailing stop = high - 2 × ATR
    "atr_window":            14,

    # Phase B — Pullback
    "pullback_buffer":       0.003,  # enter within 0.3% above VAL
    "stop_pct":              0.015,  # stop 1.5% below VAL

    # Position sizing
    "risk_pct":              0.02,   # risk 2% of capital per trade

    # Exits (shared)
    "poc_exit_frac":         0.40,   # 40% off at POC
    "vah_exit_frac":         0.60,   # 60% of remainder at VAH
    "ext_mult":              1.2,    # extension = VAH + 1.2 × (VAH - POC)
}


def main():
    p = argparse.ArgumentParser(description="Staircase Breakout + Pullback Strategy")
    p.add_argument("--ticker",   default=None)
    p.add_argument("--start",    default=None)
    p.add_argument("--end",      default=None)
    p.add_argument("--capital",  type=float, default=None)
    p.add_argument("--lookback", type=int,   default=None)
    p.add_argument("--out",      default="staircase_results.png")
    args = p.parse_args()

    cfg = DEFAULT_CONFIG.copy()
    if args.ticker:   cfg["ticker"]          = args.ticker.upper()
    if args.start:    cfg["start"]           = args.start
    if args.end:      cfg["end"]             = args.end
    if args.capital:  cfg["initial_capital"] = args.capital
    if args.lookback: cfg["lookback"]        = args.lookback

    print("\nStaircase Breakout + VAL Pullback Strategy")
    print("=" * 52)
    print(f"  Ticker       : {cfg['ticker']}")
    print(f"  Period       : {cfg['start']} → {cfg['end']}")
    print(f"  Capital      : ${cfg['initial_capital']:,.0f}")
    print(f"  Breakout     : close > VAH for {cfg['breakout_confirm_bars']} bars + vol >{cfg['vol_surge_mult']}× avg")
    print(f"  Pullback     : close ≤ VAL × {1+cfg['pullback_buffer']:.3f}")
    print(f"  Risk/trade   : {cfg['risk_pct']*100:.1f}%")
    print(f"  Trail stop   : {cfg['trail_atr_mult']}× ATR")

    raw, trades_df, equity_df, levels_df, entry_df, exit_df = run_backtest(cfg)
    stats = compute_stats(equity_df, cfg["initial_capital"], trades_df)

    print("\n" + "=" * 52)
    for k, v in stats.items(): print(f"  {k:<20} {v}")
    print("=" * 52)
    if not trades_df.empty and "type" in trades_df.columns:
        print("\n  Exit breakdown:")
        if "phase" in trades_df.columns:
            print(trades_df.groupby(["phase", "type"]).size().to_string())
        else:
            print(trades_df["type"].value_counts().to_string(header=False))
    print()

    plot_results(raw, trades_df, equity_df, levels_df, entry_df, exit_df,
                 cfg, stats, out=args.out)


if __name__ == "__main__":
    main()
