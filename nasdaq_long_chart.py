#!/usr/bin/env python3
"""
Nasdaq 100 — VAH→VAL Flip Long  (variable timeframe)

Hardcoded to Nasdaq 100 (QQQ). Pick a timeframe and see every long entry
marked on the chart with a green arrow and the fill price labelled.

Supported timeframes
  1H  2H  4H   — downloaded as 1H from Yahoo, resampled  (max ~730 days)
  1D           — daily bars  (full history)
  1W           — weekly bars (full history)

Usage
  python nasdaq_long_chart.py                   # interactive prompt
  python nasdaq_long_chart.py --tf 4H
  python nasdaq_long_chart.py --tf 1D --start 2022-01-01
  python nasdaq_long_chart.py --tf 4H --data qqq_1h.csv
"""

import argparse
import sys
import warnings
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
matplotlib.rcParams["figure.dpi"] = 110

try:
    import yfinance as yf
    _YF = True
except ImportError:
    _YF = False

TICKER = "QQQ"          # Nasdaq 100 ETF — proxy for US 100 / NQ

# ── Supported timeframes ───────────────────────────────────────────────────────
TF_OPTIONS = ["1H", "2H", "4H", "1D", "1W"]

TF_YF_INTERVAL = {            # yfinance interval to download at
    "1H": "1h", "2H": "1h", "4H": "1h",
    "1D": "1d", "1W": "1wk",
}
TF_RESAMPLE = {               # pandas resample rule (None = no resample)
    "1H": None, "2H": "2h", "4H": "4h",
    "1D": None, "1W": None,
}
TF_MAX_DAYS = {               # max history available from yfinance
    "1H": 729, "2H": 729, "4H": 729,
    "1D": 5475, "1W": 5475,
}
TF_DEFAULT_DAYS = {           # sensible default window for each tf
    "1H": 180, "2H": 270, "4H": 365,
    "1D": 730, "1W": 1460,
}
TF_VP_BARS = {                # default VP lookback in bars
    "1H": 40, "2H": 30, "4H": 20, "1D": 20, "1W": 12,
}


# ── Transaction costs ─────────────────────────────────────────────────────────
COMMISSION  = 0.0005
SLIP_LIM    = 0.0005

def buy_lim(p):  return p * (1 + SLIP_LIM + COMMISSION)
def sell_lim(p): return p * (1 - SLIP_LIM - COMMISSION)


# ── Volume profile ─────────────────────────────────────────────────────────────
def compute_vp(ohlcv: pd.DataFrame, bins: int = 60):
    lo = float(ohlcv["Low"].min())
    hi = float(ohlcv["High"].max())
    if hi <= lo + 1e-8:
        mid = (lo + hi) / 2
        return mid, mid, mid
    edges = np.linspace(lo, hi, bins + 1)
    mids  = (edges[:-1] + edges[1:]) / 2
    vol   = np.zeros(bins)
    for k in range(len(ohlcv)):
        bl = float(ohlcv["Low"].iloc[k])
        bh = float(ohlcv["High"].iloc[k])
        bv = float(ohlcv["Volume"].iloc[k])
        rng = bh - bl
        if rng < 1e-10:
            idx = min(int((bl - lo) / (hi - lo) * bins), bins - 1)
            vol[idx] += bv
        else:
            ov = np.maximum(0.0,
                np.minimum(edges[1:], bh) - np.maximum(edges[:-1], bl))
            vol += bv * ov / rng
    pi  = int(np.argmax(vol))
    poc = mids[pi]
    acc = vol[pi]; tgt = vol.sum() * 0.70
    li = hi_i = pi
    while acc < tgt:
        al = vol[li-1]   if li   > 0        else -1.0
        ah = vol[hi_i+1] if hi_i < bins-1   else -1.0
        if al < 0 and ah < 0: break
        if al >= ah: li -= 1;   acc += vol[li]
        else:        hi_i += 1; acc += vol[hi_i]
    return poc, mids[li], mids[hi_i]


# ── Data loading ───────────────────────────────────────────────────────────────
def _resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    return df.resample(rule, label="left", closed="left").agg({
        "Open":   "first",
        "High":   "max",
        "Low":    "min",
        "Close":  "last",
        "Volume": "sum",
    }).dropna(subset=["Open", "Close"])


def _synthetic(tf: str, start: str, end: str) -> pd.DataFrame:
    """Fallback synthetic OHLCV at the chosen timeframe."""
    if tf in ("1D", "1W"):
        freq = "B" if tf == "1D" else "W-FRI"
        dates = pd.date_range(start, end, freq=freq)
    else:
        hrs = int(tf[:-1])
        dates = pd.date_range(start, end, freq=f"{hrs}h")

    n   = len(dates)
    rng = np.random.default_rng(42)
    mu  = 0.00015; sigma = 0.008; S0 = 450.0
    vols = np.full(n, sigma)
    for i in range(1, n):
        vols[i] = 0.94 * vols[i-1] + 0.06 * sigma * abs(rng.standard_normal()) + 0.0005
    closes = S0 * np.exp(np.cumsum(rng.standard_normal(n) * vols + mu))
    hl = closes * vols * 2.0
    highs  = closes + hl * rng.uniform(0.3, 0.7, n)
    lows   = closes - hl * rng.uniform(0.3, 0.7, n)
    opens  = np.roll(closes, 1); opens[0] = S0
    vols_v = rng.integers(500_000, 3_000_000, n)
    df = pd.DataFrame({"Open": opens, "High": highs, "Low": lows,
                       "Close": closes, "Volume": vols_v}, index=dates)
    df.index.name = "Datetime"
    return df


def load_data(tf: str, start: str, end: str,
              csv_path: str | None = None) -> pd.DataFrame:
    """Return OHLCV DataFrame at the requested timeframe."""

    # 1) CSV override
    if csv_path and Path(csv_path).exists():
        print(f"  Loading from {csv_path} …")
        df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        if hasattr(df.index, "tz") and df.index.tz:
            df.index = df.index.tz_localize(None)
        df.columns = [c.strip().title() for c in df.columns]
        rule = TF_RESAMPLE.get(tf)
        if rule:
            df = _resample_ohlcv(df, rule)
        df.sort_index(inplace=True)
        df = df.loc[start:end]
        print(f"  {len(df)} {tf} bars  ({df.index[0].date()} → {df.index[-1].date()})")
        return df

    # 2) yfinance
    if _YF:
        interval = TF_YF_INTERVAL[tf]
        try:
            # yfinance needs period= for intraday, not start/end
            if tf in ("1H", "2H", "4H"):
                raw = yf.download(TICKER, start=start, end=end,
                                  interval=interval, auto_adjust=True, progress=False)
            else:
                raw = yf.download(TICKER, start=start, end=end,
                                  interval=interval, auto_adjust=True, progress=False)

            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.droplevel(1)
            if hasattr(raw.index, "tz") and raw.index.tz:
                raw.index = raw.index.tz_localize(None)
            raw.dropna(inplace=True)

            rule = TF_RESAMPLE[tf]
            if rule:
                raw = _resample_ohlcv(raw, rule)

            if len(raw) > 10:
                print(f"  Downloaded {len(raw)} {tf} bars  "
                      f"({raw.index[0].date()} → {raw.index[-1].date()})")
                return raw
        except Exception as e:
            print(f"  yfinance error ({e})")

    # 3) Synthetic fallback
    print(f"  yfinance unavailable — using synthetic {tf} data for illustration.")
    print("  ┌─ To get real data, run on your local machine: ──────────────────┐")
    if tf in ("1H", "2H", "4H"):
        print(f"  │  python -c \"import yfinance as yf                             │")
        print(f"  │    df = yf.download('{TICKER}', start='{start}', end='{end}',    │")
        print(f"  │           interval='1h', auto_adjust=True)                  │")
        print(f"  │    df.to_csv('qqq_1h.csv')\"                                 │")
        print(f"  │  python nasdaq_long_chart.py --tf {tf} --data qqq_1h.csv      │")
    else:
        print(f"  │  python -c \"import yfinance as yf                             │")
        print(f"  │    df = yf.download('{TICKER}', start='{start}', end='{end}',    │")
        print(f"  │           auto_adjust=True)                                  │")
        print(f"  │    df.to_csv('qqq_daily.csv')\"                               │")
        print(f"  │  python nasdaq_long_chart.py --tf {tf} --data qqq_daily.csv   │")
    print(f"  └─────────────────────────────────────────────────────────────────┘")
    return _synthetic(tf, start, end)


# ── Strategy: VAH→VAL Flip Long ───────────────────────────────────────────────
def run_strategy(data: pd.DataFrame, cfg: dict):
    lb    = cfg["lookback"]; bins = cfg.get("vp_bins", 60)
    tol   = cfg["flip_tolerance"]; buf = cfg.get("entry_buffer", 0.005)
    sp    = cfg["stop_pct"]; ema_w = cfg.get("ema_window", 50)
    cap   = float(cfg["initial_capital"])
    ema   = data["Close"].ewm(span=ema_w, adjust=False).mean()
    min_i = max(lb * 2, ema_w + 5)

    in_pos = False; sh = 0.0; avg_fill = 0.0; ecost = 0.0
    e_stop = e_poc = e_vah = e_ext = 0.0
    poc_done = vah_done = False

    equity_curve  = []
    level_records = []
    entry_marks   = []      # {"bar": i, "date": ..., "price": ..., "stop": ..., "vah": ...}
    exit_marks    = []      # {"bar": i, "date": ..., "price": ..., "type": ...}
    trades        = []

    for i in range(min_i, len(data)):
        close = float(data["Close"].iloc[i])
        high  = float(data["High"].iloc[i])
        low   = float(data["Low"].iloc[i])

        poc_c, val_c, vah_c = compute_vp(data.iloc[i-lb:i], bins)
        poc_p, val_p, vah_p = compute_vp(data.iloc[i-lb*2:i-lb], bins)

        flip    = (val_c > val_p) and (
                   (val_c >= vah_p) or (vah_p > 0 and abs(val_c - vah_p) / vah_p < tol))
        uptrend = close > float(ema.iloc[i])
        stop_lvl = val_c * (1 - sp)
        ext_lvl  = vah_c + cfg["ext_mult"] * (vah_c - poc_c)

        level_records.append({"date": data.index[i], "poc": poc_c,
                               "val": val_c, "vah": vah_c, "ext": ext_lvl,
                               "flip": flip, "uptrend": uptrend})

        # ── Exits (independent checks — no elif chain) ────────────────────────
        if in_pos:
            if low <= e_stop:
                fill   = sell_lim(e_stop)
                pnl    = (fill - avg_fill) * sh
                cap   += ecost + pnl
                trades.append({"type": "stop", "pnl": pnl})
                exit_marks.append({"date": data.index[i], "price": e_stop, "type": "stop"})
                in_pos = False; sh = ecost = 0.0; poc_done = vah_done = False
            else:
                if not poc_done and high >= e_poc:
                    cs     = sh * cfg["poc_exit_frac"]
                    fill   = sell_lim(e_poc)
                    cap   += cs * avg_fill + (fill - avg_fill) * cs
                    ecost -= cs * avg_fill; sh -= cs; poc_done = True
                    exit_marks.append({"date": data.index[i], "price": e_poc, "type": "poc"})

                if poc_done and not vah_done and high >= e_vah:
                    cs     = sh * cfg["vah_exit_frac"]
                    fill   = sell_lim(e_vah)
                    cap   += cs * avg_fill + (fill - avg_fill) * cs
                    ecost -= cs * avg_fill; sh -= cs; vah_done = True
                    exit_marks.append({"date": data.index[i], "price": e_vah, "type": "vah"})

                if vah_done and sh > 0 and high >= e_ext:
                    fill   = sell_lim(e_ext)
                    pnl    = (fill - avg_fill) * sh
                    cap   += ecost + pnl
                    trades.append({"type": "target", "pnl": pnl})
                    exit_marks.append({"date": data.index[i], "price": e_ext, "type": "ext"})
                    in_pos = False; sh = ecost = 0.0; poc_done = vah_done = False

        # ── Entry ─────────────────────────────────────────────────────────────
        if not in_pos and flip and uptrend:
            raw_px = min(close, val_c * (1 + buf))
            if low <= raw_px and close > stop_lvl:
                fill_px = buy_lim(raw_px)
                rps     = max(fill_px - sell_lim(stop_lvl), 1e-6)
                n_sh    = min(cap * cfg["risk_pct"] / rps,
                              cap * cfg["max_pos_pct"] / fill_px)
                order   = n_sh * fill_px
                if order >= 50 and cap >= order:
                    sh = n_sh; avg_fill = fill_px; ecost = order
                    e_stop = stop_lvl; e_poc = poc_c; e_vah = vah_c; e_ext = ext_lvl
                    poc_done = vah_done = False; in_pos = True; cap -= order
                    entry_marks.append({
                        "date":  data.index[i],
                        "price": fill_px,
                        "stop":  stop_lvl,
                        "poc":   poc_c,
                        "vah":   vah_c,
                        "ext":   ext_lvl,
                    })

        equity_curve.append(cap + sh * close)

    if in_pos and sh > 0:
        fill = sell_lim(float(data["Close"].iloc[-1]))
        pnl  = (fill - avg_fill) * sh
        cap += ecost + pnl
        trades.append({"type": "open", "pnl": pnl})
        equity_curve[-1] = cap

    return (pd.DataFrame(level_records).set_index("date"),
            pd.DataFrame(entry_marks) if entry_marks else pd.DataFrame(),
            pd.DataFrame(exit_marks)  if exit_marks  else pd.DataFrame(),
            np.array(equity_curve),
            trades)


# ── Statistics ─────────────────────────────────────────────────────────────────
def compute_stats(equity: np.ndarray, initial: float, trades: list) -> dict:
    eq  = pd.Series(equity, dtype=float)
    fin = float(eq.iloc[-1])
    ny  = len(eq) / 252
    cagr = (fin / initial) ** (1 / ny) - 1 if ny > 0 else 0.0
    dr   = eq.pct_change().dropna()
    mu   = float(dr.mean()); sig = float(dr.std())
    dw   = dr[dr < 0]; dsig = float(dw.std()) if len(dw) > 1 else 1e-9
    sh   = mu / sig  * 252**0.5 if sig  > 0 else 0.0
    so   = mu / dsig * 252**0.5 if dsig > 0 else 0.0
    rm   = eq.cummax()
    mdd  = float(((eq - rm) / rm).min())
    nt   = len(trades)
    wr   = sum(1 for t in trades if t["pnl"] > 0) / nt * 100 if nt else 0.0
    return {
        "CAGR":       f"{cagr*100:+.1f}%",
        "Sharpe":     f"{sh:.2f}",
        "Sortino":    f"{so:.2f}",
        "Max DD":     f"{mdd*100:.1f}%",
        "Trades":     str(nt),
        "Win Rate":   f"{wr:.0f}%",
        "Final Eq":   f"${fin:,.0f}",
    }


# ── Chart ──────────────────────────────────────────────────────────────────────
BG   = "#0d1117"
GRID = "#21262d"
FG   = "#e6edf3"
BLUE = "#58a6ff"
GRN  = "#3fb950"
RED  = "#f85149"
ORG  = "#ffa657"
PRP  = "#bc8cff"
YLW  = "#e3b341"


def _style(ax):
    ax.set_facecolor(BG)
    ax.tick_params(colors=FG, labelsize=8)
    for sp in ax.spines.values(): sp.set_edgecolor(GRID)
    ax.xaxis.label.set_color(FG); ax.yaxis.label.set_color(FG)
    ax.title.set_color(FG); ax.grid(color=GRID, alpha=0.5, lw=0.5)


def plot_chart(data: pd.DataFrame, levels: pd.DataFrame,
               entry_df: pd.DataFrame, exit_df: pd.DataFrame,
               equity: np.ndarray, trades: list,
               cfg: dict, stats: dict, tf: str,
               out: str = "nasdaq_long_entries.png"):

    from matplotlib.gridspec import GridSpec
    from matplotlib.patches import FancyArrowPatch

    sl = data.loc[levels.index]

    fig = plt.figure(figsize=(22, 14), facecolor=BG)
    gs  = GridSpec(3, 1, figure=fig,
                   height_ratios=[4, 1.2, 0.55],
                   hspace=0.38)
    ax_p  = fig.add_subplot(gs[0])
    ax_eq = fig.add_subplot(gs[1])
    ax_st = fig.add_subplot(gs[2])
    for ax in (ax_p, ax_eq, ax_st): _style(ax)

    # ── Price + VP bands ─────────────────────────────────────────────────────
    ax_p.plot(sl.index, sl["Close"], color=BLUE, lw=1.1, zorder=2, label="Price (Close)")
    ax_p.fill_between(levels.index, levels["val"], levels["vah"],
                      alpha=0.09, color=GRN, zorder=1)
    ax_p.plot(levels.index, levels["val"], color=RED, lw=0.8, ls="--", alpha=0.85, label="VAL")
    ax_p.plot(levels.index, levels["poc"], color=ORG, lw=0.8, ls="-",  alpha=0.85, label="POC")
    ax_p.plot(levels.index, levels["vah"], color=GRN, lw=0.8, ls="--", alpha=0.85, label="VAH")
    ax_p.plot(levels.index, levels["ext"], color=PRP, lw=0.6, ls=":",  alpha=0.5,  label="Extension")

    # EMA
    ema = sl["Close"].ewm(span=cfg["ema_window"], adjust=False).mean()
    ax_p.plot(sl.index, ema, color=YLW, lw=0.9, ls="--", alpha=0.65,
              label=f"EMA({cfg['ema_window']}) trend filter")

    # Highlight flip bars
    flip_idx = levels.index[levels["flip"] & levels["uptrend"]]
    for d in flip_idx:
        if d in levels.index:
            ax_p.axvspan(d, d, alpha=0.0)  # placeholder for potential shading

    # ── Long entry markers — large green arrows with price labels ─────────────
    if not entry_df.empty:
        price_range = float(sl["High"].max() - sl["Low"].min())
        arrow_offset = price_range * 0.025   # arrow sits this far below the bar low

        for _, r in entry_df.iterrows():
            date  = r["date"]
            price = r["price"]

            # Find the bar's low for arrow anchor
            bar_low = float(data.loc[date, "Low"]) if date in data.index else price
            y_tail  = bar_low - arrow_offset * 1.8
            y_head  = bar_low - arrow_offset * 0.3

            # Arrow body
            ax_p.annotate(
                "",
                xy=(date, y_head),
                xytext=(date, y_tail),
                arrowprops=dict(
                    arrowstyle="->",
                    color=GRN,
                    lw=2.2,
                    mutation_scale=18,
                ),
                zorder=8,
            )

            # Filled circle at arrowhead
            ax_p.scatter(date, y_head, s=90, color=GRN,
                         zorder=9, edgecolors="white", linewidths=0.8)

            # Price label
            ax_p.text(
                date, y_tail - arrow_offset * 0.5,
                f"${price:.1f}",
                color=GRN, fontsize=7, ha="center", va="top",
                fontweight="bold", zorder=10,
                bbox=dict(boxstyle="round,pad=0.2", facecolor=BG,
                          edgecolor=GRN, alpha=0.85, lw=0.8),
            )

    # ── Exit markers ─────────────────────────────────────────────────────────
    if not exit_df.empty:
        ec = {"stop": RED, "poc": ORG, "vah": GRN, "ext": PRP}
        em = {"stop": "x",  "poc": "D", "vah": "D", "ext": "D"}
        es = {"stop": 100,  "poc": 60,  "vah": 60,  "ext": 60}
        for _, r in exit_df.iterrows():
            ax_p.scatter(r["date"], r["price"],
                         marker=em.get(r["type"], "D"),
                         s=es.get(r["type"], 60),
                         color=ec.get(r["type"], FG),
                         zorder=7, edgecolors="white", linewidths=0.7, alpha=0.9)

    # Legend
    legend_items = [
        mpatches.Patch(color=BLUE, label="Price"),
        mpatches.Patch(color=YLW,  label=f"EMA({cfg['ema_window']}) — only long above"),
        mpatches.Patch(color=RED,  label="VAL  (long entry zone)"),
        mpatches.Patch(color=ORG,  label="POC  (50% exit ◆)"),
        mpatches.Patch(color=GRN,  label="VAH  (70% of remainder exit ◆)"),
        mpatches.Patch(color=PRP,  label="Extension target ◆"),
        mpatches.Patch(color=GRN,  label="▲ Long entry (limit at VAL)"),
        mpatches.Patch(color=RED,  label="✕ Stop loss"),
    ]
    ax_p.legend(handles=legend_items, loc="upper left", fontsize=7.5,
                facecolor=BG, labelcolor=FG, framealpha=0.9,
                ncol=2, borderpad=0.8)

    n_entries = len(entry_df) if not entry_df.empty else 0
    ax_p.set_title(
        f"Nasdaq 100 (QQQ) — VAH→VAL Flip Long  ·  {tf} candles  ·  "
        f"{n_entries} long entries marked",
        fontsize=12, fontweight="bold", pad=10,
    )
    ax_p.set_ylabel("Price ($)", fontsize=9)

    # ── Equity curve ─────────────────────────────────────────────────────────
    eq_dates = levels.index[:len(equity)]
    eq_s     = pd.Series(equity[:len(eq_dates)], index=eq_dates)
    ax_eq.plot(eq_s.index, eq_s, color=GRN, lw=1.5, label="Strategy equity")
    ax_eq.axhline(cfg["initial_capital"], color=FG, lw=0.7, ls="--", alpha=0.3)
    ax_eq.fill_between(eq_s.index, cfg["initial_capital"], eq_s,
                       where=eq_s >= cfg["initial_capital"], color=GRN, alpha=0.12)
    ax_eq.fill_between(eq_s.index, cfg["initial_capital"], eq_s,
                       where=eq_s <  cfg["initial_capital"], color=RED, alpha=0.15)
    ax_eq.set_title("Equity Curve", fontsize=9)
    ax_eq.set_ylabel("Equity ($)", fontsize=8)
    ax_eq.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))

    # ── Stats table ───────────────────────────────────────────────────────────
    ax_st.axis("off")
    tbl = ax_st.table(
        cellText=[list(stats.values())],
        colLabels=list(stats.keys()),
        cellLoc="center", loc="center",
    )
    tbl.auto_set_font_size(False); tbl.set_fontsize(10); tbl.scale(1, 2.4)
    for (row, col), cell in tbl.get_celld().items():
        cell.set_facecolor("#161b22" if row == 0 else BG)
        cell.set_edgecolor(GRID); cell.set_text_props(color=FG)

    plt.suptitle(
        f"Nasdaq 100 — VAH→VAL Flip Long  ·  {tf} candles  ·  "
        f"VP lookback {cfg['lookback']} bars  ·  "
        f"Enter limit at VAL when value area steps up + EMA({cfg['ema_window']}) trend filter",
        fontsize=10.5, color=FG, y=1.002, fontweight="bold",
    )
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
    print(f"\n  Chart saved → {out}")
    plt.show()


# ── Interactive prompt ─────────────────────────────────────────────────────────
def _ask(label, default, width=26):
    try:
        v = input(f"  {label:<{width}} [{default}]: ").strip()
        return v if v else default
    except (EOFError, KeyboardInterrupt):
        print(); return default


def interactive_config(cfg: dict) -> dict:
    print("\n  Press Enter to keep the default shown in [ ].\n")

    while True:
        tf = _ask("Timeframe (1H/2H/4H/1D/1W)", cfg["tf"]).upper()
        if tf in TF_OPTIONS: break
        print(f"  Options are: {', '.join(TF_OPTIONS)}")
    cfg["tf"] = tf

    max_d = TF_MAX_DAYS[tf]
    def_d = TF_DEFAULT_DAYS[tf]
    from datetime import date, timedelta
    default_start = (date.today() - timedelta(days=def_d)).isoformat()
    default_end   = date.today().isoformat()

    while True:
        start = _ask("Start date (YYYY-MM-DD)", default_start)
        try:
            pd.Timestamp(start); break
        except Exception:
            print(f"  Use YYYY-MM-DD format.")
    cfg["start"] = start

    while True:
        end = _ask("End   date (YYYY-MM-DD)", default_end)
        try:
            et = pd.Timestamp(end)
            if et > pd.Timestamp(start):
                # warn on intraday history limits
                days = (et - pd.Timestamp(start)).days
                if tf in ("1H", "2H", "4H") and days > max_d:
                    print(f"  Warning: {tf} data is only available for the last "
                          f"{max_d} days from Yahoo Finance.")
                break
            print("  End date must be after start date.")
        except Exception:
            print("  Use YYYY-MM-DD format.")
    cfg["end"] = end

    while True:
        lb_s = _ask("VP lookback (bars)", str(cfg["lookback"]))
        try:
            lb = int(lb_s)
            if lb >= 5: break
            print("  Minimum 5 bars.")
        except ValueError:
            print("  Enter an integer.")
    cfg["lookback"] = lb

    while True:
        cap_s = _ask("Starting capital ($)", f"{cfg['initial_capital']:,.0f}")
        try:
            cap = float(cap_s.replace(",", "").replace("$", ""))
            if cap >= 100: break
            print("  Minimum $100.")
        except ValueError:
            print("  Enter a number.")
    cfg["initial_capital"] = cap

    return cfg


# ── Main ───────────────────────────────────────────────────────────────────────
from datetime import date, timedelta

DEFAULT_TF = "4H"
DEFAULT_CFG = dict(
    tf             = DEFAULT_TF,
    start          = (date.today() - timedelta(days=TF_DEFAULT_DAYS[DEFAULT_TF])).isoformat(),
    end            = date.today().isoformat(),
    lookback       = TF_VP_BARS[DEFAULT_TF],
    initial_capital= 10_000,
    vp_bins        = 60,
    # strategy params
    ema_window     = 50,
    flip_tolerance = 0.03,
    entry_buffer   = 0.005,
    stop_pct       = 0.015,
    ext_mult       = 1.0,
    poc_exit_frac  = 0.50,
    vah_exit_frac  = 0.70,
    risk_pct       = 0.02,
    max_pos_pct    = 0.40,
)


def main():
    ap = argparse.ArgumentParser(
        description="Nasdaq 100 VAH→VAL Flip Long — variable timeframe",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python nasdaq_long_chart.py\n"
            "  python nasdaq_long_chart.py --tf 4H\n"
            "  python nasdaq_long_chart.py --tf 1D --start 2021-01-01\n"
            "  python nasdaq_long_chart.py --tf 4H --data qqq_1h.csv\n"
        ),
    )
    ap.add_argument("--tf",       default=None, choices=TF_OPTIONS,
                    help="Timeframe: 1H, 2H, 4H, 1D, 1W")
    ap.add_argument("--start",    default=None, help="Start date YYYY-MM-DD")
    ap.add_argument("--end",      default=None, help="End   date YYYY-MM-DD")
    ap.add_argument("--lookback", type=int, default=None,
                    help="VP lookback in bars")
    ap.add_argument("--capital",  type=float, default=None,
                    help="Starting capital USD")
    ap.add_argument("--data",     default=None,
                    help="Path to local OHLCV CSV (bypasses yfinance)")
    ap.add_argument("--out",      default="nasdaq_long_entries.png",
                    help="Output chart filename")
    ap.add_argument("--no-prompt", action="store_true",
                    help="Skip interactive prompt, use CLI args / defaults")
    args = ap.parse_args()

    cfg = DEFAULT_CFG.copy()
    if args.tf:       cfg["tf"]              = args.tf.upper()
    if args.start:    cfg["start"]           = args.start
    if args.end:      cfg["end"]             = args.end
    if args.lookback: cfg["lookback"]        = args.lookback
    if args.capital:  cfg["initial_capital"] = args.capital

    # Update defaults that depend on tf
    if args.tf and not args.lookback:
        cfg["lookback"] = TF_VP_BARS[cfg["tf"]]
    if args.tf and not args.start:
        cfg["start"] = (date.today() - timedelta(
            days=TF_DEFAULT_DAYS[cfg["tf"]])).isoformat()

    print("\n╔══════════════════════════════════════════════════════╗")
    print("║  Nasdaq 100 — VAH→VAL Flip Long  (variable TF)      ║")
    print("╚══════════════════════════════════════════════════════╝")

    cli_full = all([args.tf, args.start, args.end, args.capital])
    if not args.no_prompt and not cli_full:
        cfg = interactive_config(cfg)

    tf = cfg["tf"]
    print(f"\n  Ticker    : QQQ  (Nasdaq 100)")
    print(f"  Timeframe : {tf}")
    print(f"  Period    : {cfg['start']} → {cfg['end']}")
    print(f"  VP window : {cfg['lookback']} bars")
    print(f"  Capital   : ${cfg['initial_capital']:,.0f}")
    print(f"  EMA filter: {cfg['ema_window']}-bar")
    print(f"  Stop      : {cfg['stop_pct']*100:.1f}% below VAL")
    print()

    data = load_data(tf, cfg["start"], cfg["end"], csv_path=args.data)

    if len(data) < cfg["lookback"] * 2 + 10:
        sys.exit(f"Not enough bars ({len(data)}) for lookback {cfg['lookback']}×2. "
                 f"Try a wider date range or shorter lookback.")

    levels, entry_df, exit_df, equity, trades = run_strategy(data, cfg)

    stats = compute_stats(equity, cfg["initial_capital"], trades)

    n_entries = len(entry_df) if not entry_df.empty else 0
    print(f"  Long entries found : {n_entries}")
    if not entry_df.empty:
        print(f"  First entry        : {entry_df['date'].iloc[0].date()}  "
              f"@ ${entry_df['price'].iloc[0]:.2f}")
        print(f"  Last  entry        : {entry_df['date'].iloc[-1].date()}  "
              f"@ ${entry_df['price'].iloc[-1]:.2f}")

    print("\n  ── Results ──────────────────────────────────────")
    for k, v in stats.items():
        print(f"  {k:<16} {v}")
    print()

    plot_chart(data, levels, entry_df, exit_df, equity,
               trades, cfg, stats, tf, out=args.out)


if __name__ == "__main__":
    main()
