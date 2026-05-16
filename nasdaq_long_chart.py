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
import webbrowser
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

try:
    import yfinance as yf
    _YF = True
except ImportError:
    _YF = False

TICKER = "NQ=F"         # Nasdaq 100 E-mini futures (continuous front-month)

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
# ── Colour palette ────────────────────────────────────────────────────────────
BG   = "#0d1117"
BLUE = "#58a6ff"
GRN  = "#3fb950"
RED  = "#f85149"
ORG  = "#ffa657"
PRP  = "#bc8cff"
YLW  = "#e3b341"
GRID = "rgba(33,38,45,0.8)"


# ── Interactive Plotly chart ───────────────────────────────────────────────────
def plot_chart(data: pd.DataFrame, levels: pd.DataFrame,
               entry_df: pd.DataFrame, exit_df: pd.DataFrame,
               equity: np.ndarray, trades: list,
               cfg: dict, stats: dict, tf: str,
               out: str = "nasdaq_long_entries.html"):
    """
    Renders a fully interactive Plotly chart saved as HTML.
    Scroll to zoom, drag to pan, double-click to reset, click legend to toggle.
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    sl = data.loc[levels.index]
    ema = sl["Close"].ewm(span=cfg["ema_window"], adjust=False).mean()

    # Equity series aligned to levels index
    eq_dates = levels.index[:len(equity)]
    eq_s     = pd.Series(equity[:len(eq_dates)], index=eq_dates, dtype=float)

    n_entries = len(entry_df) if not entry_df.empty else 0
    title = (f"Nasdaq 100 Futures (NQ) — VAH→VAL Flip Long  ·  {tf} candles  ·  "
             f"VP {cfg['lookback']}-bar window  ·  {n_entries} long entries")

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.72, 0.28],
        vertical_spacing=0.04,
        subplot_titles=("", "Equity Curve ($)"),
    )

    # ── Candlesticks ──────────────────────────────────────────────────────────
    fig.add_trace(go.Candlestick(
        x=sl.index, open=sl["Open"], high=sl["High"],
        low=sl["Low"], close=sl["Close"],
        increasing_line_color=GRN, decreasing_line_color=RED,
        increasing_fillcolor=GRN, decreasing_fillcolor=RED,
        line_width=1, name="NQ Futures",
        hovertext=[
            f"O: {o:.2f}  H: {h:.2f}  L: {l:.2f}  C: {c:.2f}"
            for o, h, l, c in zip(sl["Open"], sl["High"], sl["Low"], sl["Close"])
        ],
        hoverinfo="x+text",
    ), row=1, col=1)

    # ── Volume profile levels ─────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=levels.index, y=levels["val"],
        line=dict(color=RED, width=1.2, dash="dash"),
        name="VAL", opacity=0.9,
        hovertemplate="VAL: $%{y:.2f}<extra></extra>",
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=levels.index, y=levels["poc"],
        line=dict(color=ORG, width=1.2),
        name="POC", opacity=0.9,
        hovertemplate="POC: $%{y:.2f}<extra></extra>",
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=levels.index, y=levels["vah"],
        line=dict(color=GRN, width=1.2, dash="dash"),
        name="VAH", opacity=0.9,
        hovertemplate="VAH: $%{y:.2f}<extra></extra>",
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=levels.index, y=levels["ext"],
        line=dict(color=PRP, width=0.8, dash="dot"),
        name="Extension target", opacity=0.6,
        hovertemplate="Ext: $%{y:.2f}<extra></extra>",
    ), row=1, col=1)

    # Value area shading — fill between VAL and VAH
    fig.add_trace(go.Scatter(
        x=list(levels.index) + list(levels.index[::-1]),
        y=list(levels["vah"]) + list(levels["val"][::-1]),
        fill="toself",
        fillcolor="rgba(63,185,80,0.07)",
        line=dict(width=0),
        name="Value Area",
        showlegend=True,
        hoverinfo="skip",
    ), row=1, col=1)

    # ── EMA trend filter ──────────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=sl.index, y=ema,
        line=dict(color=YLW, width=1.0, dash="dash"),
        name=f"EMA({cfg['ema_window']}) trend filter",
        opacity=0.7,
        hovertemplate="EMA: $%{y:.2f}<extra></extra>",
    ), row=1, col=1)

    # ── Long entry markers ────────────────────────────────────────────────────
    if not entry_df.empty:
        # Pin arrow tip to bar low, offset slightly below
        bar_lows = []
        for d in entry_df["date"]:
            bar_lows.append(float(data.loc[d, "Low"]) if d in data.index
                            else float(entry_df.loc[entry_df["date"] == d, "price"].iloc[0]))
        price_range = float(sl["High"].max() - sl["Low"].min())
        offset      = price_range * 0.018

        fig.add_trace(go.Scatter(
            x=entry_df["date"],
            y=[lo - offset for lo in bar_lows],
            mode="markers+text",
            marker=dict(
                symbol="triangle-up",
                size=14,
                color=GRN,
                line=dict(color="white", width=1),
            ),
            text=[f"${p:.1f}" for p in entry_df["price"]],
            textposition="bottom center",
            textfont=dict(color=GRN, size=9, family="monospace"),
            name=f"Long entry ({n_entries})",
            customdata=list(zip(
                entry_df["price"],
                entry_df.get("stop", [0]*n_entries),
                entry_df.get("vah",  [0]*n_entries),
            )),
            hovertemplate=(
                "<b>LONG ENTRY</b><br>"
                "Fill:  $%{customdata[0]:.2f}<br>"
                "Stop:  $%{customdata[1]:.2f}<br>"
                "Target VAH: $%{customdata[2]:.2f}<br>"
                "<extra></extra>"
            ),
        ), row=1, col=1)

    # ── Exit markers ─────────────────────────────────────────────────────────
    if not exit_df.empty:
        exit_styles = {
            "stop": dict(symbol="x",            color=RED, size=11, name="Stop loss"),
            "poc":  dict(symbol="diamond",       color=ORG, size=9,  name="POC exit (50%)"),
            "vah":  dict(symbol="diamond",       color=GRN, size=9,  name="VAH exit (70% rem.)"),
            "ext":  dict(symbol="diamond",       color=PRP, size=9,  name="Extension exit"),
        }
        for etype, style in exit_styles.items():
            sub = exit_df[exit_df["type"] == etype]
            if sub.empty: continue
            fig.add_trace(go.Scatter(
                x=sub["date"], y=sub["price"],
                mode="markers",
                marker=dict(
                    symbol=style["symbol"],
                    size=style["size"],
                    color=style["color"],
                    line=dict(color="white", width=0.8),
                ),
                name=style["name"],
                hovertemplate=f"<b>{style['name']}</b><br>$%{{y:.2f}}<extra></extra>",
            ), row=1, col=1)

    # ── Equity curve ─────────────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=eq_s.index, y=eq_s,
        line=dict(color=GRN, width=1.8),
        fill="tozeroy",
        fillcolor="rgba(63,185,80,0.10)",
        name="Equity",
        hovertemplate="$%{y:,.0f}<extra>Equity</extra>",
    ), row=2, col=1)

    fig.add_hline(
        y=cfg["initial_capital"], row=2, col=1,
        line=dict(color="rgba(230,237,243,0.3)", width=1, dash="dash"),
    )

    # ── Stats annotation box ──────────────────────────────────────────────────
    stats_text = "  ".join(f"<b>{k}</b> {v}" for k, v in stats.items())
    fig.add_annotation(
        xref="paper", yref="paper",
        x=0.01, y=-0.04,
        text=stats_text,
        showarrow=False,
        font=dict(size=11, color="#e6edf3", family="monospace"),
        bgcolor="#161b22",
        bordercolor="#21262d",
        borderwidth=1,
        borderpad=6,
        align="left",
    )

    # ── Layout ────────────────────────────────────────────────────────────────
    fig.update_layout(
        title=dict(text=title, font=dict(size=14, color="#e6edf3"), x=0.01),
        paper_bgcolor=BG,
        plot_bgcolor=BG,
        font=dict(color="#e6edf3", family="sans-serif"),
        legend=dict(
            bgcolor="rgba(22,27,34,0.9)",
            bordercolor="#21262d",
            borderwidth=1,
            font=dict(size=11),
            x=0.01, y=0.99,
            xanchor="left", yanchor="top",
            itemclick="toggle",
            itemdoubleclick="toggleothers",
        ),
        hovermode="x unified",
        xaxis_rangeslider_visible=False,
        margin=dict(l=60, r=30, t=60, b=100),
    )

    # Style both subplots
    for row in (1, 2):
        fig.update_xaxes(
            showgrid=True, gridcolor=GRID, gridwidth=1,
            zeroline=False,
            showspikes=True, spikecolor="#58a6ff",
            spikedash="dot", spikethickness=1,
            row=row, col=1,
        )
        fig.update_yaxes(
            showgrid=True, gridcolor=GRID, gridwidth=1,
            zeroline=False,
            tickprefix="$",
            row=row, col=1,
        )

    # Price panel: zoom tools
    fig.update_xaxes(
        rangeselector=dict(
            buttons=[
                dict(count=5,  label="5d",  step="day",   stepmode="backward"),
                dict(count=1,  label="1m",  step="month", stepmode="backward"),
                dict(count=3,  label="3m",  step="month", stepmode="backward"),
                dict(count=6,  label="6m",  step="month", stepmode="backward"),
                dict(count=1,  label="1y",  step="year",  stepmode="backward"),
                dict(step="all", label="All"),
            ],
            bgcolor="#161b22", activecolor="#21262d",
            font=dict(color="#e6edf3"),
        ),
        row=1, col=1,
    )

    # Save
    html_out = out if out.endswith(".html") else Path(out).with_suffix(".html").as_posix()
    fig.write_html(
        html_out,
        include_plotlyjs="cdn",
        config=dict(
            scrollZoom=True,
            displayModeBar=True,
            modeBarButtonsToAdd=["drawline", "eraseshape"],
            toImageButtonOptions=dict(
                format="png", filename="nasdaq_long_chart", scale=2,
            ),
        ),
    )
    print(f"\n  Interactive chart → {html_out}")
    print("  Open in your browser — scroll to zoom, drag to pan, "
          "click legend to toggle traces.")
    try:
        webbrowser.open(f"file://{Path(html_out).resolve()}")
    except Exception:
        pass


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
    ap.add_argument("--out",      default="nasdaq_long_entries.html",
                    help="Output chart filename (.html = interactive, .png = static)")
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
    print("║  Nasdaq 100 Futures (NQ) — VAH→VAL Flip Long        ║")
    print("╚══════════════════════════════════════════════════════╝")

    cli_full = all([args.tf, args.start, args.end, args.capital])
    if not args.no_prompt and not cli_full:
        cfg = interactive_config(cfg)

    tf = cfg["tf"]
    print(f"\n  Ticker    : NQ=F  (Nasdaq 100 E-mini Futures)")
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

    out = args.out
    if not out.endswith((".html", ".png")):
        out += ".html"
    plot_chart(data, levels, entry_df, exit_df, equity,
               trades, cfg, stats, tf, out=out)


if __name__ == "__main__":
    main()
