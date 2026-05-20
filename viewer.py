#!/usr/bin/env python3
"""
VAH / VAL Flip — Interactive Trade Viewer
==========================================
A browser-based inspection tool for verifying that fills land at the
correct VAH / VAL / POC levels and for zooming into individual trades.

Features
--------
• Candlestick chart with zoom, pan, and crosshair
• Rolling VAH / POC / VAL lines + EMA overlay
• Long (▲) and Short (▼) entry markers — hover to see exact fill price,
  stop-loss level, and all three profit targets with their % distances
• Per-trade SL and TP lines drawn for the full life of each position
• Live Volume Profile in a side panel that UPDATES as you zoom the chart
  — the VP always reflects the currently visible price range
• Click any trade marker for a detailed breakdown card below the chart

Install
-------
  pip install dash plotly

Run
---
  python viewer.py              ← interactive prompt (ticker / timeframe / days)
  python viewer.py --quick      ← skip prompt, run with defaults from CFG
"""

import sys
import threading
import webbrowser

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, callback_context, dcc, html
from plotly.subplots import make_subplots

# ── Import the strategy ────────────────────────────────────────────────────────
try:
    import vah_val_flip_long as S
except ModuleNotFoundError:
    print("ERROR: vah_val_flip_long.py must be in the same directory as viewer.py")
    sys.exit(1)

PORT = 8050

# ── Global cache — stores the most recent run ──────────────────────────────────
_G: dict = {}

# ── Colour constants (plotly uses hex strings or CSS colours) ──────────────────
BG      = "#0d1117"
PANEL   = "#161b22"
BORDER  = "#30363d"
TEXT    = "#e6edf3"
MUTED   = "#8b949e"
GREEN   = "#3fb950"
RED     = "#f85149"
ORANGE  = "#ffa657"
BLUE    = "#58a6ff"
YELLOW  = "#e3b341"
PURPLE  = "#bc8cff"
BULL    = "#26a69a"
BEAR    = "#ef5350"


# ══════════════════════════════════════════════════════════════════════════════
#  CHART BUILDERS
# ══════════════════════════════════════════════════════════════════════════════

def _base_layout(**extra) -> dict:
    """Shared dark-theme layout settings for every figure."""
    return dict(
        paper_bgcolor=BG,
        plot_bgcolor=PANEL,
        font=dict(color=TEXT, family="monospace", size=11),
        xaxis=dict(gridcolor=BORDER, zerolinecolor=BORDER),
        yaxis=dict(gridcolor=BORDER, zerolinecolor=BORDER),
        margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(bgcolor=PANEL, bordercolor=BORDER, borderwidth=1,
                    font=dict(size=10)),
        **extra,
    )


def build_main_figure(df: pd.DataFrame, levels: list,
                      entries: list, exits: list,
                      trades: list) -> go.Figure:
    """
    Two-row figure: candlesticks + level lines on top, volume bars below.
    Trade markers are drawn as separate scatter traces so Dash can pick up
    click/hover events on them individually.
    """
    lv = pd.DataFrame(levels).set_index("dt") if levels else pd.DataFrame()

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.78, 0.22],
        vertical_spacing=0.02,
    )

    # ── Candlesticks ──────────────────────────────────────────────────────────
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"],
        low=df["Low"],  close=df["Close"],
        increasing_line_color=BULL, decreasing_line_color=BEAR,
        increasing_fillcolor=BULL, decreasing_fillcolor=BEAR,
        line_width=1, name="Price",
    ), row=1, col=1)

    # ── Rolling VP lines ──────────────────────────────────────────────────────
    if not lv.empty:
        for col, colour, dash, label in [
            ("ema", YELLOW, "dash",   f"EMA({S.CFG['ema_period']})"),
            ("vah", GREEN,  "dot",    "VAH"),
            ("poc", ORANGE, "solid",  "POC"),
            ("val", RED,    "dot",    "VAL"),
        ]:
            fig.add_trace(go.Scatter(
                x=lv.index, y=lv[col],
                line=dict(color=colour, width=1.2, dash=dash),
                opacity=0.85, name=label, hoverinfo="skip",
            ), row=1, col=1)

        # Faint value-area fill
        fig.add_trace(go.Scatter(
            x=pd.concat([lv.index.to_series(), lv.index.to_series()[::-1]]),
            y=pd.concat([lv["vah"], lv["val"][::-1]]),
            fill="toself", fillcolor="rgba(88,166,255,0.04)",
            line=dict(color="rgba(0,0,0,0)"),
            hoverinfo="skip", showlegend=False,
        ), row=1, col=1)

    # ── Per-trade SL / TP lines ───────────────────────────────────────────────
    # Build a mapping from entry_dt → trade so we can draw level lines
    trade_map: dict = {}
    for tr in trades:
        edt = tr.get("entry_dt")
        if edt is not None:
            trade_map[edt] = tr

    for e in entries:
        tr = trade_map.get(e["dt"]) or e   # fall back to entry dict if no match
        edt = e["dt"]
        # Find exit time: the trade's dt field is the close time
        matched = trade_map.get(edt)
        x_end   = matched["dt"] if matched else df.index[-1]

        for level, colour, label in [
            (e.get("sl"),  RED,    "SL"),
            (e.get("tp1"), ORANGE, "TP1"),
            (e.get("tp2"), GREEN,  "TP2"),
            (e.get("tp3"), PURPLE, "TP3"),
        ]:
            if level is None:
                continue
            fig.add_trace(go.Scatter(
                x=[edt, x_end], y=[level, level],
                mode="lines",
                line=dict(color=colour, width=0.8, dash="dot"),
                opacity=0.45, hoverinfo="skip", showlegend=False,
            ), row=1, col=1)

    # ── Entry markers ─────────────────────────────────────────────────────────
    for direction, marker, colour in [("long", "triangle-up", GREEN),
                                       ("short", "triangle-down", RED)]:
        side = [e for e in entries if e["dir"] == direction]
        if not side:
            continue

        hover_parts = []
        for e in side:
            fill  = e["price"]
            sl_p  = e.get("sl",  fill)
            tp1_p = e.get("tp1", fill)
            tp2_p = e.get("tp2", fill)
            tp3_p = e.get("tp3", fill)

            # Distances shown as % from fill
            def pct(a, b, sign=1):
                return f"{sign*(a-b)/b*100:+.2f}%" if b else "—"

            s = sign = 1 if direction == "long" else -1
            hover_parts.append(
                f"<b>{'LONG ▲' if direction == 'long' else 'SHORT ▼'}</b><br>"
                f"Fill : <b>{fill:,.2f}</b><br>"
                f"SL   : {sl_p:,.2f}  ({pct(sl_p, fill, s)})<br>"
                f"TP1 (POC) : {tp1_p:,.2f}  ({pct(tp1_p, fill, s)})<br>"
                f"TP2 (VAH/VAL) : {tp2_p:,.2f}  ({pct(tp2_p, fill, s)})<br>"
                f"TP3 (ext) : {tp3_p:,.2f}  ({pct(tp3_p, fill, s)})<br>"
                f"R/R (to TP2) : "
                f"{abs(tp2_p - fill) / max(abs(fill - sl_p), 1e-8):.2f}x"
            )

        fig.add_trace(go.Scatter(
            x=[e["dt"] for e in side],
            y=[e["price"] for e in side],
            mode="markers",
            marker=dict(symbol=marker, size=13, color=colour,
                        line=dict(color=BG, width=1)),
            name=f"{'Long' if direction == 'long' else 'Short'} entry",
            hovertemplate="%{customdata}<extra></extra>",
            customdata=hover_parts,
        ), row=1, col=1)

    # ── Exit markers ──────────────────────────────────────────────────────────
    exit_colours = {"stop": RED, "tp1": ORANGE, "tp2": GREEN, "tp3": PURPLE}
    for xtype in ("stop", "tp1", "tp2", "tp3"):
        pts = [x for x in exits if x["type"] == xtype]
        if not pts:
            continue
        label_map = {"stop": "Stop", "tp1": "Exit TP1", "tp2": "Exit TP2", "tp3": "Exit TP3"}
        fig.add_trace(go.Scatter(
            x=[x["dt"] for x in pts],
            y=[x["price"] for x in pts],
            mode="markers",
            marker=dict(symbol="x", size=9, color=exit_colours[xtype],
                        line=dict(color=BG, width=1)),
            name=label_map[xtype],
            hovertemplate=f"{label_map[xtype]}: %{{y:,.2f}}<extra></extra>",
        ), row=1, col=1)

    # ── Volume bars ───────────────────────────────────────────────────────────
    bull_mask = df["Close"] >= df["Open"]
    fig.add_trace(go.Bar(
        x=df.index,
        y=df["Volume"],
        marker_color=np.where(bull_mask, BULL, BEAR),
        opacity=0.6, name="Volume", showlegend=False,
        hovertemplate="%{x}<br>Vol: %{y:,.0f}<extra></extra>",
    ), row=2, col=1)

    # ── Default view: last 200 bars so candlesticks are immediately visible ───
    INITIAL_BARS = 200
    if len(df) > INITIAL_BARS:
        x_start = df.index[-INITIAL_BARS]
        x_end   = df.index[-1]
    else:
        x_start = df.index[0]
        x_end   = df.index[-1]

    # ── Layout ────────────────────────────────────────────────────────────────
    fig.update_layout(
        **_base_layout(
            xaxis_rangeslider_visible=False,
            hovermode="x unified",
            dragmode="pan",
        ),
        height=580,
    )
    fig.update_yaxes(
        tickformat=",.0f", row=1, col=1,
        gridcolor=BORDER, zerolinecolor=BORDER,
    )
    fig.update_yaxes(
        tickformat=".2s", row=2, col=1,
        gridcolor=BORDER, zerolinecolor=BORDER,
    )
    fig.update_xaxes(
        gridcolor=BORDER, zerolinecolor=BORDER,
        rangeslider_visible=False,
        range=[x_start, x_end],
    )
    # Range selector buttons at the top of the x-axis
    fig.update_xaxes(
        rangeselector=dict(
            bgcolor=PANEL, activecolor=BORDER,
            buttons=[
                dict(count=1,  label="1D",  step="day",  stepmode="backward"),
                dict(count=3,  label="3D",  step="day",  stepmode="backward"),
                dict(count=7,  label="1W",  step="day",  stepmode="backward"),
                dict(step="all", label="All"),
            ],
        ),
        row=1, col=1,
    )

    return fig


def build_vp_figure(df: pd.DataFrame, bins: int = 60,
                    title: str = "Volume Profile") -> go.Figure:
    """
    Horizontal volume profile for the given slice of data.
    Price on the y-axis, volume on the x-axis — matches the main chart orientation.
    """
    if df.empty or len(df) < 3:
        return go.Figure(layout=_base_layout(title=dict(text="No data", font=dict(size=11))))

    poc, val, vah = S.compute_vp(df, bins)

    lo, hi = float(df["Low"].min()), float(df["High"].max())
    edges  = np.linspace(lo, hi, bins + 1)
    mids   = (edges[:-1] + edges[1:]) / 2
    vols   = np.zeros(bins)

    for k in range(len(df)):
        b_lo  = float(df["Low"].iloc[k])
        b_hi  = float(df["High"].iloc[k])
        b_v   = float(df["Volume"].iloc[k])
        span  = b_hi - b_lo
        if span < 1e-10:
            idx = min(int((b_lo - lo) / (hi - lo) * bins), bins - 1)
            vols[idx] += b_v
        else:
            overlap = np.maximum(
                0.0,
                np.minimum(edges[1:], b_hi) - np.maximum(edges[:-1], b_lo),
            )
            vols += b_v * overlap / span

    # Colour each bucket: VAH zone → green, VAL zone → red, in-between → blue
    colours = [
        GREEN  if m >= vah else
        RED    if m <= val else
        BLUE
        for m in mids
    ]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=vols, y=mids,
        orientation="h",
        marker_color=colours,
        opacity=0.75,
        width=(hi - lo) / bins * 0.9,
        hovertemplate="Price: %{y:,.1f}<br>Vol: %{x:,.0f}<extra></extra>",
        name="Volume",
        showlegend=False,
    ))

    for level, colour, label in [(vah, GREEN, "VAH"),
                                  (poc, ORANGE, "POC"),
                                  (val, RED,    "VAL")]:
        fig.add_hline(
            y=level, line_color=colour, line_width=1.5, line_dash="dash",
            annotation_text=f" {label} {level:,.1f}",
            annotation_font=dict(color=colour, size=9),
            annotation_position="right",
        )

    n_bars = len(df)
    fig.update_layout(
        **_base_layout(
            title=dict(text=f"{title}  ({n_bars} bars)", font=dict(size=10)),
            showlegend=False,
        ),
        height=580,
        xaxis=dict(title="Volume", gridcolor=BORDER, tickformat=".2s"),
        yaxis=dict(title="Price",  gridcolor=BORDER, tickformat=",.0f"),
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════════
#  DASH APP
# ══════════════════════════════════════════════════════════════════════════════

def _inp(id_: str, value, type_="text", **kw):
    return dcc.Input(
        id=id_, value=value, type=type_,
        debounce=True,
        style=dict(
            backgroundColor=PANEL, color=TEXT, border=f"1px solid {BORDER}",
            borderRadius="4px", padding="6px 10px", fontFamily="monospace",
            fontSize="13px", width="110px", **kw,
        ),
    )


app = Dash(__name__, title="VAH/VAL Flip Viewer")
app.layout = html.Div(
    style={"backgroundColor": BG, "color": TEXT,
           "fontFamily": "monospace", "padding": "18px 22px"},
    children=[

        # ── Header ────────────────────────────────────────────────────────────
        html.Div([
            html.Span("VAH / VAL Flip  ·  Trade Viewer",
                      style={"color": BLUE, "fontSize": "18px",
                             "fontWeight": "bold"}),
            html.Span("  zoom the chart — the Volume Profile updates automatically",
                      style={"color": MUTED, "fontSize": "11px"}),
        ], style={"marginBottom": "16px"}),

        # ── Controls ──────────────────────────────────────────────────────────
        html.Div(
            style={"display": "flex", "gap": "14px", "alignItems": "flex-end",
                   "marginBottom": "16px", "flexWrap": "wrap"},
            children=[
                html.Div([
                    html.Label("Ticker", style={"color": MUTED, "fontSize": "11px",
                                                "display": "block", "marginBottom": "4px"}),
                    _inp("ticker", S.CFG["ticker"], width="100px"),
                ]),
                html.Div([
                    html.Label("Timeframe", style={"color": MUTED, "fontSize": "11px",
                                                   "display": "block", "marginBottom": "4px"}),
                    dcc.Dropdown(
                        id="interval",
                        options=[{"label": f"{k}  ·  {v['desc'][:38]}", "value": k}
                                 for k, v in S.TIMEFRAMES.items()],
                        value="5m",
                        clearable=False,
                        style={"backgroundColor": PANEL, "color": TEXT,
                               "border": f"1px solid {BORDER}",
                               "fontFamily": "monospace", "fontSize": "12px",
                               "width": "350px"},
                    ),
                ]),
                html.Div([
                    html.Label("Days back", style={"color": MUTED, "fontSize": "11px",
                                                   "display": "block", "marginBottom": "4px"}),
                    _inp("days", S.CFG["days"], type_="number", width="80px"),
                ]),
                html.Button(
                    "▶  Run Backtest", id="run-btn", n_clicks=0,
                    style={
                        "backgroundColor": "#1f6feb", "color": TEXT,
                        "border": "none", "borderRadius": "6px",
                        "padding": "8px 18px", "cursor": "pointer",
                        "fontFamily": "monospace", "fontSize": "13px",
                        "fontWeight": "bold",
                    },
                ),
                html.Div(id="status",
                         style={"color": MUTED, "fontSize": "11px",
                                "alignSelf": "center"}),
            ],
        ),

        # ── Main chart + VP side by side ──────────────────────────────────────
        html.Div(
            style={"display": "flex", "gap": "12px", "alignItems": "flex-start"},
            children=[
                # Main candlestick chart
                html.Div(
                    style={"flex": "1 1 0", "minWidth": "0"},
                    children=[
                        dcc.Loading(
                            type="circle", color=BLUE,
                            children=dcc.Graph(
                                id="main-chart",
                                config={
                                    "scrollZoom": True,
                                    "displayModeBar": True,
                                    "modeBarButtonsToRemove": ["autoScale2d"],
                                    "toImageButtonOptions": {"format": "png", "scale": 2},
                                    "doubleClick": "reset",
                                    "showTips": False,
                                },
                                style={"height": "580px"},
                            ),
                        ),
                    ],
                ),
                # Volume Profile (dynamic — updates on zoom)
                html.Div(
                    style={"flex": "0 0 280px"},
                    children=[
                        dcc.Graph(
                            id="vp-chart",
                            config={"displayModeBar": False},
                            style={"height": "580px"},
                        ),
                    ],
                ),
            ],
        ),

        # ── Trade detail card ─────────────────────────────────────────────────
        html.Div(
            id="trade-detail",
            style={
                "marginTop": "14px",
                "padding": "16px 20px",
                "backgroundColor": PANEL,
                "border": f"1px solid {BORDER}",
                "borderRadius": "8px",
                "fontSize": "13px",
                "lineHeight": "1.9",
                "minHeight": "60px",
            },
            children=html.Span(
                "Hover over a ▲ Long or ▼ Short marker on the chart "
                "to inspect its fill level, stop, and targets.",
                style={"color": MUTED},
            ),
        ),
    ],
)


# ── Callback: run backtest ─────────────────────────────────────────────────────
@app.callback(
    Output("main-chart",  "figure"),
    Output("vp-chart",    "figure"),
    Output("status",      "children"),
    Input("run-btn",      "n_clicks"),
    State("ticker",       "value"),
    State("interval",     "value"),
    State("days",         "value"),
    prevent_initial_call=False,
)
def run_backtest(n_clicks, ticker, interval, days):
    ticker   = (ticker or S.CFG["ticker"]).upper().strip()
    interval = interval or "5m"
    days     = int(days or S.CFG["days"])
    tf       = S.TIMEFRAMES[interval]
    lookback = tf["lookback"]

    cfg          = S.CFG.copy()
    cfg["ticker"] = ticker
    cfg["days"]   = days

    df = S.get_data(ticker, interval, days)
    min_needed = lookback * 2 + cfg["ema_period"] + 5
    if df.empty or len(df) < min_needed:
        msg = f"Only {len(df)} bars — need {min_needed}. Try more days."
        return go.Figure(layout=_base_layout()), go.Figure(layout=_base_layout()), msg

    trades, equity, levels, entries, exits = S.run_backtest(df, cfg, lookback)

    _G.update(df=df, trades=trades, levels=levels,
              entries=entries, exits=exits, cfg=cfg)

    main_fig = build_main_figure(df, levels, entries, exits, trades)
    vp_fig   = build_vp_figure(df, title="Vol Profile  (full range)")

    n_long  = sum(1 for e in entries if e["dir"] == "long")
    n_short = sum(1 for e in entries if e["dir"] == "short")
    status  = (f"{ticker} [{interval}]  ·  {len(df)} bars  ·  "
               f"{n_long} long  {n_short} short  ·  "
               f"zoom the chart → VP updates automatically")

    return main_fig, vp_fig, status


# ── Callback: update VP on zoom ────────────────────────────────────────────────
@app.callback(
    Output("vp-chart", "figure", allow_duplicate=True),
    Input("main-chart", "relayoutData"),
    prevent_initial_call=True,
)
def update_vp_on_zoom(relay):
    """
    Fires every time the user zooms or pans the main chart.
    Recomputes the Volume Profile for the newly visible price range.
    """
    df = _G.get("df")
    if df is None or relay is None:
        return go.Figure(layout=_base_layout())

    # Extract visible x-axis range from Plotly's relayoutData dict
    x0 = relay.get("xaxis.range[0]") or (relay.get("xaxis.range") or [None])[0]
    x1 = relay.get("xaxis.range[1]") or (relay.get("xaxis.range") or [None, None])[1]

    if x0 and x1:
        try:
            t0 = pd.Timestamp(x0)
            t1 = pd.Timestamp(x1)
            visible = df[(df.index >= t0) & (df.index <= t1)]
        except Exception:
            visible = df
    else:
        visible = df   # auto-range or reset zoom → show full dataset

    if len(visible) < 5:
        visible = df

    n = len(visible)
    title = f"Vol Profile  ({n} bars visible)"
    return build_vp_figure(visible, title=title)


# ── Callback: trade detail on hover ───────────────────────────────────────────
@app.callback(
    Output("trade-detail", "children"),
    Input("main-chart", "hoverData"),
    prevent_initial_call=True,
)
def show_trade_detail(hover):
    """
    When the user hovers over a Long ▲ or Short ▼ marker, display a
    formatted card showing the exact fill, stop, and all target levels.
    """
    if not hover:
        return html.Span("Hover a trade marker to see fill details.",
                         style={"color": MUTED})

    # Walk through every point in the hover event looking for a trade marker
    for pt in hover.get("points", []):
        cd = pt.get("customdata")
        if not cd or not isinstance(cd, str):
            continue
        if "Fill" not in cd:
            continue

        # customdata is already the formatted hover string — parse it back
        lines = cd.replace("<b>", "").replace("</b>", "").replace("<br>", "\n").split("\n")
        is_long = "LONG" in lines[0] if lines else True
        colour  = GREEN if is_long else RED

        children = [
            html.Div(lines[0],
                     style={"color": colour, "fontWeight": "bold",
                            "fontSize": "14px", "marginBottom": "8px"}),
        ]
        for line in lines[1:]:
            if not line.strip():
                continue
            label, _, value = line.partition(":")
            children.append(html.Div([
                html.Span(label + ":", style={"color": MUTED, "width": "180px",
                                              "display": "inline-block"}),
                html.Span(value.strip(), style={"color": TEXT}),
            ]))

        return html.Div(children)

    return html.Span("Hover a ▲ Long or ▼ Short entry marker to see its fill details.",
                     style={"color": MUTED})


# ══════════════════════════════════════════════════════════════════════════════
#  PROMPT & MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    ticker   = S.CFG["ticker"]
    interval = "5m"
    days     = S.CFG["days"]

    # Pre-load data so the chart is ready immediately when the browser opens
    print(f"\n  Pre-loading {ticker} [{interval}] {days}d …")
    cfg          = S.CFG.copy()
    cfg["ticker"] = ticker
    cfg["days"]   = days
    tf       = S.TIMEFRAMES[interval]
    lookback = tf["lookback"]

    df = S.get_data(ticker, interval, days)
    if len(df) >= lookback * 2 + cfg["ema_period"] + 5:
        trades, equity, levels, entries, exits = S.run_backtest(df, cfg, lookback)
        _G.update(df=df, trades=trades, levels=levels,
                  entries=entries, exits=exits, cfg=cfg)
        print(f"  {len(entries)} entries found  "
              f"({sum(1 for e in entries if e['dir']=='long')} long, "
              f"{sum(1 for e in entries if e['dir']=='short')} short)")
    else:
        print("  Not enough bars for signals — chart will show price only.")

    url = f"http://localhost:{PORT}"
    print(f"\n  Opening browser at {url}")
    print("  Zoom the main chart — the Volume Profile panel updates live.")
    print("  Hover over a ▲ ▼ marker to inspect fill levels.")
    print("  Press Ctrl+C to stop.\n")

    threading.Timer(1.2, lambda: webbrowser.open(url)).start()
    app.run(debug=False, port=PORT, use_reloader=False)


if __name__ == "__main__":
    main()
