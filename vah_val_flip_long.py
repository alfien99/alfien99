#!/usr/bin/env python3
"""
Strategy 1: VAH→VAL Flip Long Only

Pattern: When a prior period's VAH becomes the current period's VAL,
that level is a high-conviction support zone. Price broke above it once
(showing buyers dominated), then the market re-anchors its value area there.
A pullback to that flipped level is a long entry.

Entry rules:
  1. Compute rolling VP for the past `lookback` bars → get VAH(t)
  2. Compute rolling VP for the past `lookback` bars ending one period back → VAH(t-1)
  3. If |VAL(t) - VAH(t-1)| / VAH(t-1) < flip_tolerance → a flip has occurred
  4. Enter long when close pulls back into the VAL zone (close <= VAL(t) * (1 + entry_buffer))
  5. Only enter if price is above the 50-bar EMA (trend filter)

Exits:
  - 50% at POC
  - 70% of remainder at VAH
  - Rest at VAH + 1× (VAH - POC) extension
  - Stop: below VAL × (1 - stop_pct)

Usage:
  python vah_val_flip_long.py
  python vah_val_flip_long.py --ticker NQ=F --start 2023-01-01 --end 2025-01-01
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
    raw      = _download(cfg["ticker"], cfg["start"], cfg["end"])
    lookback = cfg["lookback"]
    bins     = cfg["vp_bins"]
    tol      = cfg["flip_tolerance"]
    buf      = cfg["entry_buffer"]
    stop_p   = cfg["stop_pct"]
    ema_win  = cfg["ema_window"]
    capital  = float(cfg["initial_capital"])

    ema = raw["Close"].ewm(span=ema_win, adjust=False).mean()

    in_pos      = False
    shares      = 0.0
    avg_px      = 0.0
    cost        = 0.0
    entry_stop  = 0.0
    entry_poc   = 0.0
    entry_vah   = 0.0
    entry_ext   = 0.0
    poc_done    = False
    vah_done    = False

    equity_records = []
    level_records  = []
    trade_records  = []
    entry_marks    = []
    exit_marks     = []
    flip_marks     = []

    min_i = max(lookback * 2, ema_win + 5)

    for i in range(min_i, len(raw)):
        date  = raw.index[i]
        close = float(raw["Close"].iloc[i])
        high  = float(raw["High"].iloc[i])
        low   = float(raw["Low"].iloc[i])

        poc_c, val_c, vah_c, _, _ = compute_vp(raw.iloc[i - lookback: i], bins)
        poc_p, val_p, vah_p, _, _ = compute_vp(raw.iloc[i - lookback * 2: i - lookback], bins)

        # "Flip" = value area stepped up: current VAL is above previous VAL,
        # AND current VAL is within tol of previous VAH (old resistance = new support).
        # Also accept if current VAL is simply above previous VAH (clean step up).
        val_above_prev_val = val_c > val_p
        near_prev_vah = (vah_p > 0 and abs(val_c - vah_p) / vah_p < tol) or val_c >= vah_p
        flip = val_above_prev_val and near_prev_vah
        uptrend = close > float(ema.iloc[i])

        stop_lvl = val_c * (1 - stop_p)
        ext_lvl  = vah_c + cfg["ext_mult"] * (vah_c - poc_c)

        level_records.append({"date": date, "poc": poc_c, "val": val_c,
                               "vah": vah_c, "vah_prev": vah_p,
                               "flip": flip, "ema": float(ema.iloc[i])})
        if flip:
            flip_marks.append({"date": date, "level": val_c})

        # ── Exits ────────────────────────────────────────────────────────────
        if in_pos:
            if low <= entry_stop:
                pnl = (entry_stop - avg_px) * shares
                capital += cost + pnl
                trade_records.append({"date": date, "type": "stop",
                                      "entry": avg_px, "exit": entry_stop, "pnl": pnl})
                exit_marks.append({"date": date, "price": entry_stop, "type": "stop"})
                in_pos = False; shares = cost = 0.0
                poc_done = vah_done = False

            elif not poc_done and high >= entry_poc:
                cs       = shares * cfg["poc_exit_frac"]
                pnl_p    = (entry_poc - avg_px) * cs
                capital += cs * avg_px + pnl_p
                cost    -= cs * avg_px
                shares  -= cs
                poc_done = True
                exit_marks.append({"date": date, "price": entry_poc, "type": "poc"})

            elif poc_done and not vah_done and high >= entry_vah:
                cs       = shares * cfg["vah_exit_frac"]
                pnl_p    = (entry_vah - avg_px) * cs
                capital += cs * avg_px + pnl_p
                cost    -= cs * avg_px
                shares  -= cs
                vah_done = True
                exit_marks.append({"date": date, "price": entry_vah, "type": "vah"})

            elif vah_done and shares > 0 and high >= entry_ext:
                pnl = (entry_ext - avg_px) * shares
                capital += cost + pnl
                trade_records.append({"date": date, "type": "target",
                                      "entry": avg_px, "exit": entry_ext, "pnl": pnl})
                exit_marks.append({"date": date, "price": entry_ext, "type": "ext"})
                in_pos = False; shares = cost = 0.0
                poc_done = vah_done = False

        # ── Entry: flip confirmed, bar's LOW touches VAL zone ────────────────
        # We model a limit order at VAL: if low dips to VAL, fill at VAL price.
        entry_px = val_c * (1 + buf)
        if not in_pos and flip and uptrend and low <= entry_px and close > stop_lvl:
            fill_px    = min(close, entry_px)   # limit fill
            risk_cash  = capital * cfg["risk_pct"]
            risk_per_sh = max(fill_px - stop_lvl, 1e-6)
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
                capital   -= order_cash
                entry_marks.append({"date": date, "price": fill_px})

        equity_records.append({"date": date,
                                "equity": capital + shares * close})

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
    entry_df  = pd.DataFrame(entry_marks) if entry_marks  else pd.DataFrame()
    exit_df   = pd.DataFrame(exit_marks)  if exit_marks   else pd.DataFrame()
    flip_df   = pd.DataFrame(flip_marks)  if flip_marks   else pd.DataFrame()
    return raw, trades_df, equity_df, levels_df, entry_df, exit_df, flip_df


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
    else:
        total_t = wins = 0; win_rate = avg_win = avg_loss = 0.0
    return {"Total Return": f"{total_r:+.1f}%", "Ann. Return": f"{ann_r:+.1f}%",
            "Max Drawdown": f"{max_dd:.1f}%", "Sharpe": f"{sharpe:.2f}",
            "Trades": str(total_t), "Win Rate": f"{win_rate:.0f}%",
            "Avg Win": f"${avg_win:,.0f}", "Avg Loss": f"${avg_loss:,.0f}",
            "Final Equity": f"${final:,.0f}"}


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
                 flip_df, cfg, stats, out="flip_long_results.png"):
    fig = plt.figure(figsize=(20, 15), facecolor=BG)
    gs  = GridSpec(4, 1, figure=fig, height_ratios=[3, 1, 1, 0.55], hspace=0.42)
    ax_p, ax_eq, ax_dd, ax_st = [fig.add_subplot(gs[i]) for i in range(4)]
    for ax in (ax_p, ax_eq, ax_dd, ax_st): _style(ax)

    sl = raw.loc[levels_df.index]
    ax_p.plot(sl.index, sl["Close"], color=BLUE, lw=1.0, label="Price")
    ax_p.plot(levels_df.index, levels_df["ema"],  color=YLW, lw=0.9, ls="--", alpha=0.7, label=f"EMA({cfg['ema_window']})")
    ax_p.fill_between(levels_df.index, levels_df["val"], levels_df["vah"],
                      alpha=0.07, color=GRN)
    ax_p.plot(levels_df.index, levels_df["val"], color=RED, lw=0.8, ls="--", alpha=0.9, label="VAL")
    ax_p.plot(levels_df.index, levels_df["poc"], color=ORG, lw=0.8, ls="-",  alpha=0.9, label="POC")
    ax_p.plot(levels_df.index, levels_df["vah"], color=GRN, lw=0.8, ls="--", alpha=0.9, label="VAH")

    # Highlight flip zones
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
        mpatches.Patch(color=YLW,  label="EMA trend filter"),
        mpatches.Patch(color=RED,  label="VAL"),
        mpatches.Patch(color=ORG,  label="POC"),
        mpatches.Patch(color=GRN,  label="VAH"),
        mpatches.Patch(color=YLW,  alpha=0.4, label="Flip zone (VAH→VAL)"),
        mpatches.Patch(color=GRN,  label="▲ Entry at flipped VAL"),
        mpatches.Patch(color=RED,  label="◆ Stop exit"),
        mpatches.Patch(color=PRP,  label="◆ Extension target"),
    ], loc="upper left", fontsize=7, facecolor=BG, labelcolor=FG, framealpha=0.8, ncol=3)
    ax_p.set_title(
        f"{cfg['ticker']}  ·  VAH→VAL Flip Long  ·  {cfg['start']} → {cfg['end']}",
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

    plt.suptitle("Strategy 1: VAH→VAL Flip Long  —  Buy the level that was once resistance, now confirmed support",
                 fontsize=11, color=FG, y=1.005, fontweight="bold")
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
    print(f"  Chart saved → {out}")
    plt.show()


# ── Config & entry ─────────────────────────────────────────────────────────────

DEFAULT_CONFIG = {
    "ticker":          "SPY",
    "start":           "2022-01-01",
    "end":             "2025-01-01",
    "lookback":        20,        # bars per VP window
    "vp_bins":         100,
    "initial_capital": 10_000,
    "ema_window":      50,        # trend filter: only long above this EMA
    "flip_tolerance":  0.03,      # VAL(t) within 3% of VAH(t-1), OR VAL(t) >= VAH(t-1)
    "entry_buffer":    0.005,     # enter when close <= VAL * (1 + 0.5%)
    "risk_pct":        0.02,      # risk 2% of capital per trade
    "max_position_pct": 0.40,    # never deploy more than 40% of capital in one trade
    "stop_pct":        0.015,     # stop 1.5% below VAL
    "poc_exit_frac":   0.50,
    "vah_exit_frac":   0.70,
    "ext_mult":        1.0,       # target = VAH + ext_mult × (VAH - POC)
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
    print(f"  EMA filter   : {cfg['ema_window']}-bar")
    print(f"  Risk/trade   : {cfg['risk_pct']*100:.1f}%")
    print(f"  Stop         : {cfg['stop_pct']*100:.1f}% below VAL")

    raw, trades_df, equity_df, levels_df, entry_df, exit_df, flip_df = run_backtest(cfg)
    stats = compute_stats(equity_df, cfg["initial_capital"], trades_df)

    print("\n" + "=" * 50)
    for k, v in stats.items(): print(f"  {k:<20} {v}")
    print("=" * 50)
    if not trades_df.empty and "type" in trades_df.columns:
        print("\n  Exit breakdown:")
        print(trades_df["type"].value_counts().to_string(header=False))
    print()

    plot_results(raw, trades_df, equity_df, levels_df, entry_df, exit_df,
                 flip_df, cfg, stats, out=args.out)


if __name__ == "__main__":
    main()
