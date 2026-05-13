#!/usr/bin/env python3
"""
Volume Profile Trading Strategy Backtester

Strategy rules:
  Entry  — Long when daily close drops below the rolling VAL (Value Area Low).
            Position size scales up with distance below VAL:
            each successive 1% further away triggers a larger add.
  Exits  — 50% of position closed at POC (Point of Control)
            70% of remainder closed at VAH (Value Area High)
            Remaining ~15% closed above VAH (VAH + 1× VAH–POC extension)
  Stop   — Fixed 3% below entry VAL

Usage:
  pip install yfinance matplotlib pandas numpy
  python volume_profile_strategy.py
  python volume_profile_strategy.py --ticker AAPL --start 2022-01-01 --end 2025-01-01
"""

import argparse
import sys
import warnings

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from matplotlib.gridspec import GridSpec

warnings.filterwarnings("ignore")

try:
    import yfinance as yf
    _YF_AVAILABLE = True
except ImportError:
    _YF_AVAILABLE = False


def _synthetic_ohlcv(ticker: str, start: str, end: str) -> pd.DataFrame:
    """
    Generates realistic synthetic OHLCV data using geometric Brownian motion
    with mild mean reversion and clustered volatility regimes.
    Used as a fallback when Yahoo Finance is unreachable (e.g. offline/CI).
    """
    rng    = np.random.default_rng(abs(hash(ticker)) % (2**31))
    dates  = pd.bdate_range(start, end)
    n      = len(dates)

    # Drift and volatility parameters
    mu      = 0.0003    # ~7.5 % annual drift
    sigma   = 0.012     # ~19 % annual vol
    S0      = 450.0

    # Volatility regimes (GARCH-lite)
    vols = np.full(n, sigma)
    for i in range(1, n):
        shock  = abs(rng.standard_normal())
        vols[i] = 0.92 * vols[i-1] + 0.08 * sigma * shock + 0.001

    log_ret = rng.standard_normal(n) * vols + mu
    closes  = S0 * np.exp(np.cumsum(log_ret))

    # Build OHLC from close
    hl_range = closes * vols * 2.2
    highs    = closes + hl_range * rng.uniform(0.3, 0.7, n)
    lows     = closes - hl_range * rng.uniform(0.3, 0.7, n)
    opens    = np.roll(closes, 1)
    opens[0] = S0

    volumes = (rng.integers(2_000_000, 8_000_000, n)
               * (1 + 3 * (vols / sigma - 1).clip(0))).astype(int)

    df = pd.DataFrame({
        "Open":   opens,
        "High":   highs,
        "Low":    lows,
        "Close":  closes,
        "Volume": volumes,
    }, index=dates)
    df.index.name = "Date"
    return df


def _download(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Download from Yahoo Finance; fall back to synthetic data on any error."""
    if _YF_AVAILABLE:
        try:
            raw = yf.download(ticker, start=start, end=end,
                              auto_adjust=True, progress=False)
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.droplevel(1)
            raw.dropna(inplace=True)
            if len(raw) > 20:
                print(f"  Downloaded {len(raw)} bars from Yahoo Finance.")
                return raw
            print("  Yahoo Finance returned too few bars — using synthetic data.")
        except Exception as e:
            print(f"  Yahoo Finance unavailable ({e}) — using synthetic data.")
    else:
        print("  yfinance not installed — using synthetic data.")
        print("  Install with:  pip install yfinance")

    print(f"  Generating synthetic OHLCV for {ticker} ({start} → {end}) …")
    return _synthetic_ohlcv(ticker, start, end)


# ── Volume profile ─────────────────────────────────────────────────────────────

def compute_vp(ohlcv: pd.DataFrame, bins: int = 100):
    """
    Distribute each bar's volume across price bins proportional to the
    fraction of the bar's high-low range that overlaps with each bin.
    Returns (poc, val, vah, bin_midpoints, volume_per_bin).
    """
    lo = float(ohlcv["Low"].min())
    hi = float(ohlcv["High"].max())
    if hi <= lo + 1e-8:
        mid = (hi + lo) / 2
        return mid, mid, mid, np.array([mid]), np.array([1.0])

    edges = np.linspace(lo, hi, bins + 1)
    mids  = (edges[:-1] + edges[1:]) / 2
    vol   = np.zeros(bins)

    bar_lo  = ohlcv["Low"].values.astype(float)
    bar_hi  = ohlcv["High"].values.astype(float)
    bar_vol = ohlcv["Volume"].values.astype(float)

    for k in range(len(bar_lo)):
        rng = bar_hi[k] - bar_lo[k]
        if rng < 1e-10:
            idx = min(int((bar_lo[k] - lo) / (hi - lo) * bins), bins - 1)
            vol[idx] += bar_vol[k]
        else:
            overlap = np.maximum(
                0.0,
                np.minimum(edges[1:], bar_hi[k]) - np.maximum(edges[:-1], bar_lo[k]),
            )
            vol += bar_vol[k] * overlap / rng

    poc_idx = int(np.argmax(vol))
    poc     = mids[poc_idx]

    # Expand from POC symmetrically until 70 % of total volume is captured
    total  = vol.sum()
    target = total * 0.70
    li = hi_i = poc_idx
    acc = vol[poc_idx]

    while acc < target:
        add_lo = vol[li - 1]   if li   > 0       else -1.0
        add_hi = vol[hi_i + 1] if hi_i < bins - 1 else -1.0
        if add_lo < 0 and add_hi < 0:
            break
        if add_lo >= add_hi:
            li  -= 1
            acc += vol[li]
        else:
            hi_i += 1
            acc  += vol[hi_i]

    return poc, mids[li], mids[hi_i], mids, vol


# ── Back-test ──────────────────────────────────────────────────────────────────

def run_backtest(cfg: dict):
    raw = _download(cfg["ticker"], cfg["start"], cfg["end"])

    if len(raw) < cfg["lookback"] + 5:
        sys.exit("Not enough data — try a longer date range.")

    lookback = cfg["lookback"]
    capital  = float(cfg["initial_capital"])

    # ── Position state ──
    in_pos        = False
    pos_shares    = 0.0
    pos_avg       = 0.0      # average entry price
    pos_cost      = 0.0      # total cash deployed (used for accounting)
    entry_val     = 0.0
    entry_poc     = 0.0
    entry_vah     = 0.0
    entry_stop    = 0.0
    entry_above   = 0.0
    poc_done      = False
    vah_done      = False
    # One boolean per scale tranche (0 = base entry, 1..max_scales = scale-ins)
    n_tranches    = cfg["max_scales"] + 1
    tranches_done = [False] * n_tranches

    # ── Output containers ──
    equity_records = []
    level_records  = []
    trade_records  = []
    entry_markers  = []
    exit_markers   = []

    for i in range(lookback, len(raw)):
        date  = raw.index[i]
        close = float(raw["Close"].iloc[i])
        high  = float(raw["High"].iloc[i])
        low   = float(raw["Low"].iloc[i])

        poc, val, vah, _mids, _vpvol = compute_vp(raw.iloc[i - lookback: i], cfg["vp_bins"])
        stop  = val  * (1 - cfg["stop_pct"])
        above = vah  + cfg["above_vah_ext"] * (vah - poc)

        level_records.append({"date": date, "poc": poc, "val": val,
                               "vah": vah, "above": above})

        stopped_today = False

        # ── Exit logic ──────────────────────────────────────────────────────
        if in_pos:

            # Stop loss  (checked first — most conservative bar assumption)
            if low <= entry_stop:
                pnl = (entry_stop - pos_avg) * pos_shares
                capital += pos_cost + pnl
                trade_records.append({
                    "date": date, "type": "stop",
                    "entry": pos_avg, "exit": entry_stop, "pnl": pnl,
                })
                exit_markers.append({"date": date, "price": entry_stop, "type": "stop"})
                in_pos        = False
                pos_shares    = pos_cost = 0.0
                poc_done      = vah_done = False
                tranches_done = [False] * n_tranches
                stopped_today = True

            # Partial exit at POC — 50 % of remaining shares
            elif not poc_done and high >= entry_poc:
                close_sh  = pos_shares * cfg["poc_exit_frac"]
                pnl_p     = (entry_poc - pos_avg) * close_sh
                capital  += close_sh * pos_avg + pnl_p
                pos_cost -= close_sh * pos_avg
                pos_shares -= close_sh
                poc_done  = True
                exit_markers.append({"date": date, "price": entry_poc, "type": "poc"})

            # Partial exit at VAH — 70 % of remaining (≈ 35 % of original)
            elif poc_done and not vah_done and high >= entry_vah:
                close_sh   = pos_shares * cfg["vah_exit_frac"]
                pnl_p      = (entry_vah - pos_avg) * close_sh
                capital   += close_sh * pos_avg + pnl_p
                pos_cost  -= close_sh * pos_avg
                pos_shares -= close_sh
                vah_done   = True
                exit_markers.append({"date": date, "price": entry_vah, "type": "vah"})

            # Final exit above VAH — remaining ~15 % of original
            elif vah_done and pos_shares > 0 and high >= entry_above:
                pnl      = (entry_above - pos_avg) * pos_shares
                capital += pos_cost + pnl
                trade_records.append({
                    "date": date, "type": "target",
                    "entry": pos_avg, "exit": entry_above, "pnl": pnl,
                })
                exit_markers.append({"date": date, "price": entry_above, "type": "above"})
                in_pos        = False
                pos_shares    = pos_cost = 0.0
                poc_done      = vah_done = False
                tranches_done = [False] * n_tranches

        # ── Entry / scale-in logic ───────────────────────────────────────────
        if close < val and not stopped_today:
            equity = capital + pos_shares * close

            for t in range(n_tranches):
                if tranches_done[t]:
                    continue

                # Reference VAL: use the VAL at entry for existing position
                ref_val   = entry_val if in_pos else val
                threshold = ref_val * (1.0 - t * cfg["scale_step_pct"])

                if close > threshold:
                    continue    # haven't fallen far enough for this tranche yet

                size_pct   = cfg["base_risk_pct"] * (cfg["scale_factor"] ** t)
                order_cash = equity * size_pct

                if order_cash < 50 or capital < order_cash:
                    tranches_done[t] = True   # skip — insufficient capital
                    continue

                new_shares = order_cash / close

                if not in_pos:
                    pos_avg    = close
                    pos_shares = new_shares
                    pos_cost   = order_cash
                    entry_val  = val
                    entry_poc  = poc
                    entry_vah  = vah
                    entry_stop = stop
                    entry_above= above
                    poc_done   = False
                    vah_done   = False
                    in_pos     = True
                else:
                    # Weighted average entry price
                    total_cost  = pos_cost + order_cash
                    pos_avg     = (pos_avg * pos_shares + close * new_shares) / (pos_shares + new_shares)
                    pos_shares += new_shares
                    pos_cost    = total_cost

                capital -= order_cash
                tranches_done[t] = True
                entry_markers.append({"date": date, "price": close, "level": t})

        equity_records.append({"date": date, "equity": capital + pos_shares * close})

    # Close any open position at last close price
    if in_pos and pos_shares > 0:
        lp  = float(raw["Close"].iloc[-1])
        pnl = (lp - pos_avg) * pos_shares
        capital += pos_cost + pnl
        trade_records.append({
            "date": raw.index[-1], "type": "expired",
            "entry": pos_avg, "exit": lp, "pnl": pnl,
        })

    equity_df = pd.DataFrame(equity_records).set_index("date")
    levels_df = pd.DataFrame(level_records).set_index("date")
    trades_df = (pd.DataFrame(trade_records)
                 if trade_records else pd.DataFrame(columns=["pnl", "type"]))
    entry_df  = pd.DataFrame(entry_markers) if entry_markers else pd.DataFrame()
    exit_df   = pd.DataFrame(exit_markers)  if exit_markers  else pd.DataFrame()

    return raw, trades_df, equity_df, levels_df, entry_df, exit_df


# ── Statistics ─────────────────────────────────────────────────────────────────

def compute_stats(equity_df, initial_capital, trades_df):
    final   = float(equity_df["equity"].iloc[-1])
    total_r = (final - initial_capital) / initial_capital * 100
    n_days  = max((equity_df.index[-1] - equity_df.index[0]).days, 1)
    ann_r   = ((final / initial_capital) ** (365 / n_days) - 1) * 100

    roll_max = equity_df["equity"].cummax()
    dd       = (equity_df["equity"] - roll_max) / roll_max * 100
    max_dd   = float(dd.min())

    daily_r = equity_df["equity"].pct_change().dropna()
    sharpe  = (daily_r.mean() / daily_r.std() * np.sqrt(252)
               if daily_r.std() > 0 else 0.0)

    if not trades_df.empty and "pnl" in trades_df.columns:
        wins     = int((trades_df["pnl"] > 0).sum())
        total_t  = len(trades_df)
        win_rate = wins / total_t * 100 if total_t else 0.0
        avg_win  = float(trades_df.loc[trades_df["pnl"] > 0, "pnl"].mean() or 0)
        avg_loss = float(trades_df.loc[trades_df["pnl"] <= 0, "pnl"].mean() or 0)
    else:
        wins = total_t = 0
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


# ── Visualisation ──────────────────────────────────────────────────────────────

BG   = "#0d1117"
GRID = "#21262d"
FG   = "#e6edf3"
BLUE = "#58a6ff"
RED  = "#f85149"
ORG  = "#ffa657"
GRN  = "#3fb950"
PRP  = "#bc8cff"

SCALE_COLORS = [BLUE, ORG, RED, PRP, GRN]
EXIT_COLORS  = {"stop": RED, "poc": ORG, "vah": GRN, "above": PRP}


def _style(ax):
    ax.set_facecolor(BG)
    ax.tick_params(colors=FG, labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID)
    ax.xaxis.label.set_color(FG)
    ax.yaxis.label.set_color(FG)
    ax.title.set_color(FG)
    ax.grid(color=GRID, alpha=0.6, lw=0.5)


def plot_results(raw, trades_df, equity_df, levels_df, entry_df, exit_df, cfg, stats, out="backtest_results.png"):
    fig = plt.figure(figsize=(20, 15), facecolor=BG)
    gs  = GridSpec(4, 2, figure=fig,
                   height_ratios=[3, 1, 1, 0.55],
                   hspace=0.45, wspace=0.25)

    ax_price  = fig.add_subplot(gs[0, 0])
    ax_vp     = fig.add_subplot(gs[0, 1])
    ax_equity = fig.add_subplot(gs[1, :])
    ax_dd     = fig.add_subplot(gs[2, :])
    ax_stats  = fig.add_subplot(gs[3, :])

    for ax in (ax_price, ax_vp, ax_equity, ax_dd, ax_stats):
        _style(ax)

    # ── Price chart ──────────────────────────────────────────────────────────
    price_slice = raw.loc[levels_df.index]
    ax_price.plot(price_slice.index, price_slice["Close"],
                  color=BLUE, lw=1.0, zorder=2, label="Price")

    ax_price.fill_between(levels_df.index, levels_df["val"], levels_df["vah"],
                          alpha=0.07, color=GRN, label="Value Area")
    ax_price.plot(levels_df.index, levels_df["val"],   color=RED, lw=0.8, ls="--", alpha=0.9, label="VAL")
    ax_price.plot(levels_df.index, levels_df["poc"],   color=ORG, lw=0.8, ls="-",  alpha=0.9, label="POC")
    ax_price.plot(levels_df.index, levels_df["vah"],   color=GRN, lw=0.8, ls="--", alpha=0.9, label="VAH")
    ax_price.plot(levels_df.index, levels_df["above"], color=PRP, lw=0.6, ls=":",  alpha=0.55, label="Above Target")

    if not entry_df.empty:
        for _, r in entry_df.iterrows():
            c = SCALE_COLORS[min(int(r["level"]), len(SCALE_COLORS) - 1)]
            ax_price.scatter(r["date"], r["price"], marker="^", s=65, color=c,
                             zorder=5, edgecolors="white", linewidths=0.4)

    if not exit_df.empty:
        for _, r in exit_df.iterrows():
            ax_price.scatter(r["date"], r["price"], marker="v", s=65,
                             color=EXIT_COLORS.get(r["type"], FG),
                             zorder=5, edgecolors="white", linewidths=0.4)

    legend_items = [
        mpatches.Patch(color=BLUE, label="Price"),
        mpatches.Patch(color=GRN,  label="Value Area (70% vol)"),
        mpatches.Patch(color=RED,  label="VAL / Stop exit ▼"),
        mpatches.Patch(color=ORG,  label="POC / Scale-1 entry ▲"),
        mpatches.Patch(color=GRN,  label="VAH exit ▼"),
        mpatches.Patch(color=PRP,  label="Above-VAH exit ▼"),
        mpatches.Patch(color=BLUE, label="Base entry ▲ (lvl 0)"),
    ]
    ax_price.legend(handles=legend_items, loc="upper left", fontsize=7,
                    facecolor=BG, labelcolor=FG, framealpha=0.8, ncol=2)
    ax_price.set_title(
        f"{cfg['ticker']}  ·  Volume Profile Strategy  ·  {cfg['start']} → {cfg['end']}",
        fontsize=11, fontweight="bold",
    )
    ax_price.set_ylabel("Price ($)")

    # ── Volume profile (most recent window) ──────────────────────────────────
    last_win = raw.iloc[-cfg["lookback"]:]
    poc, val, vah, mids, vpvol = compute_vp(last_win, cfg["vp_bins"])
    norm = Normalize(vmin=vpvol.min(), vmax=vpvol.max())
    bh   = (mids[1] - mids[0]) * 0.9 if len(mids) > 1 else 1.0

    for b in range(len(mids)):
        ax_vp.barh(mids[b], vpvol[b], height=bh,
                   color=plt.cm.plasma(norm(vpvol[b])), alpha=0.85)

    ax_vp.axhline(poc, color=ORG, lw=1.8, label=f"POC  ${poc:.2f}")
    ax_vp.axhline(val, color=RED, lw=1.8, ls="--", label=f"VAL  ${val:.2f}")
    ax_vp.axhline(vah, color=GRN, lw=1.8, ls="--", label=f"VAH  ${vah:.2f}")
    ax_vp.set_title(f"Volume Profile — last {cfg['lookback']} bars", fontsize=10)
    ax_vp.set_xlabel("Volume")
    ax_vp.legend(fontsize=8, facecolor=BG, labelcolor=FG, loc="lower right")

    # ── Equity curve ─────────────────────────────────────────────────────────
    eq = equity_df["equity"]
    ax_equity.plot(eq.index, eq, color=GRN, lw=1.5)
    ax_equity.axhline(cfg["initial_capital"], color=FG, lw=0.7, ls="--", alpha=0.35)
    ax_equity.fill_between(eq.index, cfg["initial_capital"], eq,
                           where=eq >= cfg["initial_capital"], color=GRN, alpha=0.12)
    ax_equity.fill_between(eq.index, cfg["initial_capital"], eq,
                           where=eq <  cfg["initial_capital"], color=RED, alpha=0.12)
    ax_equity.set_title("Equity Curve", fontsize=10)
    ax_equity.set_ylabel("Equity ($)")

    # ── Drawdown ─────────────────────────────────────────────────────────────
    roll_max = eq.cummax()
    dd_pct   = (eq - roll_max) / roll_max * 100
    ax_dd.fill_between(dd_pct.index, dd_pct, 0, color=RED, alpha=0.55)
    ax_dd.plot(dd_pct.index, dd_pct, color=RED, lw=0.8)
    ax_dd.set_title("Drawdown (%)", fontsize=10)
    ax_dd.set_ylabel("DD %")

    # ── Stats table ───────────────────────────────────────────────────────────
    ax_stats.axis("off")
    col_labels = list(stats.keys())
    col_vals   = [str(v) for v in stats.values()]

    tbl = ax_stats.table(
        cellText=[col_vals],
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1, 2.4)
    for (row, col), cell in tbl.get_celld().items():
        cell.set_facecolor("#161b22" if row == 0 else BG)
        cell.set_edgecolor(GRID)
        cell.set_text_props(color=FG)

    plt.suptitle(
        "Volume Profile Strategy  —  Long below VAL, scale in on weakness, exit at POC / VAH / extension",
        fontsize=12, color=FG, y=1.005, fontweight="bold",
    )

    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
    print(f"  Chart saved → {out}")
    plt.show()


# ── Entry point ────────────────────────────────────────────────────────────────

DEFAULT_CONFIG = {
    # Data
    "ticker":   "SPY",
    "start":    "2022-01-01",
    "end":      "2025-01-01",
    "lookback": 20,          # bars in rolling volume profile window
    "vp_bins":  100,         # price-level resolution of the volume profile

    # Capital
    "initial_capital": 10_000,

    # Sizing — tranche 0 = base entry, tranches 1-N are scale-ins
    # Size of tranche t  =  base_risk_pct  ×  scale_factor^t
    "base_risk_pct":  0.05,   # 5 % of equity for the base tranche
    "scale_step_pct": 0.01,   # add next tranche every additional 1 % below VAL
    "scale_factor":   1.5,    # each tranche is 1.5× the base size
    "max_scales":     4,      # maximum 4 additional tranches (5 total)

    # Exits — fractions of *remaining* shares at each level
    "poc_exit_frac":  0.50,   # close 50 % of remaining at POC
    "vah_exit_frac":  0.70,   # close 70 % of remainder at VAH (≈ 35 % of original)
    # Final ~15 % exits at:
    "above_vah_ext":  1.0,    # above_target = VAH + 1.0 × (VAH − POC)

    # Stop loss
    "stop_pct": 0.03,         # stop = entry VAL × (1 − 3 %)
}


def _ask(label: str, default: str, width: int = 24) -> str:
    """Prompt the user; return default if they press Enter or input is non-interactive."""
    try:
        answer = input(f"  {label:<{width}} [{default}]: ").strip()
        return answer if answer else default
    except (EOFError, KeyboardInterrupt):
        print()
        return default


def _parse_capital(raw: str) -> float:
    cleaned = raw.replace(",", "").replace("$", "").replace("_", "").strip()
    return float(cleaned)


def _validate_date(s: str) -> str:
    pd.Timestamp(s)   # raises ValueError on bad format
    return s


def _interactive_config(cfg: dict) -> dict:
    """Ask the user for the four most-common settings; all others keep defaults."""
    print("\n  Press Enter to accept the value shown in [ ].\n")

    while True:
        ticker = _ask("Ticker", cfg["ticker"]).upper()
        if ticker:
            break
        print("  Ticker cannot be empty.")

    while True:
        start = _ask("Start date (YYYY-MM-DD)", cfg["start"])
        try:
            _validate_date(start)
            break
        except Exception:
            print(f"  '{start}' is not a valid date — use YYYY-MM-DD format.")

    while True:
        end = _ask("End date   (YYYY-MM-DD)", cfg["end"])
        try:
            _validate_date(end)
            if pd.Timestamp(end) > pd.Timestamp(start):
                break
            print("  End date must be after start date.")
        except Exception:
            print(f"  '{end}' is not a valid date — use YYYY-MM-DD format.")

    while True:
        cap_str = _ask("Starting capital ($)", f"{cfg['initial_capital']:,.0f}")
        try:
            cap = _parse_capital(cap_str)
            if cap >= 100:
                break
            print("  Capital must be at least $100.")
        except ValueError:
            print(f"  '{cap_str}' is not a valid number.")

    while True:
        lb_str = _ask("VP lookback (bars)", str(cfg["lookback"]))
        try:
            lb = int(lb_str)
            if lb >= 5:
                break
            print("  Lookback must be at least 5 bars.")
        except ValueError:
            print(f"  '{lb_str}' is not a valid integer.")

    cfg = cfg.copy()
    cfg["ticker"]          = ticker
    cfg["start"]           = start
    cfg["end"]             = end
    cfg["initial_capital"] = cap
    cfg["lookback"]        = lb
    return cfg


def main():
    parser = argparse.ArgumentParser(
        description="Volume Profile Strategy Backtester",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "If run with no arguments an interactive prompt lets you set\n"
            "ticker, dates, capital and lookback before the backtest starts.\n\n"
            "Examples:\n"
            "  python volume_profile_strategy.py\n"
            "  python volume_profile_strategy.py --ticker AAPL --start 2020-01-01 --end 2025-01-01\n"
            "  python volume_profile_strategy.py --ticker QQQ  --capital 50000 --lookback 10\n"
        ),
    )
    parser.add_argument("--ticker",   default=None,  help="Ticker symbol (e.g. AAPL, QQQ, TSLA)")
    parser.add_argument("--start",    default=None,  help="Start date YYYY-MM-DD")
    parser.add_argument("--end",      default=None,  help="End date   YYYY-MM-DD")
    parser.add_argument("--capital",  type=float, default=None, help="Starting capital in USD")
    parser.add_argument("--lookback", type=int,   default=None, help="Rolling VP window in bars")
    parser.add_argument("--out", default="backtest_results.png", help="Output chart path")
    parser.add_argument("--no-prompt", action="store_true",
                        help="Skip interactive prompt and use defaults / CLI args only")
    args = parser.parse_args()

    cfg = DEFAULT_CONFIG.copy()

    # Apply any CLI overrides first
    if args.ticker:   cfg["ticker"]          = args.ticker.upper()
    if args.start:    cfg["start"]           = args.start
    if args.end:      cfg["end"]             = args.end
    if args.capital:  cfg["initial_capital"] = args.capital
    if args.lookback: cfg["lookback"]        = args.lookback

    # Show interactive prompt unless the user explicitly passed --no-prompt
    # or supplied every key argument on the command line
    cli_fully_specified = all([args.ticker, args.start, args.end, args.capital])
    if not args.no_prompt and not cli_fully_specified:
        print("\n╔══════════════════════════════════════════════════╗")
        print("║   Volume Profile Strategy  —  Configuration      ║")
        print("╚══════════════════════════════════════════════════╝")
        cfg = _interactive_config(cfg)

    print("\nVolume Profile Strategy Backtester")
    print("=" * 50)
    print(f"  Ticker   : {cfg['ticker']}")
    print(f"  Period   : {cfg['start']} → {cfg['end']}")
    print(f"  Capital  : ${cfg['initial_capital']:,.0f}")
    print(f"  Lookback : {cfg['lookback']} bars")
    print(f"  Tranches : base + {cfg['max_scales']} scale-ins "
          f"(×{cfg['scale_factor']} each, every {cfg['scale_step_pct']*100:.1f}% below VAL)")
    print(f"  Exits    : {cfg['poc_exit_frac']*100:.0f}% @ POC  ·  "
          f"{cfg['vah_exit_frac']*100:.0f}% of remainder @ VAH  ·  rest above VAH")
    print(f"  Stop     : {cfg['stop_pct']*100:.1f}% below entry VAL")

    raw, trades_df, equity_df, levels_df, entry_df, exit_df = run_backtest(cfg)

    stats = compute_stats(equity_df, cfg["initial_capital"], trades_df)

    print("\n" + "=" * 50)
    print("  RESULTS")
    print("=" * 50)
    for k, v in stats.items():
        print(f"  {k:<20} {v}")
    print("=" * 50)

    if not trades_df.empty and "type" in trades_df.columns:
        print("\n  Exit breakdown:")
        print(trades_df["type"].value_counts().to_string(header=False))
        print()

    plot_results(raw, trades_df, equity_df, levels_df, entry_df, exit_df, cfg, stats, out=args.out)


if __name__ == "__main__":
    main()
