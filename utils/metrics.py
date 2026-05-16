"""
Performance metric calculations — shared across all strategy files.

Usage:
    from utils.metrics import compute_stats

    stats = compute_stats(equity_series, trades_list, initial_capital=10_000)
    print(stats["sharpe"], stats["max_dd"])
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import List


def compute_stats(
    equity: pd.Series,
    trades: List[dict],
    initial_capital: float,
    total_costs: float = 0.0,
) -> dict:
    """
    Compute a standard set of backtest performance metrics.

    Parameters
    ----------
    equity          : daily (or bar-by-bar) equity Series with DatetimeIndex
    trades          : list of dicts, each with keys: entry_cost, pnl, ret_pct
    initial_capital : starting capital in dollars
    total_costs     : cumulative transaction costs in dollars

    Returns
    -------
    dict with keys:
        cagr, sharpe, sortino, max_dd, calmar,
        win_rate, avg_trade_pct, n_trades,
        turnover, cost_drag
    """
    if len(equity) < 10:
        nan = float("nan")
        return dict(cagr=nan, sharpe=nan, sortino=nan, max_dd=nan, calmar=nan,
                    win_rate=nan, avg_trade_pct=nan, n_trades=0,
                    turnover=nan, cost_drag=nan)

    # ── Return series ─────────────────────────────────────────────────────────
    daily_ret = equity.pct_change().dropna()
    mu        = daily_ret.mean()
    sig       = daily_ret.std(ddof=1)

    # ── CAGR ─────────────────────────────────────────────────────────────────
    n_years = (equity.index[-1] - equity.index[0]).days / 365.25
    cagr    = (equity.iloc[-1] / initial_capital) ** (1 / n_years) - 1 if n_years > 0 else 0.0

    # ── Sharpe (annualised, risk-free = 0) ───────────────────────────────────
    sharpe = mu / sig * 252 ** 0.5 if sig > 1e-12 else 0.0

    # ── Sortino ──────────────────────────────────────────────────────────────
    downside = daily_ret[daily_ret < 0]
    dsig     = downside.std(ddof=1) if len(downside) > 1 else 1e-9
    sortino  = mu / dsig * 252 ** 0.5 if dsig > 1e-12 else 0.0

    # ── Max drawdown ─────────────────────────────────────────────────────────
    roll_max = equity.cummax()
    dd       = (equity - roll_max) / roll_max
    max_dd   = float(dd.min())

    # ── Calmar ───────────────────────────────────────────────────────────────
    calmar = cagr / abs(max_dd) if abs(max_dd) > 1e-6 else 0.0

    # ── Trade-level metrics ───────────────────────────────────────────────────
    n_trades = len(trades)
    if n_trades > 0:
        pnls     = [t["pnl"]     for t in trades if "pnl"     in t]
        ret_pcts = [t["ret_pct"] for t in trades if "ret_pct" in t]
        win_rate     = sum(1 for p in pnls if p > 0) / len(pnls) if pnls else float("nan")
        avg_trade_pct = float(np.mean(ret_pcts)) if ret_pcts else float("nan")
    else:
        win_rate = avg_trade_pct = float("nan")

    # ── Cost metrics ─────────────────────────────────────────────────────────
    cost_drag = total_costs / initial_capital if initial_capital > 0 else 0.0

    # Annualised turnover: gross_volume / capital / years
    # Gross volume is recovered from total_costs / cost_rate_per_dollar
    from utils.constants import COMMISSION, SLIP_MKT
    cost_rate = SLIP_MKT + COMMISSION
    gross_vol  = total_costs / cost_rate if cost_rate > 0 else 0.0
    turnover   = gross_vol / initial_capital / n_years if n_years > 0 else 0.0

    return dict(
        cagr          = round(cagr, 4),
        sharpe        = round(sharpe, 3),
        sortino       = round(sortino, 3),
        max_dd        = round(max_dd, 4),
        calmar        = round(calmar, 3),
        win_rate      = round(win_rate, 4) if not np.isnan(win_rate) else float("nan"),
        avg_trade_pct = round(avg_trade_pct, 4) if not np.isnan(avg_trade_pct) else float("nan"),
        n_trades      = n_trades,
        turnover      = round(turnover, 3),
        cost_drag     = round(cost_drag, 4),
    )


def format_stats(stats: dict) -> str:
    """Return a compact single-line summary of key metrics."""
    def pct(x):
        return f"{x*100:+.1f}%" if not np.isnan(x) else "n/a"
    def f2(x):
        return f"{x:.2f}" if not np.isnan(x) else "n/a"

    return (
        f"CAGR {pct(stats['cagr'])}  "
        f"Sharpe {f2(stats['sharpe'])}  "
        f"MaxDD {pct(stats['max_dd'])}  "
        f"WinRate {pct(stats['win_rate'])}  "
        f"Trades {stats['n_trades']}"
    )
