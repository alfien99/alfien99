#!/usr/bin/env python3
"""
Walk-forward evaluation for the VAH→VAL flip long strategy.

Usage:
    python walk_forward_flip.py --data qqq_daily.csv

Download real QQQ data first:
    python -c "
    import yfinance as yf
    yf.download('QQQ', start='2010-01-01', end='2025-01-01',
                auto_adjust=True).to_csv('qqq_daily.csv')
    "

Methodology:
  - Rolling 2-year train / 1-year test windows, starting 2012-01-01
  - Grid search on TRAIN only (4 params × ≤5 values = ≤625 combos)
  - Objective: maximise Sharpe on train window
  - Record only OUT-OF-SAMPLE metrics on test window
  - Buy-and-hold QQQ over the same OOS windows as the benchmark
  - Stop immediately on synthetic data with a clear error

Parameters optimised:
  lookback       [10, 15, 20, 30, 40]
  flip_tolerance [0.01, 0.02, 0.03, 0.05]
  stop_pct       [0.010, 0.015, 0.020, 0.030]
  ema_window     [20, 50, 100, 200]

Section 5 acceptance criterion:
  Median OOS strategy Sharpe must exceed median B&H Sharpe by ≥ 0.10.
  If it doesn't, the script says so plainly and explains why.
"""

import argparse
import itertools
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# Re-use the strategy's cost constants and pure functions
from vah_val_flip_long import (
    compute_levels,
    generate_signals,
    simulate_trades,
    buy_lim,
    sell_lim,
    COMMISSION,
    SLIP_MKT,
)

# ── Constants ──────────────────────────────────────────────────────────────────

TRAIN_YRS   = 2
TEST_YRS    = 1
INIT_CAP    = 10_000.0
FIRST_TEST  = 2012   # first test year; needs 2 prior years of train data

GRID = {
    "lookback":        [10, 15, 20, 30, 40],
    "flip_tolerance":  [0.01, 0.02, 0.03, 0.05],
    "stop_pct":        [0.010, 0.015, 0.020, 0.030],
    "ema_window":      [20, 50, 100, 200],
}

BASE_CFG = {
    "vp_bins":          100,
    "initial_capital":  INIT_CAP,
    "entry_buffer":     0.005,
    "risk_pct":         0.02,
    "max_position_pct": 0.40,
    "poc_exit_frac":    0.50,
    "vah_exit_frac":    0.70,
    "ext_mult":         1.0,
    # grid params filled in per window:
    "lookback":         20,
    "flip_tolerance":   0.03,
    "stop_pct":         0.015,
    "ema_window":       50,
}


# ── Metrics ────────────────────────────────────────────────────────────────────

def _metrics(equity: pd.Series, initial: float) -> dict:
    """Return CAGR, Sharpe, Sortino, MaxDD from an equity Series."""
    if len(equity) < 10:
        return dict(cagr=np.nan, sharpe=np.nan, sortino=np.nan, max_dd=np.nan)
    n_days   = max((equity.index[-1] - equity.index[0]).days, 1)
    final    = float(equity.iloc[-1])
    cagr     = (final / initial) ** (365 / n_days) - 1
    dr       = equity.pct_change().dropna()
    sig      = dr.std(ddof=1)
    sharpe   = dr.mean() / sig * 252**0.5 if sig > 1e-12 else 0.0
    down     = dr[dr < 0]
    dsig     = down.std(ddof=1) if len(down) > 1 else 1e-9
    sortino  = dr.mean() / dsig * 252**0.5 if dsig > 1e-12 else 0.0
    roll_max = equity.cummax()
    max_dd   = float(((equity - roll_max) / roll_max).min())
    return dict(cagr=cagr, sharpe=sharpe, sortino=sortino, max_dd=max_dd)


def _bah_equity(price_series: pd.Series, initial: float) -> pd.Series:
    """Buy-and-hold: one buy_lim entry, one sell_lim exit, same costs as strategy."""
    entry   = buy_lim(float(price_series.iloc[0]))
    shares  = initial / entry
    equity  = shares * price_series.copy().astype(float)
    equity.iloc[-1] = shares * sell_lim(float(price_series.iloc[-1]))
    return equity


# ── Strategy runner (silent — no prints during grid search) ───────────────────

def _run_strategy_silent(data: pd.DataFrame, cfg: dict) -> pd.Series:
    """
    Run the full strategy pipeline and return the equity curve.
    Suppresses all print output so the grid search stays readable.
    """
    import io, contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        try:
            levels  = compute_levels(data, cfg)
            signals = generate_signals(data, levels, cfg)
            _, equity_df, _, _, _ = simulate_trades(data, signals, levels, cfg)
            return equity_df["equity"]
        except Exception:
            return pd.Series(dtype=float)


# ── Grid search on train window ────────────────────────────────────────────────

def _grid_search(train: pd.DataFrame) -> dict:
    """Return the parameter combo with the highest Sharpe on the train window."""
    keys       = list(GRID.keys())
    best_sh    = -np.inf
    best_params = {}

    for combo in itertools.product(*[GRID[k] for k in keys]):
        cfg = {**BASE_CFG, **dict(zip(keys, combo)), "initial_capital": INIT_CAP}
        eq  = _run_strategy_silent(train, cfg)
        if len(eq) < 10:
            continue
        m = _metrics(eq, INIT_CAP)
        if np.isfinite(m["sharpe"]) and m["sharpe"] > best_sh:
            best_sh     = m["sharpe"]
            best_params = dict(zip(keys, combo))

    return best_params if best_params else {k: BASE_CFG[k] for k in keys}


# ── Walk-forward loop ──────────────────────────────────────────────────────────

def walk_forward(prices: pd.DataFrame) -> list[dict]:
    """
    Roll 2yr-train / 1yr-test windows across the data.
    Returns one result dict per window.
    """
    years = sorted(prices.index.year.unique())
    results = []

    test_years = [y for y in years if y >= FIRST_TEST and y + TEST_YRS - 1 <= years[-1]]

    for test_yr in test_years:
        train_start = f"{test_yr - TRAIN_YRS}-01-01"
        train_end   = f"{test_yr}-01-01"
        test_start  = f"{test_yr}-01-01"
        test_end    = f"{test_yr + TEST_YRS}-01-01"

        train = prices[(prices.index >= train_start) & (prices.index < train_end)]
        test  = prices[(prices.index >= test_start)  & (prices.index < test_end)]

        if len(train) < 100 or len(test) < 50:
            continue

        print(f"  Window {test_yr}:  train {train_start}–{train_end}  "
              f"({len(train)} bars)  |  test {test_start}–{test_end}  ({len(test)} bars)")

        # Optimise on train
        best_params = _grid_search(train)
        print(f"    Best params: {best_params}")

        # Evaluate on test (OOS)
        oos_cfg = {**BASE_CFG, **best_params, "initial_capital": INIT_CAP}
        oos_eq  = _run_strategy_silent(test, oos_cfg)
        bah_eq  = _bah_equity(test["Close"], INIT_CAP)

        strat_m = _metrics(oos_eq if len(oos_eq) >= 10 else
                           pd.Series([INIT_CAP] * len(test), index=test.index), INIT_CAP)
        bah_m   = _metrics(bah_eq, INIT_CAP)

        results.append({
            "test_year":    test_yr,
            "train_period": f"{train_start} → {train_end}",
            "test_period":  f"{test_start} → {test_end}",
            "params":       best_params,
            "strat_sharpe": strat_m["sharpe"],
            "bah_sharpe":   bah_m["sharpe"],
            "strat_cagr":   strat_m["cagr"],
            "bah_cagr":     bah_m["cagr"],
            "strat_max_dd": strat_m["max_dd"],
            "bah_max_dd":   bah_m["max_dd"],
            "strat_sortino":strat_m["sortino"],
            "bah_sortino":  bah_m["sortino"],
        })
        print(f"    OOS Sharpe  strategy={strat_m['sharpe']:+.3f}  "
              f"B&H={bah_m['sharpe']:+.3f}")

    return results


# ── Results table ──────────────────────────────────────────────────────────────

def print_results(results: list[dict]) -> None:
    if not results:
        print("\n  No walk-forward windows completed.")
        return

    # Per-window table
    w = 12
    sep = "─" * 100
    hdr = (f"  {'Year':>{w}} {'Train period':>24} {'Test period':>24}"
           f" {'Sharpe':>8} {'B&H Sh':>8} {'CAGR':>8} {'B&H CAGR':>9} {'MaxDD':>8}")
    print(f"\n{sep}")
    print(hdr)
    print(sep)
    for r in results:
        strat_sh = r["strat_sharpe"]
        bah_sh   = r["bah_sharpe"]
        beat     = "✓" if (np.isfinite(strat_sh) and np.isfinite(bah_sh) and strat_sh >= bah_sh) else "✗"
        print(f"  {r['test_year']:>{w}} {r['train_period']:>24} {r['test_period']:>24}"
              f" {strat_sh:>7.3f}  {bah_sh:>7.3f}"
              f" {r['strat_cagr']:>+7.1%} {r['bah_cagr']:>+8.1%}"
              f" {r['strat_max_dd']:>7.1%}  {beat}")
    print(sep)

    # Aggregate stats
    strat_sharpes = [r["strat_sharpe"] for r in results if np.isfinite(r["strat_sharpe"])]
    bah_sharpes   = [r["bah_sharpe"]   for r in results if np.isfinite(r["bah_sharpe"])]

    if not strat_sharpes:
        print("\n  No finite Sharpe values to aggregate.")
        return

    med_s = float(np.median(strat_sharpes))
    avg_s = float(np.mean(strat_sharpes))
    std_s = float(np.std(strat_sharpes))
    med_b = float(np.median(bah_sharpes))

    print(f"\n  OOS Sharpe across {len(strat_sharpes)} windows:")
    print(f"    Strategy  — median {med_s:+.3f}   mean {avg_s:+.3f}   std {std_s:.3f}")
    print(f"    Buy-hold  — median {med_b:+.3f}")
    diff = med_s - med_b
    print(f"    Difference (strategy − B&H): {diff:+.3f}")

    # ── Section 5 acceptance criterion ────────────────────────────────────────
    EDGE_THRESHOLD = 0.10
    print()
    if diff >= EDGE_THRESHOLD:
        print(f"  ✓  PASSES: Median OOS Sharpe exceeds B&H by {diff:+.3f} "
              f"(threshold ≥ {EDGE_THRESHOLD:.2f}).")
        print(f"     The strategy demonstrates edge over buy-and-hold on real data.")
    else:
        print(f"  ✗  FAILS: Median OOS Sharpe ({med_s:+.3f}) does NOT exceed "
              f"B&H ({med_b:+.3f}) by the required {EDGE_THRESHOLD:.2f}.")
        print()
        print("  Likely reasons the strategy fails to beat buy-and-hold:")
        if avg_s < 0.3:
            print("   • Low absolute Sharpe — the signal itself may be noise on daily bars.")
            print("     VP levels computed from daily OHLCV are very coarse; intraday data")
            print("     would produce more meaningful value areas.")
        if med_b > 0.8:
            print("   • Strong structural bull trend in QQQ makes B&H hard to beat:")
            print("     a strategy that is out of the market most of the time (low trade count)")
            print("     systematically underperforms a buy-and-hold in a one-directional trend.")
        costs_note = any(r["strat_cagr"] < r["bah_cagr"] for r in results)
        if costs_note:
            print("   • Transaction costs eat into CAGR on most windows.")
            print("     With 0.15% stop slippage and partial exits, round-trip costs add up")
            print("     quickly when win rate is below ~40%.")
        n_windows     = len(results)
        beats         = sum(1 for r in results
                            if np.isfinite(r["strat_sharpe"]) and np.isfinite(r["bah_sharpe"])
                            and r["strat_sharpe"] >= r["bah_sharpe"])
        print(f"   • Strategy beats B&H in only {beats}/{n_windows} windows.")
        print()
        print("  Recommendation: do NOT tune parameters further to hide this result.")
        print("  Use intraday (15-min or 1-hr) data and session-based VP windows.")


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Walk-forward evaluation for VAH→VAL flip long")
    p.add_argument("--data", required=True,
                   help="Path to a QQQ (or similar) daily OHLCV CSV with columns "
                        "Open,High,Low,Close,Volume and a Date index.")
    args = p.parse_args()

    # ── Load and validate CSV ──────────────────────────────────────────────────
    try:
        prices = pd.read_csv(args.data, index_col=0, parse_dates=True)
    except FileNotFoundError:
        print(f"\n  ERROR: file not found: {args.data}")
        print(  "  Download real data first:")
        print( '    python -c "import yfinance as yf; '
               'yf.download(\'QQQ\', start=\'2010-01-01\', end=\'2025-01-01\','
               " auto_adjust=True).to_csv('qqq_daily.csv')\"")
        sys.exit(1)

    if "Adj Close" in prices.columns and "Close" not in prices.columns:
        prices = prices.rename(columns={"Adj Close": "Close"})

    required = {"Open", "High", "Low", "Close", "Volume"}
    missing  = required - set(prices.columns)
    if missing:
        print(f"\n  ERROR: CSV is missing columns: {missing}")
        sys.exit(1)

    prices = prices[list(required)].dropna()

    # Require real data — refuse to run on 1-bar synthetic or obviously fake data
    price_range = prices["Close"].max() / prices["Close"].min()
    if len(prices) < 200 or price_range < 1.01:
        print("\n  ERROR: Data looks synthetic or too short for a meaningful walk-forward.")
        print("  Sections 2–5 require real QQQ daily data (2010–2025).")
        print("  Download with yfinance and pass via --data.")
        sys.exit(1)

    print("\nWalk-Forward Evaluation — VAH→VAL Flip Long Strategy")
    print("=" * 60)
    print(f"  Data file    : {args.data}")
    print(f"  Bars loaded  : {len(prices)}")
    print(f"  Date range   : {prices.index[0].date()} → {prices.index[-1].date()}")
    print(f"  Train window : {TRAIN_YRS} years")
    print(f"  Test window  : {TEST_YRS} year")
    print(f"  Grid combos  : {len(list(itertools.product(*GRID.values())))}")
    print(f"  Init capital : ${INIT_CAP:,.0f}")
    n_windows = sum(1 for y in prices.index.year.unique()
                    if y >= FIRST_TEST and y + TEST_YRS - 1 <= prices.index.year.max())
    print(f"  Test windows : ~{n_windows}")
    print()

    results = walk_forward(prices)
    print_results(results)


if __name__ == "__main__":
    main()
