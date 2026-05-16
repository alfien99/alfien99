"""
Unit tests for vah_val_flip_long.py — Section 3 requirement.

Three tests:
  1. compute_vp: POC lands at the highest-volume bin on a hand-constructed input.
  2. flip detection: known flip case and known non-flip case via compute_levels().
  3. position sizing: risk_pct respected when stop is hit.

Run with:
    pytest tests/test_vah_flip.py -v
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import pytest

from vah_val_flip_long import (
    compute_vp,
    compute_levels,
    generate_signals,
    simulate_trades,
    DEFAULT_CONFIG,
    buy_lim,
    sell_stop,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_ohlcv(highs, lows, closes=None, opens=None, volumes=None, freq="D"):
    """Build a minimal OHLCV DataFrame from lists."""
    n = len(highs)
    if closes is None:
        closes = [(h + l) / 2 for h, l in zip(highs, lows)]
    if opens is None:
        opens = closes
    if volumes is None:
        volumes = [1_000_000] * n
    dates = pd.date_range("2024-01-01", periods=n, freq=freq)
    return pd.DataFrame({
        "Open":   opens,
        "High":   highs,
        "Low":    lows,
        "Close":  closes,
        "Volume": volumes,
    }, index=dates)


# ── Test 1: POC at highest-volume bin ─────────────────────────────────────────

def test_poc_at_highest_volume_bin():
    """
    Construct bars where most volume is concentrated in a narrow band (104-106)
    and a little volume is spread over the full 100-110 range.  POC must land
    inside the 104-106 band.
    """
    # 20 wide bars with low volume — distribute over 100-110
    wide_highs   = [110.0] * 20
    wide_lows    = [100.0] * 20
    wide_volumes = [500] * 20          # very low — will not dominate

    # 10 narrow bars with heavy volume — concentrated at 104-106
    narrow_highs   = [106.0] * 10
    narrow_lows    = [104.0] * 10
    narrow_volumes = [50_000] * 10     # 100× more volume than wide bars

    df = _make_ohlcv(
        highs   = wide_highs   + narrow_highs,
        lows    = wide_lows    + narrow_lows,
        volumes = wide_volumes + narrow_volumes,
    )

    poc, val, vah, mids, vol = compute_vp(df, bins=100)

    assert 103.5 <= poc <= 106.5, (
        f"POC {poc:.2f} should be in the high-volume 104-106 band, not outside it"
    )
    assert val <= poc, f"VAL ({val:.2f}) must be ≤ POC ({poc:.2f})"
    assert poc <= vah, f"POC ({poc:.2f}) must be ≤ VAH ({vah:.2f})"

    # The highest-volume bin index should equal the poc_idx used internally
    poc_bin_idx = int(np.argmax(vol))
    assert abs(mids[poc_bin_idx] - poc) < 0.5, (
        f"POC price {poc:.2f} should match highest-volume bin mid {mids[poc_bin_idx]:.2f}"
    )


# ── Test 2: flip detection — known flip and non-flip ──────────────────────────

def _cfg_for_flip_test():
    cfg = DEFAULT_CONFIG.copy()
    cfg.update({
        "lookback":        5,    # short lookback so the test data is small
        "vp_bins":         50,
        "flip_tolerance":  0.05,
        "ema_window":      10,
        "entry_buffer":    0.005,
        "stop_pct":        0.015,
        "ext_mult":        1.0,
        "initial_capital": 10_000,
        "risk_pct":        0.02,
        "max_position_pct": 0.40,
        "poc_exit_frac":   0.50,
        "vah_exit_frac":   0.70,
    })
    return cfg


def test_flip_detected_when_value_area_steps_up():
    """
    Construct a dataset where a clear value-area step-up falls exactly inside
    the two rolling windows at the first valid bar.

    With lookback=5 and ema_window=10:
      min_i = max(5*2, 10+5) = 15

    At bar 15:
      current  window → raw[10:15]  (bars 10-14, high regime 108-118)
      previous window → raw[ 5:10]  (bars  5-9,  low  regime 100-110)

    VAL(high regime) ≈ 109 ≥ VAH(low regime) ≈ 109 → flip = True.

    Bars 0-4 are preamble so the EMA has data; they use the low-regime range.
    """
    cfg = _cfg_for_flip_test()

    # Preamble bars 0-4: low regime (100-110), feeds EMA warm-up
    preamble_highs  = [110.0] * 5
    preamble_lows   = [100.0] * 5
    preamble_closes = [105.0] * 5

    # Bars 5-9 → "previous" window at bar 15: low regime 100-110
    prev_highs  = [110.0] * 5
    prev_lows   = [100.0] * 5
    prev_closes = [105.0] * 5

    # Bars 10-14 → "current" window at bar 15: high regime 108-118
    curr_highs  = [118.0] * 5
    curr_lows   = [108.0] * 5
    curr_closes = [113.0] * 5

    # Bars 15-24: stay in high regime so the flip can fire with uptrend intact
    pad_highs  = [118.0] * 10
    pad_lows   = [108.0] * 10
    pad_closes = [113.0] * 10

    df = _make_ohlcv(
        highs  = preamble_highs  + prev_highs  + curr_highs  + pad_highs,
        lows   = preamble_lows   + prev_lows   + curr_lows   + pad_lows,
        closes = preamble_closes + prev_closes + curr_closes + pad_closes,
    )

    levels  = compute_levels(df, cfg)
    signals = generate_signals(df, levels, cfg)

    assert levels["flip"].any(), (
        "Expected at least one flip signal when value area steps from 100-110 (bars 5-9) "
        "to 108-118 (bars 10-14).  VAL(current) should be near/above VAH(previous)."
    )


def test_no_flip_when_value_area_stays_flat():
    """
    Construct a dataset where price stays in the same 100-110 range throughout.
    VAL(current) ≈ VAL(previous), so no flip should be detected.
    """
    cfg = _cfg_for_flip_test()

    # Same range for all bars — no step-up
    n = 40
    df = _make_ohlcv(
        highs  = [110.0] * n,
        lows   = [100.0] * n,
        closes = [105.0] * n,
    )

    levels = compute_levels(df, cfg)

    assert not levels["flip"].any(), (
        "Expected zero flip signals when price stays in the same 100-110 range"
    )


# ── Test 3: position sizing respects risk_pct when stop is hit ────────────────

def test_risk_pct_respected_on_stop():
    """
    Construct a dataset that forces exactly one entry then a stop hit.
    Verify the realized loss equals initial_capital × risk_pct.

    Dataset layout (lookback=5, ema_window=10 → min_i=15):
      bars  0- 4: preamble low regime 100-110
      bars  5- 9: "previous" window low regime 100-110
      bars 10-14: "current"  window high regime 108-118  → flip fires at bar 15
      bar  15:    entry bar — low=108 touches entry_limit ≈ 109.5
      bar  16:    stop  bar — low=100, far below stop ≈ 107  → stop hit
      bars 17-24: continuation

    With risk_pct=0.02 and initial_capital=10_000, the expected loss is
    exactly (capital × risk_pct) = $200, because:
        n_sh = risk_cash / risk_per_sh
        PnL  = (sell_stop(stop) - fill_px) × n_sh = -risk_per_sh × n_sh = -$200
    """
    cfg = _cfg_for_flip_test()
    cfg["risk_pct"]  = 0.02   # $200 expected loss on $10,000
    cfg["stop_pct"]  = 0.02   # stop 2% below VAL

    # Bars 0-4: EMA preamble
    pre_highs  = [110.0] * 5;  pre_lows  = [100.0] * 5;  pre_cls = [105.0] * 5
    # Bars 5-9: previous VP window (low regime)
    pr_highs   = [110.0] * 5;  pr_lows   = [100.0] * 5;  pr_cls  = [105.0] * 5
    # Bars 10-14: current VP window (high regime → triggers flip at bar 15)
    cu_highs   = [118.0] * 5;  cu_lows   = [108.0] * 5;  cu_cls  = [113.0] * 5
    # Bar 15: entry bar — low touches VAL zone (≈108-109)
    e_high = [113.0]; e_low = [108.0]; e_cls = [113.0]
    # Bar 16: stop bar — low crashes well below stop level (stop ≈ 107)
    s_high = [112.0]; s_low = [100.0]; s_cls = [101.0]
    # Bars 17-24: continuation
    pa_highs = [118.0] * 8; pa_lows = [108.0] * 8; pa_cls = [113.0] * 8

    df = _make_ohlcv(
        highs  = pre_highs  + pr_highs  + cu_highs  + e_high + s_high + pa_highs,
        lows   = pre_lows   + pr_lows   + cu_lows   + e_low  + s_low  + pa_lows,
        closes = pre_cls    + pr_cls    + cu_cls    + e_cls  + s_cls  + pa_cls,
    )

    levels  = compute_levels(df, cfg)
    signals = generate_signals(df, levels, cfg)
    trades_df, equity_df, _, _, _ = simulate_trades(df, signals, levels, cfg)

    stop_trades = trades_df[trades_df["type"] == "stop"]
    assert not stop_trades.empty, (
        "Expected at least one stop exit.  Check that bar 16's low (100) "
        "is below the stop level (~107) and that an entry fired at bar 15."
    )

    initial      = cfg["initial_capital"]
    risk_target  = initial * cfg["risk_pct"]   # $200

    # Each stop trade should lose ≈ risk_target.
    # PnL = (sell_stop(stop_level) - fill_px) × n_sh = -risk_per_sh × n_sh = -risk_cash
    # So loss = initial_capital × risk_pct exactly.
    # Allow ±10% for rounding in price bin computation.
    for _, trade in stop_trades.iterrows():
        actual_loss = abs(trade["pnl"])
        assert actual_loss <= risk_target * 1.10, (
            f"Stop loss ${actual_loss:.2f} exceeds 1.1× risk target ${risk_target:.2f}. "
            f"Position sizing is not controlling risk correctly."
        )
        assert actual_loss >= risk_target * 0.90, (
            f"Stop loss ${actual_loss:.2f} is less than 0.9× risk target ${risk_target:.2f}. "
            f"Risk per trade is lower than configured."
        )
