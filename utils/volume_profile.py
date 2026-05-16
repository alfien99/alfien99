"""
Shared volume profile computation.

Drop-in replacement for the compute_vp() function that is duplicated
across volume_profile_strategy.py, vah_val_flip_long.py,
staircase_breakout_strategy.py, and nasdaq_long_chart.py.

Usage:
    from utils.volume_profile import compute_vp

    poc, val, vah = compute_vp(df.iloc[i-20:i])
"""

import numpy as np
from utils.constants import VALUE_AREA_PCT, VP_BINS


def compute_vp(df, bins: int = VP_BINS):
    """
    Build a volume profile for a slice of OHLCV data and return the
    Point of Control, Value Area Low, and Value Area High.

    Parameters
    ----------
    df   : DataFrame with columns Open, High, Low, Close, Volume
    bins : number of price buckets (default 100)

    Returns
    -------
    (poc, val, vah) : floats — price levels
    """
    df = df.dropna(subset=["High", "Low", "Close", "Volume"])

    hi = df["High"].max()
    lo = df["Low"].min()

    # Edge case: zero-range bar (e.g. opening auction, bad data)
    if hi <= lo + 1e-8:
        mid = (hi + lo) / 2
        return mid, mid, mid

    edges = np.linspace(lo, hi, bins + 1)
    mids  = (edges[:-1] + edges[1:]) / 2
    vol   = np.zeros(bins)

    for _, row in df.iterrows():
        bar_range = row["High"] - row["Low"]
        if bar_range < 1e-10:
            # Single-price bar — all volume goes to the one bin
            idx = min(int((row["Close"] - lo) / (hi - lo) * bins), bins - 1)
            vol[idx] += row["Volume"]
            continue

        # Distribute volume proportionally across the bins the bar spans
        low_idx  = max(np.searchsorted(edges, row["Low"],  side="left"),  0)
        high_idx = min(np.searchsorted(edges, row["High"], side="right") - 1, bins - 1)
        for k in range(low_idx, high_idx + 1):
            overlap  = min(edges[k + 1], row["High"]) - max(edges[k], row["Low"])
            vol[k]  += row["Volume"] * overlap / bar_range

    # Point of Control = highest-volume bin
    poc_idx = int(np.argmax(vol))

    # Expand the value area outward from POC until VALUE_AREA_PCT of volume is captured
    target  = vol.sum() * VALUE_AREA_PCT
    low_idx = high_idx = poc_idx
    running = vol[poc_idx]

    while running < target and (low_idx > 0 or high_idx < bins - 1):
        add_low  = vol[low_idx  - 1] if low_idx  > 0       else -1
        add_high = vol[high_idx + 1] if high_idx < bins - 1 else -1
        if add_low >= add_high:
            low_idx  -= 1
            running  += vol[low_idx]
        else:
            high_idx += 1
            running  += vol[high_idx]

    poc = mids[poc_idx]
    val = mids[low_idx]
    vah = mids[high_idx]
    return poc, val, vah
