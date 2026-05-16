"""
Shared three-tranche exit handler for all strategy files.

The pattern — close a fraction at POC, another fraction of the remainder
at VAH, and the rest at an extension target — appears in every strategy.
This module centralises it so the logic only needs to be maintained once.

Usage:
    from utils.exits import ExitState, handle_exits

    state = ExitState(shares=100, entry_cost=5000.0)
    fills = handle_exits(state, high=bar_high, poc=poc, vah=vah, ext=ext)
    for fill in fills:
        capital += fill.proceeds
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List
from utils.costs import sell_lim
from utils.constants import POC_EXIT_FRAC, VAH_EXIT_FRAC


@dataclass
class ExitFill:
    """One partial exit that fired on this bar."""
    level: str       # "poc", "vah", or "ext"
    shares: float    # shares sold
    price: float     # raw (pre-cost) exit price
    proceeds: float  # net proceeds after sell_lim() friction


@dataclass
class ExitState:
    """Mutable position state passed into handle_exits() each bar."""
    shares: float
    entry_cost: float        # total dollars paid at entry (for P&L %)
    poc_done: bool = False
    vah_done: bool = False
    ext_done: bool = False
    total_proceeds: float = 0.0
    fills: List[ExitFill] = field(default_factory=list)

    @property
    def closed(self) -> bool:
        """True when all shares have been exited."""
        return self.shares <= 0


def handle_exits(
    state: ExitState,
    high: float,
    poc: float,
    vah: float,
    ext: float,
    poc_frac: float = POC_EXIT_FRAC,
    vah_frac: float = VAH_EXIT_FRAC,
) -> List[ExitFill]:
    """
    Check all three exit levels against the current bar's high and fire
    whichever are triggered. Multiple levels can fire on the same bar
    (e.g. a gap-up that blows straight past VAH).

    Parameters
    ----------
    state    : mutable ExitState — updated in place
    high     : current bar's high price
    poc, vah : Volume Profile levels
    ext      : extension target (e.g. VAH + 1.0 × (VAH - POC))
    poc_frac : fraction of position to close at POC (default 0.50)
    vah_frac : fraction of *remainder* to close at VAH (default 0.70)

    Returns
    -------
    List of ExitFill objects for fills that fired this bar (empty if none).
    """
    new_fills: List[ExitFill] = []

    # ── Level 1: POC ─────────────────────────────────────────────────────────
    if not state.poc_done and high >= poc:
        close_shares = round(state.shares * poc_frac, 6)
        if close_shares > 0:
            raw_px   = poc
            proceeds = sell_lim(raw_px) * close_shares
            state.shares         -= close_shares
            state.total_proceeds += proceeds
            state.poc_done        = True
            fill = ExitFill(level="poc", shares=close_shares, price=raw_px, proceeds=proceeds)
            new_fills.append(fill)
            state.fills.append(fill)

    # ── Level 2: VAH ─────────────────────────────────────────────────────────
    if state.poc_done and not state.vah_done and high >= vah:
        close_shares = round(state.shares * vah_frac, 6)
        if close_shares > 0:
            raw_px   = vah
            proceeds = sell_lim(raw_px) * close_shares
            state.shares         -= close_shares
            state.total_proceeds += proceeds
            state.vah_done        = True
            fill = ExitFill(level="vah", shares=close_shares, price=raw_px, proceeds=proceeds)
            new_fills.append(fill)
            state.fills.append(fill)

    # ── Level 3: Extension ───────────────────────────────────────────────────
    if state.vah_done and state.shares > 0 and high >= ext:
        close_shares = state.shares  # close everything remaining
        raw_px   = ext
        proceeds = sell_lim(raw_px) * close_shares
        state.shares         -= close_shares
        state.total_proceeds += proceeds
        state.ext_done        = True
        fill = ExitFill(level="ext", shares=close_shares, price=raw_px, proceeds=proceeds)
        new_fills.append(fill)
        state.fills.append(fill)

    return new_fills
