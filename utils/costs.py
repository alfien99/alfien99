"""
Transaction cost helpers — shared across all strategy files.

Usage:
    from utils.costs import buy_mkt, sell_mkt, buy_lim, sell_lim, trade_cost

These functions return the *effective fill price* after commission and
slippage have been applied, so position sizing and P&L calculations
automatically account for friction.
"""

from utils.constants import COMMISSION, SLIP_MKT, SLIP_LIM


def buy_mkt(price: float) -> float:
    """Market buy: price rises by slippage + commission."""
    return price * (1 + SLIP_MKT + COMMISSION)


def sell_mkt(price: float) -> float:
    """Market sell: price falls by slippage + commission."""
    return price * (1 - SLIP_MKT - COMMISSION)


def buy_lim(price: float) -> float:
    """Limit buy: tighter slippage (order was passive), plus commission."""
    return price * (1 + SLIP_LIM + COMMISSION)


def sell_lim(price: float) -> float:
    """Limit sell: tighter slippage, plus commission."""
    return price * (1 - SLIP_LIM - COMMISSION)


def trade_cost(raw_price: float, shares: float, is_limit: bool = False) -> float:
    """
    Total transaction cost in dollars for one fill.

    Parameters
    ----------
    raw_price : mid-market price before friction
    shares    : number of shares / contracts
    is_limit  : True for limit orders (lower slippage), False for market orders
    """
    slip = SLIP_LIM if is_limit else SLIP_MKT
    return raw_price * shares * (slip + COMMISSION)
