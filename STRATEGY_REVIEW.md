# VAL/VAH Volume Profile Strategy — Review Request

## Context

This repo contains three Python backtester scripts built around Market Profile / Volume Profile levels (VAL, POC, VAH) on a US 100 (Nasdaq) 4H chart showing a clear staircase accumulation uptrend.

## Chart Patterns Observed

1. **VAH→VAL flip** — Prior period's VAH becomes next period's VAL as price steps up. Old resistance becomes new support. This is the dominant, highest-probability setup.
2. **POC magnetism** — Price gravitates back to POC mid-consolidation before next leg.
3. **VAL holds on 1-2 retests** — Multiple touches without breaking = accumulation.
4. **Volume expansion on VAH breaks** — High volume on breakout bars, low volume on pullbacks.
5. **Staircase structure** — Price consolidates → breaks VAH → consolidates at higher level → repeats.

## Three Scripts

### 1. `volume_profile_strategy.py` — Bidirectional VAL/VAH
- **Long**: enter below VAL, scale in every 1% lower (up to 5 tranches)
- **Short**: enter above VAH, scale in every 1% higher (up to 5 tranches)
- Exits: 50% at POC → 70% of remainder at VAH/VAL → rest at extension
- Stop: 3% beyond entry level
- `--no-short` flag for long-only mode

### 2. `vah_val_flip_long.py` — VAH→VAL Flip Long Only
- Detects when current VAL ≥ previous VAL AND near/above prior VAH (value area stepped up)
- Entry via **limit order at VAL** (triggers when bar's low touches VAL zone)
- Trend filter: price above EMA(50)
- Risk-sized: 2% capital at risk per trade, capped at 40% position size
- Exits: 50% at POC → 70% of remainder at VAH → rest at VAH + 1× extension
- Stop: 1.5% below VAL

### 3. `staircase_breakout_strategy.py` — Best Strategy (Breakout + Pullback)
- **Phase A Breakout**: close > VAH for 2 confirmed bars + optional volume surge → enter at close, ATR trailing stop
- **Phase B Pullback**: after any VAH break, limit order at VAL zone → tighter stop, 1.5× risk size
- EMA(200) trend filter
- Three-tranche exits: 40% at POC → 60% of remainder at VAH → rest at 1.2× extension

## Key Parameters (shared across scripts)

| Param | Value | Notes |
|---|---|---|
| `lookback` | 20 bars | Rolling VP window |
| `vp_bins` | 100 | Price resolution |
| `stop_pct` | 1.5–3% | Below VAL / above VAH |
| `risk_pct` | 2% | Capital at risk per trade |
| `max_position_pct` | 40% | Max capital deployed per trade |
| `poc_exit_frac` | 50% | Partial exit at POC |
| `vah_exit_frac` | 70% | Of remainder at VAH |
| `ext_mult` | 1.0–1.2× | Final target extension beyond VAH |
| `flip_tolerance` | 3% | How close VAL(t) must be to VAH(t-1) |
| `ema_window` | 50 / 200 | Trend filter |

## Known Issues / Areas to Improve

1. **Synthetic data bias**: yfinance is blocked in this env, so all backtests use geometric Brownian motion (no structural trend). Performance on real data will differ significantly — strategies need a trending instrument to show edge.

2. **Flip detection**: Currently uses two non-overlapping rolling VP windows (bars `[i-40:i-20]` and `[i-20:i]`). This is crude — a session-based or fixed-period VP (daily/weekly) would be more realistic.

3. **Entry timing**: Limit orders at VAL are modelled by checking if `low <= VAL`. In reality on a daily bar this is very coarse — intraday data or a more refined entry model would improve fill accuracy.

4. **No long/short conflict guard**: The bidirectional strategy can technically hold both a long and short at the same time if conditions overlap. A mutual exclusion check is missing.

5. **Scale-in averaging down**: The VAL long scale-in strategy averages down, which is high-risk in trending bear markets. A market-regime filter (e.g. 200-day EMA or ADX) would reduce this exposure.

6. **Breakout confirmation**: Only uses bar count (N closes above VAH). Could add: RSI > 50, ATR expansion, or VWAP confirmation.

7. **Exit logic is sequential**: POC must be hit before VAH, VAH before extension. If price gaps straight to VAH, the POC exit is missed. Consider checking all levels on the same bar.

8. **No slippage/commission model**: All fills assume zero cost. Real-world friction will reduce returns, especially on the scale-in tranches.

9. **Fixed lookback**: A 20-bar rolling window doesn't adapt to volatility regimes. An ATR-normalised or volatility-adaptive lookback could improve VP quality.

10. **Pullback phase in staircase strategy**: Currently any pullback to VAL after any VAH break qualifies — no recency check. A stale `last_vah` from many bars ago shouldn't still trigger entries.

## Running the Scripts

```bash
pip install yfinance matplotlib pandas numpy

# Long only flip strategy
python vah_val_flip_long.py --ticker QQQ --start 2021-01-01 --end 2025-01-01

# Best strategy
python staircase_breakout_strategy.py --ticker QQQ --start 2021-01-01 --end 2025-01-01

# Bidirectional
python volume_profile_strategy.py --ticker SPY --start 2022-01-01 --end 2025-01-01 --no-prompt
```

## Request

Please review all three scripts and suggest concrete improvements to:
- Strategy logic (entry/exit rules, filters)
- Risk management (position sizing, stop placement)
- Code quality and correctness
- Any bugs or edge cases
