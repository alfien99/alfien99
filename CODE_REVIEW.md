# Code Review — VAL/VAH Volume Profile Strategy Suite

**Files reviewed:** `volume_profile_strategy.py` (809 lines) · `vah_val_flip_long.py` (445 lines) · `staircase_breakout_strategy.py` (523 lines) · `nasdaq_long_chart.py` (821 lines) · `walk_forward_eval.py` (700 lines)

---

## Executive Summary

The suite is **functionally correct** — the core strategy logic, transaction cost model, walk-forward framework, and Plotly chart all work as intended. The primary problems are organisational: the same code is copied across four files, configuration is inconsistent between strategies, and several pieces of logic rely on undocumented magic numbers that will be painful to tune later.

**Top 5 issues by impact:**

| # | Issue | Why it matters |
|---|-------|----------------|
| 1 | `compute_vp()` copied identically into 4 files | A bug fix or improvement must be applied 4× |
| 2 | Magic numbers for value area %, exit fractions, stop % | Changing a threshold means hunting across all files |
| 3 | Redundant entry-condition OR in `volume_profile_strategy.py` | Both branches are identical; one is always dead code |
| 4 | Dead ADX filter in `walk_forward_eval.py` | Defined but never called; Stage 4 enhancement silently does nothing |
| 5 | Silent `except Exception: pass` in grid search | Parameter combos that crash are skipped with no warning |

**Quick wins (< 30 min each):** items 2, 3, 4, 5 above.  
**Larger refactors:** shared utility module (item 1), position state dataclass, shared exit helper.

---

## Section 1 — Cross-File Issues

These affect multiple files and have the highest leverage to fix.

### 1.1 Duplicated `compute_vp()` — 4 identical copies

`volume_profile_strategy.py`, `vah_val_flip_long.py`, `staircase_breakout_strategy.py`, and `nasdaq_long_chart.py` all contain the same ~30-line volume profile function. Any bug fix (e.g., the zero-range bar edge case described in §3) must be applied four times.

**Fix:** Extract to `utils/volume_profile.py` and `from utils.volume_profile import compute_vp` in each file.

```python
# utils/volume_profile.py
VALUE_AREA_PCT = 0.70   # 70% of volume defines the value area

def compute_vp(df, bins=100):
    """Return (poc, val, vah) for the given OHLCV slice."""
    hi, lo = df["High"].max(), df["Low"].min()
    if hi <= lo + 1e-8:
        mid = (hi + lo) / 2
        return mid, mid, mid
    edges = np.linspace(lo, hi, bins + 1)
    mids  = (edges[:-1] + edges[1:]) / 2
    vol   = np.zeros(bins)
    for _, row in df.iterrows():
        rng = row["High"] - row["Low"]
        if rng < 1e-10:
            idx = min(int((row["Close"] - lo) / (hi - lo) * bins), bins - 1)
            vol[idx] += row["Volume"]
            continue
        lo_i = np.searchsorted(edges, row["Low"],  side="left")
        hi_i = np.searchsorted(edges, row["High"], side="right") - 1
        for k in range(lo_i, hi_i + 1):
            ov   = min(edges[k+1], row["High"]) - max(edges[k], row["Low"])
            vol[k] += row["Volume"] * ov / rng
    poc_idx = int(np.argmax(vol))
    target  = vol.sum() * VALUE_AREA_PCT
    lo_i = hi_i = poc_idx
    running = vol[poc_idx]
    while running < target and (lo_i > 0 or hi_i < bins - 1):
        add_lo = vol[lo_i - 1] if lo_i > 0 else -1
        add_hi = vol[hi_i + 1] if hi_i < bins - 1 else -1
        if add_lo >= add_hi:
            lo_i -= 1; running += vol[lo_i]
        else:
            hi_i += 1; running += vol[hi_i]
    return mids[poc_idx], mids[lo_i], mids[hi_i]
```

---

### 1.2 Duplicated data loading — 3 copies

`_download()` and `_synthetic_ohlcv()` appear identically in files 1, 2, and 3. File 4 extends them with timeframe resampling. Any change to the synthetic data generator (e.g., adding a structural trend for more realistic testing) must be replicated three times.

**Fix:** Extract to `utils/data_loader.py`. File 4's `load_data()` can call the shared version and add resampling on top.

---

### 1.3 Duplicated exit logic — POC → VAH → extension pattern

Every strategy file implements the same three-tranche exit sequence. The implementations differ only in the fraction constants and whether they use an `elif` chain or independent `if` blocks:

| File | Exit style | POC frac | VAH frac |
|------|-----------|---------|---------|
| `volume_profile_strategy.py` | `elif` chain | 0.50 | 0.70 |
| `vah_val_flip_long.py` | `elif` chain | 0.50 | 0.70 |
| `staircase_breakout_strategy.py` | independent `if` | 0.40 | 0.60 |
| `nasdaq_long_chart.py` | independent `if` | 0.50 | 0.70 |
| `walk_forward_eval.py` | independent `if` | 0.50 | 0.70 |

The `elif` chain means only one tranche can fire per bar. The independent `if` style allows multiple tranches in the same bar (correct for gap-up moves). These two behaviours should be a conscious choice, not an accident of copy-pasting.

**Fix:** Extract `_handle_exits(pos, high, poc, vah, ext, fracs)` to the shared module and use independent `if` blocks consistently (the walk-forward version is correct).

---

### 1.4 Magic numbers for core fractions

The numbers `0.70`, `0.50`, `0.40`, `0.60` appear across all files without explanation. A reader has no way to know if they are meant to be the same constant or accidentally similar values.

**Fix:** Define named constants once, import everywhere.

```python
VALUE_AREA_PCT  = 0.70   # portion of volume that defines the value area
POC_EXIT_FRAC   = 0.50   # close half the position at POC
VAH_EXIT_FRAC   = 0.70   # close 70% of remainder at VAH
STOP_PCT        = 0.015  # default stop 1.5% below VAL
RISK_PCT        = 0.02   # risk 2% of capital per trade
```

---

### 1.5 No shared configuration

Each file has its own `DEFAULT_CONFIG` dict. `stop_pct` is 0.015 in files 2, 3, 4 but 0.03 in file 1. There is no way to know which value is intentional without reading the comments. When walk-forward imports strategy logic, it creates its own base config that can silently differ.

**Fix:** A single `DEFAULT_CONFIG` in the shared module, with each strategy overriding only the parameters that differ.

---

## Section 2 — `volume_profile_strategy.py`

### 2.1 Redundant entry condition (bug)

Lines ~297–298:
```python
if close < val and not long_stopped and not long_pos["in_pos"] or \
   (close < val and not long_stopped and long_pos["in_pos"]):
```
Both branches share the same guard (`close < val and not long_stopped`). The only difference is `not in_pos` vs. `in_pos`, so this collapses to `close < val and not long_stopped` regardless of position state. The scale-in logic that follows uses `long_pos["in_pos"]` to branch, making the outer condition redundant.

**Fix:**
```python
if close < val and not long_stopped:
```

### 2.2 Position state dict is hard to follow

A 14-key dict (`long_pos = {"in_pos": False, "shares": 0, ...}`) is used for both long and short sides. Exit conditions reference keys like `long_pos["poc_done"]` scattered across ~60 lines. A Python `dataclass` would give named fields with type hints and be easier to follow.

### 2.3 Exit logic duplicated within the file

The long-exit block and short-exit block follow identical patterns (check level → calculate shares → update capital → append marker). These could share a helper, halving the code in `run_backtest()`.

### 2.4 Unused return values from `compute_vp()`

In the backtest loop, `compute_vp()` is called and only `(poc, val, vah)` are used. The function also returns `_mids, _vpvol` (midpoint array and volume array) which are only needed for the chart. The backtest loop could call a lighter version that skips computing those arrays.

### 2.5 Silent exception in `_download()`

```python
except Exception:
    print("yfinance failed — using synthetic data")
```
The exception type is swallowed. A network error and a missing-column error look identical in the output. Use `except Exception as e: print(f"yfinance failed ({e}) — using synthetic data")`.

---

## Section 3 — `vah_val_flip_long.py`

### 3.1 Division by zero risk in flip detection

```python
near_prev_vah = (vah_p > 0 and abs(val_c - vah_p) / vah_p < tol) or val_c >= vah_p
```
The `vah_p > 0` guard prevents strict division by zero, but `vah_p` could be a very small positive number (synthetic data starting near zero), which would make the ratio arbitrarily large and the condition always False when it should be True. Add `vah_p > 1e-6` or normalise differently.

### 3.2 Limit order fill undocumented

```python
if low <= val_c * (1 + buf):
    fill_px = val_c * (1 + buf)
```
This models a limit order triggered when the bar's low touches the limit price. It is the correct approach, but without a comment a reader will assume it's a market order at a strange price. Add one line: `# limit order: filled when bar low touches the limit price`.

### 3.3 Warm-up formula unexplained

```python
min_i = max(lookback * 2, ema_win + 5)
```
Why `lookback * 2`? The VP requires two non-overlapping windows (current and previous), so `lookback * 2` is correct. Why `+ 5`? A comment would prevent confusion.

### 3.4 Inconsistent position state pattern

Files 1 uses a nested dict; files 2, 3, 4 use loose variables (`in_pos`, `shares`, `avg_px`). Pick one pattern and use it everywhere.

---

## Section 4 — `staircase_breakout_strategy.py`

### 4.1 Undocumented 1.5× pullback risk multiplier

```python
# Breakout
risk_cash = capital * cfg["risk_pct"]

# Pullback
risk_cash = capital * cfg["risk_pct"] * 1.5
```
Pullback entries receive 50% more risk capital than breakout entries. This may be intentional (pullbacks are higher-confidence entries) but there is no comment explaining it. If it's intentional, make it a named config parameter (`pullback_risk_mult`).

### 4.2 EMA 200 hardcoded and not configurable

```python
ema200 = raw["Close"].ewm(span=200, adjust=False).mean()
```
The span is not in `DEFAULT_CONFIG`. Any walk-forward optimisation that wants to test EMA 100 vs. EMA 200 cannot do so without editing the source.

### 4.3 Fragile string-based phase state machine

```python
phase = None  # or "breakout" or "pullback"
```
String comparisons (`if phase == "breakout"`) are error-prone. A `from enum import Enum` approach would catch typos at definition time:

```python
from enum import Enum, auto
class Phase(Enum):
    NONE      = auto()
    BREAKOUT  = auto()
    PULLBACK  = auto()
```

### 4.4 Volume surge filter reads backwards

```python
use_vol_filter = vol_surge or not cfg.get("require_vol_surge", True)
```
This is logically correct (`allow entry if volume surged OR if we don't require a surge`) but reads as if it's the wrong way round. Add a comment: `# True when entry is allowed: surge occurred, or surge check is disabled`.

---

## Section 5 — `nasdaq_long_chart.py`

### 5.1 Five separate TF lookup dicts that can get out of sync

```python
TF_YF_INTERVAL  = {"1H": "1h",   "2H": "1h",   "4H": "1h",   "1D": "1d",   "1W": "1wk"}
TF_RESAMPLE     = {"1H": None,   "2H": "2h",   "4H": "4h",   "1D": None,   "1W": None}
TF_MAX_DAYS     = {"1H": 729,    "2H": 729,    "4H": 729,    "1D": 3650,   "1W": 3650}
TF_DEFAULT_DAYS = {"1H": 180,    "2H": 270,    "4H": 365,    "1D": 730,    "1W": 1095}
TF_VP_BARS      = {"1H": 40,     "2H": 30,     "4H": 20,     "1D": 20,     "1W": 12}
```
Adding a new timeframe requires editing five separate dicts. A missing key in one dict causes a `KeyError` at runtime with no useful error message.

**Fix:**
```python
TF_CONFIG = {
    "1H": {"yf_interval": "1h",  "resample": None,  "max_days": 729,  "default_days": 180,  "vp_bars": 40},
    "2H": {"yf_interval": "1h",  "resample": "2h",  "max_days": 729,  "default_days": 270,  "vp_bars": 30},
    "4H": {"yf_interval": "1h",  "resample": "4h",  "max_days": 729,  "default_days": 365,  "vp_bars": 20},
    "1D": {"yf_interval": "1d",  "resample": None,  "max_days": 3650, "default_days": 730,  "vp_bars": 20},
    "1W": {"yf_interval": "1wk", "resample": None,  "max_days": 3650, "default_days": 1095, "vp_bars": 12},
}
```

### 5.2 Plotly chart block is 275 lines

The `plot_chart()` function builds every trace inline with repetitive styling. Consider a small `_add_hline(fig, y, color, name, row)` helper to reduce boilerplate on the 8+ horizontal level lines.

### 5.3 Lookback validation incomplete

The interactive prompt validates `lb >= 5` but doesn't check whether the lookback is sensible for the chosen timeframe and date range. A lookback of 200 on 1H data spanning only 180 days (~1080 bars) is fine, but 200 on 1W spanning 180 days (just 26 bars) would leave almost no usable bars after the warm-up period.

---

## Section 6 — `walk_forward_eval.py`

### 6.1 Dead code: ADX filter defined but never called

`_with_adx_filter()` and `_adx_series()` are defined (lines ~540–560) and referenced in a Stage 4 comment, but `stage4()` never calls them. The ADX enhancement silently does nothing. Either wire it up or delete it.

### 6.2 Silent exception swallowing in grid search

```python
except Exception:
    pass
```
A bad parameter combo (e.g., `stop_pct=0` producing infinite position size) will be silently skipped. The grid search will return whatever best combo avoided a crash, which may not be the true optimum. At minimum, log the exception:

```python
except Exception as e:
    if os.getenv("WF_DEBUG"):
        print(f"  [grid] {dict(zip(keys, combo))} raised {e}")
```

### 6.3 Unclear variable abbreviations

The following abbreviations appear throughout the backtest loop with no inline definition. A reader cannot understand the code without tracing every assignment:

| Name | Likely meaning | First assigned |
|------|----------------|----------------|
| `sh` | shares held | ~line 175 |
| `ecost` | entry cost (price × shares) | ~line 239 |
| `rps` | risk per share | ~line 235 |
| `li`, `hi_i` | low/high bin indices in VP | ~line 82 |
| `al`, `ah` | accumulated volume at low/high | ~line 84 |
| `ov` | overlap volume fraction | ~line 77 |
| `pb_raw` | pullback raw fill price | ~line 368 |

**Fix:** Rename to full words. Modern Python has no line-length excuse for 2-character variable names outside of `i`, `j`, `k` loop counters.

### 6.4 Turnover formula undocumented

```python
turnover = (total_costs / (SLIP_MKT + COMMISSION) * 2) / initial / n_years
```
Dividing by `SLIP_MKT + COMMISSION` to recover gross dollar volume is non-standard. Add a comment explaining the algebra:
```python
# gross_volume = total_costs / cost_per_dollar; annualised turnover = gross_volume / capital / years
```

### 6.5 Position state can go stale after partial exits

If the three partial exits (POC, VAH, extension) collectively reduce `sh` to 0, the `in_pos` flag remains `True` until the end-of-loop forced-close check. This means the strategy will attempt to compute a stop on a zero-share position. The guard `if in_pos and sh > 0` catches it at the final bar, but mid-backtest the state is inconsistent. Fix by resetting `in_pos = False` when `sh` drops to 0.

### 6.6 Zero-range bar breaks flip detection

In `compute_vp()`:
```python
if hi <= lo + 1e-8:
    mid = (hi + lo) / 2
    return mid, mid, mid
```
When POC = VAL = VAH, the flip condition `val_c > val_p` will never be True for two consecutive identical-range bars. This edge case is rare on real price data but common on synthetic data (consecutive equal OHLC bars). Log a warning or skip the bar.

---

## Section 7 — Risk & Correctness

### 7.1 No parameter validation

`stop_pct = 0` would produce infinite position size (`risk_cash / stop_per_share` → ∞). `max_position_pct = 0` would prevent any entry. Neither is validated. Add assertions at strategy entry:

```python
assert 0 < cfg["stop_pct"] < 0.20,  "stop_pct must be between 0 and 20%"
assert 0 < cfg["risk_pct"] < 0.10,  "risk_pct must be between 0 and 10%"
assert 0 < cfg["max_position_pct"] <= 1.0
```

### 7.2 No unit tests

There are no tests for any of the core functions. The following are the highest-value targets for a first test file:

- `compute_vp()` — test that POC is within [VAL, VAH], that value area contains ≥ 70% of volume, and that the zero-range edge case returns equal values
- `_compute_metrics()` — test with a known equity curve that CAGR, Sharpe, and max DD are correct
- Flip detection — test that a stepping value area triggers an entry and a non-stepping one does not

### 7.3 No guard against NaN propagation

yfinance sometimes returns `NaN` in Volume on the first bar. `compute_vp()` does not drop NaN rows before computing the volume-weighted profile. A single `df = df.dropna(subset=["Open","High","Low","Close","Volume"])` at the start of `compute_vp()` (or in the data loader) prevents silent NaN contamination.

---

## Section 8 — Prioritised Fix List

| Priority | Issue | Files | Effort |
|----------|-------|-------|--------|
| **P1** | Named constants for fractions (0.70, 0.50, etc.) | All | Low — 30 min |
| **P1** | Fix redundant OR condition in entry guard | `volume_profile_strategy.py` | Low — 5 min |
| **P1** | Wire up or delete ADX dead code | `walk_forward_eval.py` | Low — 20 min |
| **P1** | Log exceptions in grid search | `walk_forward_eval.py` | Low — 5 min |
| **P2** | Merge 5 TF dicts into 1 dict-of-dicts | `nasdaq_long_chart.py` | Low — 20 min |
| **P2** | Add `dropna` before `compute_vp()` | All | Low — 5 min each |
| **P2** | Rename abbreviations (`sh`, `ecost`, etc.) | `walk_forward_eval.py` | Medium — 1 hr |
| **P2** | Reset `in_pos` when `sh` reaches 0 | `walk_forward_eval.py` | Low — 10 min |
| **P3** | Extract `compute_vp()` to shared module | All | Medium — 2 hr |
| **P3** | Extract exit logic to shared helper | All | Medium — 2 hr |
| **P3** | Add parameter assertions | All | Low — 30 min |
| **P4** | Position state → dataclass | All | High — 4 hr |
| **P4** | Unit tests for VP, metrics, flip detection | New file | High — 4 hr |
| **P4** | Make EMA window configurable | `staircase_breakout_strategy.py` | Low — 10 min |
