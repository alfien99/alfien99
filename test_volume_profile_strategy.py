"""
Tests for volume_profile_strategy.py

Coverage areas:
  1. compute_vp        — core algorithm: POC/VAL/VAH ordering, 70% value-area
                         invariant, edge cases (degenerate range, zero-range bars)
  2. compute_stats     — return / drawdown / Sharpe / win-rate arithmetic
  3. _synthetic_ohlcv  — data integrity (OHLC ordering, positive prices/volumes,
                         date range, determinism)
  4. run_backtest      — output structure, equity-curve length, capital
                         accounting, levels ordering, valid trade types
  5. _download         — fallback to synthetic when yfinance is unavailable or
                         raises, or returns too few bars
"""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock, patch

import volume_profile_strategy as vps
from volume_profile_strategy import (
    DEFAULT_CONFIG,
    _download,
    _synthetic_ohlcv,
    compute_stats,
    compute_vp,
    run_backtest,
)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_ohlcv(n: int = 50, seed: int = 42, s0: float = 100.0) -> pd.DataFrame:
    """Deterministic OHLCV DataFrame for testing — High >= Close >= Low by construction."""
    rng = np.random.default_rng(seed)
    closes = s0 + np.cumsum(rng.standard_normal(n) * 0.5)
    hl = np.abs(rng.standard_normal(n)) * 0.4 + 0.05
    highs = closes + hl
    lows = closes - hl
    opens = np.roll(closes, 1)
    opens[0] = closes[0]
    volumes = rng.integers(1_000, 10_000, n).astype(float)
    idx = pd.date_range("2023-01-01", periods=n, freq="B")
    return pd.DataFrame(
        {"Open": opens, "High": highs, "Low": lows, "Close": closes, "Volume": volumes},
        index=idx,
    )


def _make_equity(values: list, start: str = "2023-01-01") -> pd.DataFrame:
    idx = pd.date_range(start, periods=len(values), freq="B")
    return pd.DataFrame({"equity": values}, index=idx)


# ── 1. compute_vp ──────────────────────────────────────────────────────────────

class TestComputeVP:
    def test_val_le_poc_le_vah(self):
        df = _make_ohlcv(60)
        poc, val, vah, mids, vol = compute_vp(df, bins=50)
        assert val <= poc <= vah

    def test_poc_is_max_volume_bin(self):
        df = _make_ohlcv(60)
        poc, val, vah, mids, vol = compute_vp(df, bins=50)
        expected_poc = mids[int(np.argmax(vol))]
        assert poc == pytest.approx(expected_poc)

    def test_value_area_captures_70pct_volume(self):
        df = _make_ohlcv(100)
        poc, val, vah, mids, vol = compute_vp(df, bins=100)
        mask = (mids >= val - 1e-9) & (mids <= vah + 1e-9)
        captured_pct = vol[mask].sum() / vol.sum()
        assert captured_pct >= 0.69

    def test_output_arrays_match_bins(self):
        bins = 40
        df = _make_ohlcv(30)
        _, _, _, mids, vol = compute_vp(df, bins=bins)
        assert len(mids) == bins
        assert len(vol) == bins

    def test_total_volume_is_preserved(self):
        """Volume distributed into bins must equal total bar volume."""
        df = _make_ohlcv(50)
        _, _, _, _, vol = compute_vp(df, bins=80)
        assert vol.sum() == pytest.approx(df["Volume"].sum(), rel=1e-5)

    def test_degenerate_range_all_same_price(self):
        """When high == low for every bar, returns poc == val == vah == that price."""
        idx = pd.date_range("2023-01-01", periods=5, freq="B")
        df = pd.DataFrame(
            {"Open": [100.0]*5, "High": [100.0]*5, "Low": [100.0]*5,
             "Close": [100.0]*5, "Volume": [1000.0]*5},
            index=idx,
        )
        poc, val, vah, mids, vol = compute_vp(df, bins=100)
        assert poc == pytest.approx(100.0)
        assert val == pytest.approx(100.0)
        assert vah == pytest.approx(100.0)

    def test_single_bar_no_crash(self):
        idx = pd.date_range("2023-01-01", periods=1, freq="B")
        df = pd.DataFrame(
            {"Open": [98.0], "High": [102.0], "Low": [97.0],
             "Close": [100.0], "Volume": [5000.0]},
            index=idx,
        )
        poc, val, vah, mids, vol = compute_vp(df, bins=20)
        assert val <= poc <= vah

    def test_mixed_zero_range_and_normal_bars(self):
        """Bars where high == low must not cause a zero-division crash."""
        idx = pd.date_range("2023-01-01", periods=10, freq="B")
        df = pd.DataFrame(
            {"Open":   [100.0]*10,
             "High":   [100.0]*5 + [106.0]*5,
             "Low":    [100.0]*5 + [94.0]*5,
             "Close":  [100.0]*5 + [100.0]*5,
             "Volume": [500.0]*10},
            index=idx,
        )
        poc, val, vah, mids, vol = compute_vp(df, bins=50)
        assert val <= poc <= vah

    def test_poc_within_price_range(self):
        df = _make_ohlcv(60)
        poc, val, vah, mids, vol = compute_vp(df, bins=50)
        lo = float(df["Low"].min())
        hi = float(df["High"].max())
        assert lo <= poc <= hi


# ── 2. compute_stats ───────────────────────────────────────────────────────────

class TestComputeStats:
    def test_total_return_formula(self):
        eq = _make_equity([10_000, 10_500, 11_000])
        trades = pd.DataFrame({"pnl": [500.0], "type": ["target"]})
        stats = compute_stats(eq, 10_000, trades)
        assert stats["Total Return"] == "+10.0%"

    def test_zero_drawdown_on_monotone_equity(self):
        eq = _make_equity([10_000, 10_100, 10_200, 10_300])
        trades = pd.DataFrame(columns=["pnl", "type"])
        stats = compute_stats(eq, 10_000, trades)
        assert stats["Max Drawdown"] == "0.0%"

    def test_known_drawdown_value(self):
        # Peak 12 000, trough 9 000 → (9000 - 12000) / 12000 = -25 %
        eq = _make_equity([10_000, 12_000, 9_000, 10_000])
        trades = pd.DataFrame(columns=["pnl", "type"])
        stats = compute_stats(eq, 10_000, trades)
        dd = float(stats["Max Drawdown"].replace("%", ""))
        assert dd == pytest.approx(-25.0, abs=0.2)

    def test_sharpe_zero_for_flat_equity(self):
        eq = _make_equity([10_000] * 20)
        trades = pd.DataFrame(columns=["pnl", "type"])
        stats = compute_stats(eq, 10_000, trades)
        assert stats["Sharpe"] == "0.00"

    def test_no_trades_produces_zero_stats(self):
        eq = _make_equity([10_000, 10_500])
        trades = pd.DataFrame(columns=["pnl", "type"])
        stats = compute_stats(eq, 10_000, trades)
        assert stats["Win Rate"] == "0%"
        assert stats["Trades"] == "0"
        assert stats["Avg Win"] == "$0"
        assert stats["Avg Loss"] == "$0"

    def test_win_rate_all_wins(self):
        eq = _make_equity([10_000, 11_000])
        trades = pd.DataFrame({"pnl": [100.0, 200.0, 300.0], "type": ["target"]*3})
        stats = compute_stats(eq, 10_000, trades)
        assert stats["Win Rate"] == "100%"
        assert stats["Trades"] == "3"

    def test_win_rate_fifty_fifty(self):
        eq = _make_equity([10_000, 11_000])
        trades = pd.DataFrame({
            "pnl":  [100.0, -50.0, 200.0, -30.0],
            "type": ["target", "stop", "target", "stop"],
        })
        stats = compute_stats(eq, 10_000, trades)
        assert stats["Win Rate"] == "50%"

    def test_avg_win_positive_avg_loss_negative(self):
        eq = _make_equity([10_000, 11_000])
        trades = pd.DataFrame({
            "pnl":  [300.0, 200.0, -100.0, -50.0],
            "type": ["target", "target", "stop", "stop"],
        })
        stats = compute_stats(eq, 10_000, trades)
        avg_win  = float(stats["Avg Win"].replace("$", "").replace(",", ""))
        avg_loss = float(stats["Avg Loss"].replace("$", "").replace(",", ""))
        assert avg_win > 0
        assert avg_loss < 0

    def test_final_equity_matches_last_row(self):
        eq = _make_equity([10_000, 10_500, 12_345])
        trades = pd.DataFrame(columns=["pnl", "type"])
        stats = compute_stats(eq, 10_000, trades)
        assert "12,345" in stats["Final Equity"]

    def test_annualised_return_positive_for_gains(self):
        # 20 % total return over ~252 bdays ≈ 1 year → annualised ≈ +20 %
        eq = _make_equity([10_000] + [12_000] * 251)
        trades = pd.DataFrame(columns=["pnl", "type"])
        stats = compute_stats(eq, 10_000, trades)
        ann = float(stats["Ann. Return"].replace("%", "").replace("+", ""))
        assert ann > 0


# ── 3. _synthetic_ohlcv ────────────────────────────────────────────────────────

class TestSyntheticOHLCV:
    def test_required_columns_present(self):
        df = _synthetic_ohlcv("TEST", "2023-01-01", "2023-06-30")
        for col in ("Open", "High", "Low", "Close", "Volume"):
            assert col in df.columns, f"Missing column: {col}"

    def test_high_ge_close_ge_low(self):
        df = _synthetic_ohlcv("TEST", "2023-01-01", "2024-01-01")
        assert (df["High"] >= df["Close"]).all(), "High must be >= Close"
        assert (df["Close"] >= df["Low"]).all(),  "Close must be >= Low"
        assert (df["High"] >= df["Low"]).all(),   "High must be >= Low"

    def test_all_volumes_positive(self):
        df = _synthetic_ohlcv("TEST", "2023-01-01", "2023-06-30")
        assert (df["Volume"] > 0).all()

    def test_all_prices_positive(self):
        df = _synthetic_ohlcv("TEST", "2023-01-01", "2023-06-30")
        for col in ("Open", "High", "Low", "Close"):
            assert (df[col] > 0).all(), f"Non-positive prices in column {col}"

    def test_index_contains_only_business_days(self):
        df = _synthetic_ohlcv("TEST", "2023-01-01", "2023-03-31")
        assert all(d.dayofweek < 5 for d in df.index), "Index contains weekend dates"

    def test_date_range_within_bounds(self):
        start, end = "2023-01-01", "2023-03-31"
        df = _synthetic_ohlcv("TEST", start, end)
        assert df.index[0] >= pd.Timestamp(start)
        assert df.index[-1] <= pd.Timestamp(end)

    def test_deterministic_for_same_ticker(self):
        df1 = _synthetic_ohlcv("SPY", "2023-01-01", "2023-06-30")
        df2 = _synthetic_ohlcv("SPY", "2023-01-01", "2023-06-30")
        pd.testing.assert_frame_equal(df1, df2)

    def test_different_tickers_yield_different_prices(self):
        df_spy  = _synthetic_ohlcv("SPY",  "2023-01-01", "2023-06-30")
        df_aapl = _synthetic_ohlcv("AAPL", "2023-01-01", "2023-06-30")
        assert not df_spy["Close"].equals(df_aapl["Close"])

    def test_non_empty_for_valid_range(self):
        df = _synthetic_ohlcv("TEST", "2023-01-01", "2023-06-30")
        assert len(df) > 0

    def test_no_nan_values(self):
        df = _synthetic_ohlcv("TEST", "2023-01-01", "2023-06-30")
        assert not df.isnull().any().any()


# ── 4. run_backtest ────────────────────────────────────────────────────────────

class TestRunBacktest:
    """Patches _download to return synthetic data, avoiding network calls."""

    @pytest.fixture
    def base_cfg(self):
        cfg = DEFAULT_CONFIG.copy()
        cfg.update({
            "ticker": "TEST",
            "start": "2022-01-01",
            "end": "2023-01-01",
            "lookback": 10,
            "vp_bins": 30,
        })
        return cfg

    @pytest.fixture
    def synth_data(self):
        return _synthetic_ohlcv("TEST", "2022-01-01", "2023-01-01")

    def test_returns_six_element_tuple(self, base_cfg, synth_data):
        with patch("volume_profile_strategy._download", return_value=synth_data):
            result = run_backtest(base_cfg)
        assert len(result) == 6

    def test_equity_curve_correct_length(self, base_cfg, synth_data):
        with patch("volume_profile_strategy._download", return_value=synth_data):
            raw, _, equity, _, _, _ = run_backtest(base_cfg)
        expected = len(synth_data) - base_cfg["lookback"]
        assert len(equity) == expected

    def test_equity_curve_always_positive(self, base_cfg, synth_data):
        with patch("volume_profile_strategy._download", return_value=synth_data):
            _, _, equity, _, _, _ = run_backtest(base_cfg)
        assert (equity["equity"] > 0).all()

    def test_levels_df_has_required_columns(self, base_cfg, synth_data):
        with patch("volume_profile_strategy._download", return_value=synth_data):
            _, _, _, levels, _, _ = run_backtest(base_cfg)
        for col in ("poc", "val", "vah", "above"):
            assert col in levels.columns

    def test_levels_val_le_poc_le_vah(self, base_cfg, synth_data):
        with patch("volume_profile_strategy._download", return_value=synth_data):
            _, _, _, levels, _, _ = run_backtest(base_cfg)
        assert (levels["val"] <= levels["poc"]).all()
        assert (levels["poc"] <= levels["vah"]).all()

    def test_trades_df_has_pnl_and_type_columns(self, base_cfg, synth_data):
        with patch("volume_profile_strategy._download", return_value=synth_data):
            _, trades, _, _, _, _ = run_backtest(base_cfg)
        if not trades.empty:
            assert "pnl" in trades.columns
            assert "type" in trades.columns

    def test_trade_types_are_valid_values(self, base_cfg, synth_data):
        valid = {"stop", "target", "expired"}
        with patch("volume_profile_strategy._download", return_value=synth_data):
            _, trades, _, _, _, _ = run_backtest(base_cfg)
        if not trades.empty:
            unknown = set(trades["type"].unique()) - valid
            assert not unknown, f"Unexpected trade types: {unknown}"

    def test_initial_equity_does_not_exceed_capital(self, base_cfg, synth_data):
        """First equity value can't be greater than starting capital."""
        with patch("volume_profile_strategy._download", return_value=synth_data):
            _, _, equity, _, _, _ = run_backtest(base_cfg)
        first = float(equity["equity"].iloc[0])
        assert first <= base_cfg["initial_capital"] * 1.001

    def test_levels_above_ge_vah(self, base_cfg, synth_data):
        """'above' target must be at or above VAH by construction."""
        with patch("volume_profile_strategy._download", return_value=synth_data):
            _, _, _, levels, _, _ = run_backtest(base_cfg)
        assert (levels["above"] >= levels["vah"]).all()


# ── 5. _download ───────────────────────────────────────────────────────────────

class TestDownload:
    def test_returns_dataframe_with_required_columns(self):
        with patch.object(vps, "_YF_AVAILABLE", False):
            df = _download("SPY", "2023-01-01", "2023-06-30")
        for col in ("Open", "High", "Low", "Close", "Volume"):
            assert col in df.columns

    def test_fallback_when_yfinance_not_installed(self):
        with patch.object(vps, "_YF_AVAILABLE", False):
            df = _download("SPY", "2023-01-01", "2023-06-30")
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_fallback_when_yfinance_raises(self):
        with patch.object(vps, "_YF_AVAILABLE", True):
            mock_yf = MagicMock()
            mock_yf.download.side_effect = RuntimeError("network error")
            with patch("volume_profile_strategy.yf", mock_yf):
                df = _download("SPY", "2023-01-01", "2023-06-30")
        for col in ("Open", "High", "Low", "Close", "Volume"):
            assert col in df.columns
        assert len(df) > 0

    def test_fallback_when_yfinance_returns_too_few_bars(self):
        """yfinance returning ≤ 20 bars should trigger the synthetic fallback."""
        tiny = _make_ohlcv(5)   # only 5 bars; threshold in _download is > 20
        with patch.object(vps, "_YF_AVAILABLE", True):
            with patch("volume_profile_strategy.yf") as mock_yf:
                mock_yf.download.return_value = tiny
                df = _download("SPY", "2023-01-01", "2023-06-30")
        assert len(df) > 5, "Should have fallen back to synthetic data"

    def test_result_is_non_empty_dataframe(self):
        with patch.object(vps, "_YF_AVAILABLE", False):
            df = _download("TEST", "2023-01-01", "2023-03-31")
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
