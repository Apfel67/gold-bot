"""Tests für indicators.py.

Ausführen mit:  python -m pytest test_indicators.py -q
"""

import numpy as np
import pandas as pd
import pytest

from indicators import (
    ema,
    kalman_trend_strength,
    smoothed_heikin_ashi,
    wma,
)


def make_ohlc(n: int = 3000, seed: int = 1) -> pd.DataFrame:
    """Synthetischer Goldkurs (Random Walk um 4000 USD)."""
    rng = np.random.default_rng(seed)
    close = 4000 + np.cumsum(rng.normal(0, 8, n))
    open_ = np.r_[close[0], close[:-1]] + rng.normal(0, 2, n)
    high = np.maximum(open_, close) + np.abs(rng.normal(0, 5, n))
    low = np.minimum(open_, close) - np.abs(rng.normal(0, 5, n))
    volume = rng.integers(100, 5000, n).astype(float)
    idx = pd.date_range("2024-01-01", periods=n, freq="h")
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )


def kalman_matrix_reference(df, measurement_noise=500.0, q1=0.01, q2=0.01,
                            model="standard", mintick=0.01):
    """Zeile für Zeile wie das Pine-Skript, mit Matrizen (Referenz)."""
    src = df["close"].to_numpy()
    high, low = df["high"].to_numpy(), df["low"].to_numpy()
    vol = df["volume"].to_numpy()
    F = np.array([[1.0, 1.0], [1.0, 0.0]])
    P = np.eye(2)
    Q = np.array([[q1, q1 * q2], [q2 * q1, q2]])
    H = np.array([[1.0, 0.0]])
    I = np.eye(2)
    X = np.array([src[0], src[0]])
    est, osc = [], []
    for i in range(len(src)):
        X = np.array([F[0, 0] * X[0] + F[0, 1] * X[1], F[1, 1] * X[1]])
        P = F @ (P @ F.T) + Q
        R = np.array([[measurement_noise]])
        if model != "standard" and i > 2:
            if model == "volume":
                R = R * vol[i - 1] / min(vol[i - 1], vol[i])
            else:
                rr = (high[i] - low[i]) / max(high[i - 1] - low[i - 1], mintick)
                R = R * (1 + rr)
        S = H @ (P @ H.T) + R
        K = P @ (H.T @ np.linalg.inv(S))
        innovation = src[i] - (H @ X)[0]
        diff = K * innovation
        X = np.array([X[0] + diff[0, 0], X[1] + diff[1, 0]])
        P = (I + K @ H * -1) @ P
        est.append(X[0])
        osc.append(X[1])
    return np.array(est), np.array(osc)


@pytest.mark.parametrize("model", ["standard", "volume", "parkinson"])
def test_kalman_matches_matrix_reference(model):
    df = make_ohlc(800)
    ref_est, ref_osc = kalman_matrix_reference(df, model=model)
    out = kalman_trend_strength(df, model=model)
    np.testing.assert_allclose(out["kalman_estimate"], ref_est, rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(out["kalman_raw"], ref_osc, rtol=1e-9, atol=1e-9)


def test_wma_matches_naive():
    x = np.random.default_rng(3).normal(size=60)
    got = wma(x, 10)
    w = np.arange(1, 11)
    for i in range(9, 60):
        assert got[i] == pytest.approx(np.dot(x[i - 9:i + 1], w) / w.sum())
    assert np.isnan(got[:9]).all()


def test_wma_propagates_nan():
    x = np.r_[np.nan, np.arange(20.0)]
    got = wma(x, 5)
    assert np.isnan(got[:5]).all()
    assert not np.isnan(got[5])


def test_ema_recursion():
    x = np.arange(1.0, 21.0)
    got = ema(x, 10)
    alpha = 2 / 11
    expected = x[0]
    for v, g in zip(x, got):
        expected = v if v == x[0] else alpha * v + (1 - alpha) * expected
        assert g == pytest.approx(expected)


def test_oscillator_range_and_neutral():
    df = make_ohlc()
    out = kalman_trend_strength(df)
    valid = out["trend_strength"].dropna()
    assert valid.abs().max() <= 100 + 1e-9
    small = out["trend_strength"].abs() < 10
    assert out.loc[small, "osc_neutral"].all()
    # Trendstärke gibt es erst ab Kerze N2 + R2 - 1 = 19; davor gilt "neutral"
    assert out["trend_strength"].iloc[:18].isna().all()
    assert out["osc_neutral"].iloc[:18].all()


def test_sha_candles_consistent():
    df = make_ohlc()
    sha = smoothed_heikin_ashi(df).iloc[100:]
    assert (sha["sha_high"] >= sha[["sha_open", "sha_close"]].min(axis=1)).all()
    assert (sha["sha_bull"] == ~(sha["sha_open"] > sha["sha_close"])).all()
    assert sha["sha_strength"].between(0, 1).all()


def test_warmup_independent_of_start():
    """Nach WARMUP_BARS dürfen die Werte nicht vom Datenbeginn abhängen.

    Wichtig, weil TradingView und Python nie exakt dieselbe Historie laden.
    """
    from indicators import WARMUP_BARS, add_indicators
    df = make_ohlc(3000)
    full = add_indicators(df).iloc[1000 + WARMUP_BARS:]
    part = add_indicators(df.iloc[1000:]).iloc[WARMUP_BARS:]
    for col in ["sha_close", "kalman_estimate", "osc"]:
        np.testing.assert_allclose(part[col], full[col], atol=1e-6)
    assert (part["sha_bull"] == full["sha_bull"]).all()
    assert (part["osc_neutral"] == full["osc_neutral"]).all()


def test_sha_follows_trend():
    n = 400
    close = np.linspace(4000, 4400, n)
    df = pd.DataFrame({"open": close - 1, "high": close + 3,
                       "low": close - 3, "close": close})
    sha = smoothed_heikin_ashi(df)
    assert sha["sha_bull"].iloc[50:].all()
