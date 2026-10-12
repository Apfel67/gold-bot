"""Indikatoren für das Gold-Trading-Projekt.

Enthält exakte Python-Portierungen der beiden TradingView-Indikatoren:

1. Smoothed Heikin Ashi Candles (EMA -> Heikin Ashi -> EMA)
2. Adaptive Kalman filter - Trend Strength Oscillator

Lizenzhinweis zum Kalman-Teil:
    Abgeleitet von "Adaptive Kalman filter - Trend Strength Oscillator"
    © Zeiierman, lizenziert unter CC BY-NC-SA 4.0
    (https://creativecommons.org/licenses/by-nc-sa/4.0/).
    Diese Portierung steht unter derselben Lizenz (nicht kommerziell,
    Namensnennung, Weitergabe unter gleichen Bedingungen).

Erwartetes Eingabeformat: pandas.DataFrame mit den Spalten
open, high, low, close (optional volume), chronologisch sortiert,
nur abgeschlossene Kerzen.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Ab dieser Trendstärke (Betrag) ist der Oszillator nicht mehr "blau".
NEUTRAL_THRESHOLD = 10.0

# Kerzen, nach denen beide Indikatoren unabhängig vom Startpunkt der Daten
# sind (gemessen ~50-80 bei Standardeinstellungen; mit Sicherheitsreserve).
# Werte davor nicht für Signale oder Backtests verwenden.
WARMUP_BARS = 150


# ───────────────────────── Hilfsfunktionen (wie Pine ta.*) ─────────────────────────

def ema(values: np.ndarray, length: int) -> np.ndarray:
    """Exponentieller gleitender Durchschnitt wie Pine `ta.ema`.

    Startwert ist der erste gültige Wert; danach alpha = 2 / (length + 1).
    Unterschiede zur Initialisierung in TradingView verschwinden nach
    der Aufwärmphase (WARMUP_BARS).
    """
    values = np.asarray(values, dtype=float)
    alpha = 2.0 / (length + 1.0)
    out = np.full(values.shape, np.nan)
    prev = np.nan
    for i, v in enumerate(values):
        if np.isnan(v):
            out[i] = prev
            continue
        prev = v if np.isnan(prev) else alpha * v + (1.0 - alpha) * prev
        out[i] = prev
    return out


def wma(values: np.ndarray, length: int) -> np.ndarray:
    """Gewichteter gleitender Durchschnitt wie Pine `ta.wma`.

    Gewichte 1..length, neuester Wert am stärksten. Liefert NaN, solange
    das Fenster nicht vollständig mit gültigen Werten gefüllt ist.
    """
    values = np.asarray(values, dtype=float)
    out = np.full(values.shape, np.nan)
    if len(values) < length:
        return out
    weights = np.arange(1, length + 1, dtype=float)
    windows = np.lib.stride_tricks.sliding_window_view(values, length)
    valid = ~np.isnan(windows).any(axis=1)
    sums = np.where(valid, np.nan_to_num(windows) @ weights, np.nan)
    out[length - 1:] = sums / weights.sum()
    return out


def sma(values: np.ndarray, length: int) -> np.ndarray:
    """Einfacher gleitender Durchschnitt wie Pine `ta.sma`."""
    return pd.Series(values, dtype=float).rolling(length).mean().to_numpy()


# ───────────────────────── Smoothed Heikin Ashi ─────────────────────────

def smoothed_heikin_ashi(
    df: pd.DataFrame,
    len1: int = 10,
    len2: int = 10,
    avg_len: int = 50,
    small_mult: float = 0.5,
    large_mult: float = 1.5,
) -> pd.DataFrame:
    """Smoothed Heikin Ashi wie im TradingView-Skript "Smoothed Heiken Ashi Candles".

    Rückgabe-Spalten:
        sha_open, sha_high, sha_low, sha_close  geglättete HA-Kerze
        sha_bull     True = grüne Kerze (im Original: rot, wenn open > close)
        sha_strength 0..1, Kerzengröße relativ zum Durchschnitt (Farbhelligkeit)
    """
    o = ema(df["open"].to_numpy(), len1)
    h = ema(df["high"].to_numpy(), len1)
    l = ema(df["low"].to_numpy(), len1)
    c = ema(df["close"].to_numpy(), len1)

    ha_close = (o + h + l + c) / 4.0
    ha_open = np.full(len(df), np.nan)
    for i in range(len(df)):
        if i == 0 or np.isnan(ha_open[i - 1]):
            ha_open[i] = (o[i] + c[i]) / 2.0
        else:
            ha_open[i] = (ha_open[i - 1] + ha_close[i - 1]) / 2.0

    ha_high = np.maximum(h, np.maximum(ha_open, ha_close))
    ha_low = np.minimum(l, np.minimum(ha_open, ha_close))

    o2 = ema(ha_open, len2)
    c2 = ema(ha_close, len2)
    h2 = ema(ha_high, len2)
    l2 = ema(ha_low, len2)

    candle_size = np.abs(c2 - o2)
    avg_size = sma(candle_size, avg_len)
    with np.errstate(divide="ignore", invalid="ignore"):
        norm = (candle_size - avg_size * small_mult) / (avg_size * (large_mult - small_mult))
    norm = np.clip(norm, 0.0, 1.0)

    return pd.DataFrame(
        {
            "sha_open": o2,
            "sha_high": h2,
            "sha_low": l2,
            "sha_close": c2,
            "sha_bull": ~(o2 > c2),
            "sha_strength": norm,
        },
        index=df.index,
    )


# ───────────────────────── Kalman Trend Strength Oscillator ─────────────────────────

def kalman_trend_strength(
    df: pd.DataFrame,
    measurement_noise: float = 500.0,
    osc_smoothness: int = 10,
    trend_lookback: int = 10,
    strength_smoothness: int = 10,
    model: str = "standard",
    process_noise_1: float = 0.01,
    process_noise_2: float = 0.01,
    mintick: float = 0.01,
) -> pd.DataFrame:
    """Kalman Trend Strength Oscillator, 1:1 wie das Zeiierman-Skript.

    Die Matrixrechnung des Originals ist hier in Einzelformeln ausgeschrieben
    (mathematisch identisch, siehe Test gegen die Matrix-Referenz).

    Eigenheiten des Originals, die bewusst übernommen werden:
      * Übergangsmatrix F = [[1, 1], [1, 0]]; für die Zustandsvorhersage wird
        nur F[1,1] = 0 verwendet, die zweite Zustandsgröße startet also jede
        Kerze bei 0. Der "Oszillator" ist damit die Kalman-Korrektur aus der
        aktuellen Abweichung des Kurses vom Filter.
      * Erster Zustand ist [close, close]; der Filter braucht deshalb eine
        Einschwingphase. Gemessen: nach ca. 50 Kerzen sind die Werte
        unabhängig vom Startpunkt (siehe WARMUP_BARS).

    model: "standard", "volume" oder "parkinson"

    Rückgabe-Spalten:
        kalman_estimate   Filterlinie (im Chart über den Kerzen)
        kalman_raw        zweite Zustandsgröße
        trend_strength    innere Trendstärke (bestimmt Farbe / "blau")
        osc               angezeigte Oszillatorlinie (WMA der Trendstärke)
        osc_neutral       True = blau (|Trendstärke| < 10 oder noch keine Werte)
    """
    if model not in {"standard", "volume", "parkinson"}:
        raise ValueError(f"Unbekanntes Modell: {model}")

    src = df["close"].to_numpy(dtype=float)
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    volume = df["volume"].to_numpy(dtype=float) if "volume" in df else None
    if model == "volume" and volume is None:
        raise ValueError("Modell 'volume' braucht eine Spalte 'volume'.")

    n = len(src)
    q1, q2 = process_noise_1, process_noise_2
    q12 = q1 * q2

    estimate = np.full(n, np.nan)
    raw = np.full(n, np.nan)
    if n == 0:
        return pd.DataFrame(index=df.index)

    x0, x1 = src[0], src[0]
    p00, p01, p10, p11 = 1.0, 0.0, 0.0, 1.0

    for i in range(n):
        # Vorhersage: X = [x0 + x1, 0],  P = F P F^T + Q
        x0 = x0 + x1
        x1 = 0.0
        a00 = p00 + p01 + p10 + p11 + q1
        a01 = p00 + p10 + q12
        a10 = p00 + p01 + q12
        a11 = p00 + q2

        r = measurement_noise
        if model != "standard" and i > 2:
            if model == "volume":
                r = r * volume[i - 1] / min(volume[i - 1], volume[i])
            else:
                current_range = high[i] - low[i]
                previous_range = high[i - 1] - low[i - 1]
                r = r * (1.0 + current_range / max(previous_range, mintick))

        # Korrektur
        s = a00 + r
        k0 = a00 / s
        k1 = a10 / s
        innovation = src[i] - x0
        x0 = x0 + k0 * innovation
        x1 = x1 + k1 * innovation

        # P = (I - K H) P
        p00 = (1.0 - k0) * a00
        p01 = (1.0 - k0) * a01
        p10 = a10 - k1 * a00
        p11 = a11 - k1 * a01

        estimate[i] = x0
        raw[i] = x1

    # Trendstärke: Normierung auf das Betragsmaximum der letzten N2 Werte
    abs_max = pd.Series(np.abs(raw)).rolling(trend_lookback).max().to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        normalized = raw / abs_max * 100.0
    strength = wma(normalized, strength_smoothness)
    osc = wma(strength, osc_smoothness)
    neutral = ~(np.abs(strength) >= NEUTRAL_THRESHOLD)  # NaN zählt als neutral

    return pd.DataFrame(
        {
            "kalman_estimate": estimate,
            "kalman_raw": raw,
            "trend_strength": strength,
            "osc": osc,
            "osc_neutral": neutral,
        },
        index=df.index,
    )


def add_indicators(df: pd.DataFrame, sha_kwargs: dict | None = None,
                   kalman_kwargs: dict | None = None) -> pd.DataFrame:
    """Hängt beide Indikatoren an einen OHLC-DataFrame an."""
    sha = smoothed_heikin_ashi(df, **(sha_kwargs or {}))
    kal = kalman_trend_strength(df, **(kalman_kwargs or {}))
    return pd.concat([df, sha, kal], axis=1)
