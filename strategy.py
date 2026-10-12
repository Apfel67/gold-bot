"""Signal-Logik: identisch zum TradingView-Indikator "SHA + Kalman Trading-System".

prepare() nimmt Stunden- und Tageskerzen und liefert pro Stundenkerze alle
Werte, die der Bot für die Checkliste braucht. Es werden nur Informationen
verwendet, die zum Schluss der jeweiligen Kerze bekannt waren.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import Config
from indicators import WARMUP_BARS, add_indicators

EXCHANGE_TZ = "America/New_York"
SESSION_ROLL_HOUR = 18  # Gold-Futures: neuer Handelstag beginnt um 18:00 New York


def _as_utc(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    return index.tz_localize("UTC") if index.tz is None else index.tz_convert("UTC")


def trading_dates(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Handelstag einer Stundenkerze (18:00 New York = Beginn des nächsten Tages)."""
    local = _as_utc(index).tz_convert(EXCHANGE_TZ)
    shifted = local + pd.Timedelta(hours=24 - SESSION_ROLL_HOUR)
    return shifted.normalize().tz_localize(None).astype("datetime64[ns]")


def daily_keys(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Datum einer Tageskerze ohne Zeitzone."""
    if index.tz is not None:
        index = index.tz_convert(EXCHANGE_TZ).tz_localize(None)
    return index.normalize().astype("datetime64[ns]")


def daily_from_hourly(h1: pd.DataFrame) -> pd.DataFrame:
    """Baut Tageskerzen aus Stundenkerzen (für Tests und als Notlösung)."""
    keys = trading_dates(h1.index)
    agg = {"open": "first", "high": "max", "low": "min", "close": "last"}
    if "volume" in h1:
        agg["volume"] = "sum"
    d1 = h1.groupby(keys).agg(agg)
    d1.index.name = None
    return d1


def prepare(h1: pd.DataFrame, d1: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Berechnet Indikatoren, Tagesfilter und Signale für jede Stundenkerze."""
    h = add_indicators(h1, cfg.sha_kwargs(), cfg.kalman_kwargs())
    d = add_indicators(d1, cfg.sha_kwargs(), cfg.kalman_kwargs())

    # Tagesfilter: Werte der letzten ABGESCHLOSSENEN Tageskerze
    dfilt = pd.DataFrame({
        "key": daily_keys(d.index),
        "d_bull": d["sha_bull"].to_numpy(),
        "d_osc": d["osc"].to_numpy(),
        "d_neutral": d["osc_neutral"].to_numpy(),
        "d_valid": np.arange(len(d)) >= WARMUP_BARS,
    }).sort_values("key")
    left = pd.DataFrame({"key": trading_dates(h.index)})
    merged = pd.merge_asof(left, dfilt, on="key", direction="backward",
                           allow_exact_matches=False)
    merged.index = h.index

    d_valid = merged["d_valid"].fillna(False).astype(bool)
    d_bull = merged["d_bull"].fillna(False).astype(bool)
    d_neutral = merged["d_neutral"].fillna(True).astype(bool)
    d_osc = merged["d_osc"].astype(float)

    if cfg.use_daily_filter:
        d_long = d_valid & d_bull & (d_osc > 0) & ~d_neutral
        d_short = d_valid & ~d_bull & (d_osc < 0) & ~d_neutral
    else:
        d_long = pd.Series(True, index=h.index)
        d_short = pd.Series(True, index=h.index)

    bull = h["sha_bull"].astype(bool)
    neutral = h["osc_neutral"].astype(bool)
    osc = h["osc"]
    osc_prev = osc.shift(1)

    long_setup = bull & (osc > 0) & ~neutral
    short_setup = ~bull & (osc < 0) & ~neutral
    long_trig = long_setup & ~long_setup.shift(1, fill_value=False) & (osc > osc_prev)
    short_trig = short_setup & ~short_setup.shift(1, fill_value=False) & (osc < osc_prev)

    berlin_hour = _as_utc(h.index).tz_convert(cfg.timezone).hour
    if cfg.use_session:
        in_session = (berlin_hour >= cfg.session_start_hour) & (berlin_hour < cfg.session_end_hour)
    else:
        in_session = np.ones(len(h), dtype=bool)

    out = pd.DataFrame({
        "open": h["open"], "high": h["high"], "low": h["low"], "close": h["close"],
        "sha_bull": bull,
        "osc": osc,
        "neutral": neutral,
        "d_osc": d_osc,
        "d_long": d_long,
        "d_short": d_short,
        "long_trig": long_trig,
        "short_trig": short_trig,
        "exit_long": ~bull | (osc < 0),
        "exit_short": bull | (osc > 0),
        "swing_lo": h["low"].rolling(cfg.swing_len).min(),
        "swing_hi": h["high"].rolling(cfg.swing_len).max(),
        "in_session": in_session,
    }, index=h.index)
    out["valid"] = np.arange(len(out)) >= WARMUP_BARS
    return out
