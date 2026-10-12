"""Kursdaten laden: Yahoo Finance (kostenlos) oder CSV-Dateien.

Es werden nur ABGESCHLOSSENE Kerzen zurückgegeben. Die letzte, noch
laufende Stundenkerze wird verworfen.
"""

from __future__ import annotations

import time as _time
from typing import Optional

import pandas as pd

from config import Config

COLS = ["open", "high", "low", "close", "volume"]


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns=str.lower)
    if "volume" not in df:
        df["volume"] = 0.0
    df = df[COLS].astype(float).dropna(subset=["open", "high", "low", "close"])
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df


def _download(symbol: str, interval: str, period: str, tries: int = 3) -> pd.DataFrame:
    import yfinance as yf

    last_err: Optional[Exception] = None
    for attempt in range(tries):
        try:
            df = yf.download(symbol, interval=interval, period=period,
                             auto_adjust=False, progress=False, threads=False)
            if df is not None and len(df):
                return _clean(df)
            last_err = RuntimeError(f"Keine Daten für {symbol} ({interval})")
        except Exception as e:  # Netzwerk, Yahoo-Ausfall
            last_err = e
        _time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"Download fehlgeschlagen: {symbol} {interval}: {last_err}")


def drop_unfinished(h1: pd.DataFrame, now: Optional[pd.Timestamp] = None) -> pd.DataFrame:
    """Entfernt Stundenkerzen, die noch nicht abgeschlossen sind."""
    now = now or pd.Timestamp.now(tz="UTC")
    idx = h1.index if h1.index.tz is not None else h1.index.tz_localize("UTC")
    return h1[(idx + pd.Timedelta(hours=1)) <= now]


def load_yahoo(cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    h1 = drop_unfinished(_download(cfg.symbol, "60m", "730d"))
    d1 = _download(cfg.symbol, "1d", "3y")
    try:
        fx = _download(cfg.fx_symbol, "1d", "10d")
        eurusd = float(fx["close"].iloc[-1])
    except RuntimeError:
        eurusd = cfg.eurusd_fallback
    return h1, d1, eurusd


def load_csv(path: str) -> pd.DataFrame:
    """CSV mit Spalten time/date, open, high, low, close[, volume]."""
    raw = pd.read_csv(path)
    raw.columns = [c.strip().lower() for c in raw.columns]
    tcol = "time" if "time" in raw else "date" if "date" in raw else raw.columns[0]
    t = raw[tcol]
    if pd.api.types.is_numeric_dtype(t):
        idx = pd.to_datetime(t, unit="s", utc=True)
    else:
        idx = pd.to_datetime(t, utc=True)
    df = raw.drop(columns=[tcol])
    df.index = pd.DatetimeIndex(idx)
    return _clean(df)
