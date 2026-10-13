"""Test mit alten Gold-Daten (Out-of-Sample).

Lädt XAU/USD-Stundenkerzen (Spotkurs, Bid) kostenlos von Dukascopy und
rechnet die Strategie über einen frei wählbaren Zeitraum, z. B. 2010-2023:
Bullenmarkt bis 2011, Absturz 2013, Seitwärtsphase bis 2018, Rally 2019/20.

Ausgabe:
  * Ergebnis je Jahr (mit der Goldpreis-Entwicklung im selben Jahr)
  * Gold-Varianten aus dem Robustheitstest auf denselben alten Daten
"""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from config import Config
from robustness import STOP_BUFFER_PCT, build_variants, run_trades, stats
from strategy import daily_from_hourly, prepare

CACHE_DIR = Path("data_cache")


def load_dukascopy(start: datetime, end: datetime) -> pd.DataFrame:
    """XAU/USD-Stundenkerzen von Dukascopy (UTC), mit lokalem Zwischenspeicher."""
    CACHE_DIR.mkdir(exist_ok=True)
    cache = CACHE_DIR / f"xauusd_h1_{start:%Y%m%d}_{end:%Y%m%d}.csv"
    if cache.exists():
        df = pd.read_csv(cache, index_col=0, parse_dates=True)
    else:
        import dukascopy_python as dk
        from dukascopy_python.instruments import INSTRUMENT_FX_METALS_XAU_USD

        df = dk.fetch(INSTRUMENT_FX_METALS_XAU_USD, dk.INTERVAL_HOUR_1,
                      dk.OFFER_SIDE_BID, start, end)
        df.to_csv(cache)
    return clean_history(df)


def clean_history(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]].astype(float)
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    # Wochenend-/Feiertagskerzen ohne Handel entfernen (würden die Indikatoren verzerren)
    flat = (df["volume"] <= 0) & (df["high"] == df["low"])
    df = df[~flat].dropna()
    df = df[~df.index.duplicated(keep="last")].sort_index()
    if len(df) < 1000:
        raise RuntimeError(f"Zu wenige Kerzen geladen ({len(df)}).")
    median = df["close"].median()
    if not 300 < median < 20000:
        raise RuntimeError(f"Unerwartetes Preisformat (Median {median:.2f}).")
    return df


def yearly_table(trades: list, h1: pd.DataFrame) -> pd.DataFrame:
    if not trades:
        return pd.DataFrame()
    t = pd.DataFrame({
        "Jahr": [pd.Timestamp(x.entry_time).year for x in trades],
        "r": [x.r for x in trades],
    })
    out = t.groupby("Jahr")["r"].agg(
        Trades="count",
        **{"Treffer %": lambda r: 100 * (r > 0).mean(), "Ø R": "mean", "Summe R": "sum"},
    )
    closes = h1["close"].groupby(h1.index.year)
    out["Gold im Jahr %"] = (closes.last() / closes.first() - 1) * 100
    return out.reset_index()


def run_history(base: Config, h1: pd.DataFrame, start: datetime, log=print):
    """Gibt (Jahrestabelle, Variantentabelle, Basis-Trades) zurück."""
    d1 = daily_from_hourly(h1)
    start_ts = pd.Timestamp(start).tz_convert("UTC") if pd.Timestamp(start).tzinfo \
        else pd.Timestamp(start, tz="UTC")
    buffer = float(h1["close"].median()) * STOP_BUFFER_PCT  # Stop-Puffer an Preishöhe anpassen

    rows, base_trades = [], []
    for v in build_variants(base):
        if v.group == "Andere Märkte":
            continue
        cfg = replace(v.cfg, stop_buffer=buffer)
        frame = prepare(h1, d1, cfg)
        if v.transform:
            frame = v.transform(frame)
        trades = run_trades(frame[frame.index >= start_ts], cfg)
        if not base_trades:
            base_trades = trades
        rows.append({"Gruppe": v.group, "Variante": v.name, **stats(trades)})
        log(f"fertig: {v.name}")

    h1_test = h1[h1.index >= start_ts]
    return yearly_table(base_trades, h1_test), pd.DataFrame(rows), base_trades


def year_range(von: int, bis: int) -> tuple[datetime, datetime, datetime]:
    """(Download-Start mit 1 Jahr Vorlauf, Teststart, Ende)."""
    if bis < von:
        raise ValueError("'bis' muss nach 'von' liegen.")
    return (datetime(von - 1, 1, 1, tzinfo=timezone.utc),
            datetime(von, 1, 1, tzinfo=timezone.utc),
            datetime(bis + 1, 1, 1, tzinfo=timezone.utc))


def format_yearly(df: pd.DataFrame) -> str:
    out = df.copy()
    out["Treffer %"] = out["Treffer %"].map(lambda x: f"{x:.0f}")
    out["Ø R"] = out["Ø R"].map(lambda x: f"{x:+.2f}")
    out["Summe R"] = out["Summe R"].map(lambda x: f"{x:+.1f}")
    out["Gold im Jahr %"] = out["Gold im Jahr %"].map(lambda x: "" if pd.isna(x) else f"{x:+.0f}")
    return out.to_string(index=False)
