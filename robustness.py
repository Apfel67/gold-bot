"""Robustheitstests: dieselbe Strategie mit veränderten Kosten, Parametern
und auf anderen Märkten.

Der Live-Bot und config.py bleiben dabei unverändert; jede Variante bekommt
eine eigene Kopie der Einstellungen.

Bewertet wird in R (Vielfache des geplanten Risikos), damit Ergebnisse
unabhängig vom Kontostand und von der Preishöhe des Marktes vergleichbar sind.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable, Optional

import numpy as np
import pandas as pd

from broker import PaperBroker
from config import Config
from engine import Engine
from strategy import prepare

# Für Robustheitstests: keine Mindestgröße und keine Pause, damit jede Variante
# alle Signale handelt und nur die Strategie selbst verglichen wird.
TEST_OVERRIDES = dict(min_qty=1e-6, qty_step=1e-6, pause_below=0.0)


@dataclass
class Variant:
    name: str
    group: str
    cfg: Config
    symbol: Optional[str] = None                      # None = Gold-Daten
    transform: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None


# Andere Märkte: Yahoo-Symbol, typischer CFD-Spread in Preiseinheiten
MARKETS = [
    ("Silber", "SI=F", 0.03),
    ("Kupfer", "HG=F", 0.003),
    ("Öl (WTI)", "CL=F", 0.03),
    ("S&P 500", "ES=F", 0.5),
    ("DAX", "^GDAXI", 2.0),
    ("EUR/USD", "EURUSD=X", 0.0001),
]
STOP_BUFFER_PCT = 0.0005   # 2 USD bei Gold um 4.000 USD


def delay_entry(frame: pd.DataFrame) -> pd.DataFrame:
    """Einstieg eine Kerze später (verzögerte Ausführung)."""
    f = frame.copy()
    for col in ["long_trig", "short_trig", "d_long", "d_short"]:
        f[col] = f[col].shift(1, fill_value=False).astype(bool)
    return f


def build_variants(base: Config) -> list[Variant]:
    b = replace(base, **TEST_OVERRIDES)
    g = "Gold"
    v = [
        Variant("Basis (wie Live-Bot)", g, b),
        Variant("Spread 0,8 USD", "Kosten", replace(b, spread=0.8)),
        Variant("Spread 1,5 USD", "Kosten", replace(b, spread=1.5)),
        Variant("Einstieg 1 Stunde später", "Kosten", b, transform=delay_entry),
        Variant("SHA-Glättung 8", "Parameter", replace(b, sha_len1=8, sha_len2=8)),
        Variant("SHA-Glättung 12", "Parameter", replace(b, sha_len1=12, sha_len2=12)),
        Variant("Measurement Noise 250", "Parameter", replace(b, measurement_noise=250.0)),
        Variant("Measurement Noise 1000", "Parameter", replace(b, measurement_noise=1000.0)),
        Variant("Osc Smoothness 7", "Parameter", replace(b, osc_smoothness=7)),
        Variant("Osc Smoothness 14", "Parameter", replace(b, osc_smoothness=14)),
        Variant("Stop hinter 6 Kerzen", "Parameter", replace(b, swing_len=6)),
        Variant("Stop hinter 15 Kerzen", "Parameter", replace(b, swing_len=15)),
        Variant("Ohne Tagesfilter", "Parameter", replace(b, use_daily_filter=False)),
    ]
    for name, sym, spread in MARKETS:
        v.append(Variant(name, "Andere Märkte", replace(b, symbol=sym, spread=spread), symbol=sym))
    return v


def run_trades(frame: pd.DataFrame, cfg: Config) -> list:
    engine = Engine(cfg, PaperBroker(cfg))
    trades = []
    for t, row in frame.iterrows():
        trades += [e.trade for e in engine.on_bar(t, row) if e.trade is not None]
    return trades


def stats(trades: list) -> dict:
    n = len(trades)
    if n == 0:
        return {"Trades": 0}
    r = np.array([t.r for t in trades])
    side = np.array([t.side for t in trades])
    sd = r.std(ddof=1) if n > 1 else np.nan
    cum = np.cumsum(r)
    half = n // 2

    def mean(x):
        return float(np.mean(x)) if len(x) else np.nan

    return {
        "Trades": n,
        "Treffer %": 100.0 * (r > 0).mean(),
        "Ø R": r.mean(),
        "95%-Untergrenze": r.mean() - 1.96 * sd / np.sqrt(n) if n > 1 else np.nan,
        "Ø R ohne Top 5": mean(np.sort(r)[:-5]) if n > 5 else np.nan,
        "Max. Drawdown R": float((cum - np.maximum.accumulate(np.r_[0.0, cum])[1:]).min()),
        "Ø R Long": mean(r[side == 1]),
        "Ø R Short": mean(r[side == -1]),
        "Ø R 1. Hälfte": mean(r[:half]),
        "Ø R 2. Hälfte": mean(r[half:]),
    }


def run_robustness(base: Config, load: Callable[[str], tuple[pd.DataFrame, pd.DataFrame]],
                   log: Callable[[str], None] = print) -> pd.DataFrame:
    """Alle Varianten rechnen. `load(symbol)` liefert (Stunden, Tage)."""
    cache: dict[str, tuple] = {}
    rows = []
    for v in build_variants(base):
        sym = v.symbol or base.symbol
        try:
            if sym not in cache:
                cache[sym] = load(sym)
            h1, d1 = cache[sym]
            cfg = v.cfg
            if v.symbol:  # Stop-Puffer an die Preishöhe des Marktes anpassen
                cfg = replace(cfg, stop_buffer=float(h1["close"].median()) * STOP_BUFFER_PCT)
            frame = prepare(h1, d1, cfg)
            if v.transform:
                frame = v.transform(frame)
            row = {"Gruppe": v.group, "Variante": v.name, **stats(run_trades(frame, cfg))}
        except Exception as e:  # ein fehlender Markt soll den Rest nicht stoppen
            row = {"Gruppe": v.group, "Variante": v.name, "Trades": 0, "Fehler": str(e)[:80]}
        rows.append(row)
        log(f"fertig: {v.name}")
    return pd.DataFrame(rows)


def format_table(df: pd.DataFrame) -> str:
    out = df.copy()
    for c in out.columns:
        if c == "Treffer %":
            out[c] = out[c].map(lambda x: "" if pd.isna(x) else f"{x:.0f}")
        elif c not in ("Gruppe", "Variante", "Trades", "Fehler"):
            out[c] = out[c].map(lambda x: "" if pd.isna(x) else f"{x:+.2f}")
    return out.to_string(index=False)
