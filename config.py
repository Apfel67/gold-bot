"""Einstellungen des Gold-Bots.

Alle Werte entsprechen den Standardeinstellungen des TradingView-Indikators
"SHA + Kalman Trading-System", damit Bot und Chart dieselben Signale liefern.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    # ── Daten ──────────────────────────────────────────────
    symbol: str = "GC=F"            # Gold-Futures bei Yahoo Finance
    fx_symbol: str = "EURUSD=X"     # für die Umrechnung USD -> EUR
    eurusd_fallback: float = 1.12   # falls der Kurs nicht geladen werden kann

    # ── Indikatoren (wie im Pine-Skript) ───────────────────
    sha_len1: int = 10
    sha_len2: int = 10
    measurement_noise: float = 500.0
    osc_smoothness: int = 10
    trend_lookback: int = 10
    strength_smoothness: int = 10
    kalman_model: str = "standard"  # "standard", "volume", "parkinson"

    # ── Checkliste ────────────────────────────────────────
    use_daily_filter: bool = True
    use_session: bool = False       # nur zwischen session_start und session_end handeln
    session_start_hour: int = 9     # Berliner Zeit
    session_end_hour: int = 22
    max_losses_per_day: int = 2

    # ── Risiko & Stop ─────────────────────────────────────
    start_balance: float = 575.0
    risk_pct: float = 5.0
    swing_len: int = 10
    stop_buffer: float = 2.0        # USD
    min_qty: float = 0.01           # Unzen
    qty_step: float = 0.01
    pause_below: float = 400.0      # unter diesem Kontostand keine neuen Trades

    # ── Kosten des Papierkontos ───────────────────────────
    spread: float = 0.4             # USD je Unze (Kauf zu Ask, Verkauf zu Bid)
    financing_rate_pa: float = 0.06 # Annahme für Übernacht-Finanzierung (6 % p. a.)

    # ── Laufzeit ──────────────────────────────────────────
    timezone: str = "Europe/Berlin"
    max_catchup_bars: int = 72      # so viele verpasste Kerzen holt ein Lauf nach

    def sha_kwargs(self) -> dict:
        return {"len1": self.sha_len1, "len2": self.sha_len2}

    def kalman_kwargs(self) -> dict:
        return {
            "measurement_noise": self.measurement_noise,
            "osc_smoothness": self.osc_smoothness,
            "trend_lookback": self.trend_lookback,
            "strength_smoothness": self.strength_smoothness,
            "model": self.kalman_model,
        }


CONFIG = Config()
