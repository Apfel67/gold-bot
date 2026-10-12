"""Konto-Modul: Schnittstelle für einen Broker und das Papierkonto.

Die Engine spricht nur mit der Klasse `Broker`. Für echten Handel (z. B. ein
Capital.com-Demokonto) wird später eine weitere Klasse mit denselben Methoden
geschrieben; Strategie und Engine bleiben unverändert.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from typing import Optional

import pandas as pd

from config import Config


@dataclass
class Position:
    side: int            # +1 Long, -1 Short
    qty: float           # Unzen
    entry_price: float   # tatsächlicher Einstiegskurs (inkl. Spread)
    stop: float
    entry_time: str
    risk_eur: float      # geplantes Risiko bis zum Stop
    stop_dist: float     # USD je Unze zwischen Signal-Kurs und Stop
    financing_eur: float = 0.0
    financed_until: str = ""  # Datum (Berlin), bis zu dem Finanzierung abgerechnet ist


@dataclass
class Trade:
    side: int
    qty: float
    entry_time: str
    entry_price: float
    exit_time: str
    exit_price: float
    reason: str
    pnl_eur: float       # inkl. Spread und Finanzierung
    r: float             # Ergebnis in Vielfachen des geplanten Risikos
    financing_eur: float
    balance_after: float


class Broker(ABC):
    """Was die Engine von einem Konto braucht."""

    eurusd: float = 1.12

    @abstractmethod
    def balance(self) -> float:
        """Aktueller Kontostand in EUR (ohne offene Gewinne/Verluste)."""

    @abstractmethod
    def position(self) -> Optional[Position]:
        """Offene Position oder None."""

    @abstractmethod
    def sync(self, time: pd.Timestamp, bar) -> Optional[Trade]:
        """Ereignisse auf Brokerseite bis zum Ende dieser Kerze verarbeiten.

        Beim Papierkonto: Stop-Auslösung und Übernacht-Finanzierung.
        Bei einem echten Broker: offene Position abfragen und erkennen,
        ob der Stop serverseitig ausgeführt wurde.
        """

    @abstractmethod
    def open(self, time: pd.Timestamp, side: int, qty: float, price: float,
             stop: float, risk_eur: float) -> Position:
        """Position mit Stop-Loss eröffnen."""

    @abstractmethod
    def close(self, time: pd.Timestamp, price: float, reason: str) -> Trade:
        """Offene Position schließen."""

    @abstractmethod
    def to_state(self) -> dict:
        """Zustand zum Speichern zwischen zwei Läufen."""


class PaperBroker(Broker):
    """Simuliertes Konto mit Spread, Kurslücken und Übernacht-Finanzierung."""

    def __init__(self, cfg: Config, state: Optional[dict] = None, eurusd: Optional[float] = None):
        self.cfg = cfg
        state = state or {}
        self._balance = float(state.get("balance", cfg.start_balance))
        pos = state.get("position")
        self._position = Position(**pos) if pos else None
        self.eurusd = float(eurusd or state.get("eurusd") or cfg.eurusd_fallback)

    # ── Abfragen ──────────────────────────────────────────
    def balance(self) -> float:
        return self._balance

    def position(self) -> Optional[Position]:
        return self._position

    def to_state(self) -> dict:
        return {
            "balance": round(self._balance, 6),
            "position": asdict(self._position) if self._position else None,
            "eurusd": self.eurusd,
        }

    # ── Hilfen ────────────────────────────────────────────
    def _half(self) -> float:
        return self.cfg.spread / 2.0

    def _berlin_date(self, time: pd.Timestamp) -> str:
        t = time if time.tzinfo else time.tz_localize("UTC")
        return t.tz_convert(self.cfg.timezone).date().isoformat()

    def _charge_financing(self, time: pd.Timestamp, price: float) -> None:
        pos = self._position
        today = self._berlin_date(time)
        if pos is None or not pos.financed_until or today <= pos.financed_until:
            return
        nights = (pd.Timestamp(today) - pd.Timestamp(pos.financed_until)).days
        notional_eur = pos.qty * price / self.eurusd
        cost = notional_eur * self.cfg.financing_rate_pa / 365.0 * nights
        pos.financing_eur += cost
        pos.financed_until = today
        self._balance -= cost

    # ── Aktionen ──────────────────────────────────────────
    def sync(self, time: pd.Timestamp, bar) -> Optional[Trade]:
        pos = self._position
        if pos is None:
            return None
        self._charge_financing(time, float(bar["open"]))
        if pos.side == 1 and bar["low"] <= pos.stop:
            fill = min(float(bar["open"]), pos.stop)       # Kurslücke: schlechterer Kurs
            return self._exit(time, fill, "Stop")
        if pos.side == -1 and bar["high"] >= pos.stop:
            fill = max(float(bar["open"]), pos.stop)
            return self._exit(time, fill, "Stop")
        return None

    def open(self, time, side, qty, price, stop, risk_eur) -> Position:
        if self._position is not None:
            raise RuntimeError("Es ist bereits eine Position offen.")
        fill = price + self._half() * side
        self._position = Position(
            side=side, qty=qty, entry_price=fill, stop=stop,
            entry_time=pd.Timestamp(time).isoformat(), risk_eur=risk_eur,
            stop_dist=abs(price - stop), financed_until=self._berlin_date(time),
        )
        return self._position

    def close(self, time, price, reason) -> Trade:
        if self._position is None:
            raise RuntimeError("Keine offene Position.")
        return self._exit(time, price, reason)

    def _exit(self, time, mid_price: float, reason: str) -> Trade:
        pos = self._position
        fill = mid_price - self._half() * pos.side
        pnl_eur = (fill - pos.entry_price) * pos.side * pos.qty / self.eurusd
        self._balance += pnl_eur
        total = pnl_eur - pos.financing_eur
        trade = Trade(
            side=pos.side, qty=pos.qty, entry_time=pos.entry_time,
            entry_price=pos.entry_price, exit_time=pd.Timestamp(time).isoformat(),
            exit_price=fill, reason=reason, pnl_eur=total,
            r=total / pos.risk_eur if pos.risk_eur > 0 else 0.0,
            financing_eur=pos.financing_eur, balance_after=self._balance,
        )
        self._position = None
        return trade
