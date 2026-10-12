"""Engine: setzt die Checkliste Kerze für Kerze um.

Dieselbe Engine läuft im Live-Bot (eine neue Kerze pro Stunde) und im
Backtest (alle historischen Kerzen nacheinander). Dadurch sind Backtest und
Bot garantiert gleich.

Ablauf je abgeschlossener Stundenkerze:
  1. Brokerseite: Stop ausgelöst? Finanzierung über Nacht?
  2. Ausstieg: SHA-Farbwechsel oder Oszillator über die Nulllinie
  3. Einstieg: frisches Signal + Tagesfilter + Uhrzeit + Tageslimit
     + Kontostand über der Pause-Grenze + Mindestgröße
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from broker import Broker, Trade
from config import Config

SIDE_NAME = {1: "LONG", -1: "SHORT"}


@dataclass
class Event:
    kind: str   # "entry", "exit", "skip", "pause"
    time: str
    title: str
    text: str
    trade: Optional[Trade] = None


def default_engine_state() -> dict:
    return {
        "day": "", "losses_today": 0, "paused": False,
        "trades": 0, "wins": 0, "sum_r": 0.0, "loss_streak": 0, "max_streak": 0,
        "min_balance": None,
    }


class Engine:
    def __init__(self, cfg: Config, broker: Broker, state: Optional[dict] = None):
        self.cfg = cfg
        self.broker = broker
        self.state = default_engine_state()
        self.state.update(state or {})
        if self.state["min_balance"] is None:
            self.state["min_balance"] = broker.balance()

    # ── Öffentlich ────────────────────────────────────────
    def on_bar(self, time: pd.Timestamp, row) -> list[Event]:
        if not row["valid"]:
            return []
        events: list[Event] = []
        self._new_day(time)

        # 1. Brokerseite
        trade = self.broker.sync(time, row)
        if trade:
            events += self._record(trade)

        # 2. Ausstieg nach Signal
        pos = self.broker.position()
        if pos is not None:
            want_exit = row["exit_long"] if pos.side == 1 else row["exit_short"]
            if want_exit:
                if pos.side == 1:
                    reason = "SHA rot" if not row["sha_bull"] else "Oszillator < 0"
                else:
                    reason = "SHA grün" if row["sha_bull"] else "Oszillator > 0"
                events += self._record(self.broker.close(time, float(row["close"]), reason))

        # 3. Einstieg
        if self.broker.position() is None:
            for side, trig, allowed in ((1, row["long_trig"], row["d_long"]),
                                        (-1, row["short_trig"], row["d_short"])):
                if trig and allowed:
                    events += self._try_entry(time, row, side)
                    break
        return events

    def summary(self) -> dict:
        s = self.state
        n = s["trades"]
        return {
            "trades": n,
            "win_rate": s["wins"] / n if n else None,
            "avg_r": s["sum_r"] / n if n else None,
            "max_loss_streak": s["max_streak"],
            "balance": self.broker.balance(),
            "min_balance": s["min_balance"],
            "paused": s["paused"],
        }

    # ── Intern ────────────────────────────────────────────
    def _new_day(self, time: pd.Timestamp) -> None:
        t = time if time.tzinfo else time.tz_localize("UTC")
        day = t.tz_convert(self.cfg.timezone).date().isoformat()
        if day != self.state["day"]:
            self.state["day"] = day
            self.state["losses_today"] = 0

    def _try_entry(self, time, row, side: int) -> list[Event]:
        cfg, s = self.cfg, self.state
        when = pd.Timestamp(time).isoformat()
        name = SIDE_NAME[side]
        if s["paused"]:
            return [Event("skip", when, f"{name}-Signal ausgelassen",
                          f"Bot pausiert (Konto unter {cfg.pause_below:.0f} €).")]
        if not row["in_session"]:
            return []
        if s["losses_today"] >= cfg.max_losses_per_day:
            return [Event("skip", when, f"{name}-Signal ausgelassen",
                          f"Tageslimit erreicht ({s['losses_today']} Verluste heute).")]

        price = float(row["close"])
        stop = (row["swing_lo"] - cfg.stop_buffer) if side == 1 else (row["swing_hi"] + cfg.stop_buffer)
        dist = abs(price - stop)
        if not math.isfinite(dist) or dist <= 0:
            return []
        risk_eur = self.broker.balance() * cfg.risk_pct / 100.0
        eurusd = self.broker.eurusd
        qty = math.floor(risk_eur * eurusd / dist / cfg.qty_step + 1e-9) * cfg.qty_step
        qty = round(qty, 6)
        if qty < cfg.min_qty:
            return [Event("skip", when, f"{name}-Signal ausgelassen",
                          f"Größe {qty:.2f} oz unter Minimum {cfg.min_qty:.2f} oz (Stop {dist:.1f} USD).")]

        actual_risk = qty * dist / eurusd
        pos = self.broker.open(time, side, qty, price, float(stop), actual_risk)
        lev = qty * price / eurusd / self.broker.balance()
        text = (f"Einstieg {pos.entry_price:.2f}\n"
                f"Stop {stop:.2f} ({dist:.1f} USD)\n"
                f"Größe {qty:.2f} oz · Hebel {lev:.1f}×\n"
                f"Risiko {actual_risk:.2f} € · Konto {self.broker.balance():.2f} €")
        return [Event("entry", when, f"{name} Gold eröffnet (Papier)", text)]

    def _record(self, trade: Trade) -> list[Event]:
        s, cfg = self.state, self.cfg
        s["trades"] += 1
        s["sum_r"] += trade.r
        if trade.r > 0:
            s["wins"] += 1
            s["loss_streak"] = 0
        else:
            s["losses_today"] += 1
            s["loss_streak"] += 1
            s["max_streak"] = max(s["max_streak"], s["loss_streak"])
        s["min_balance"] = min(s["min_balance"], trade.balance_after)

        sign = "+" if trade.pnl_eur >= 0 else ""
        text = (f"{trade.reason}: {trade.entry_price:.2f} → {trade.exit_price:.2f}\n"
                f"Ergebnis {sign}{trade.pnl_eur:.2f} € ({sign}{trade.r:.2f} R)\n"
                f"Konto {trade.balance_after:.2f} €")
        events = [Event("exit", trade.exit_time,
                        f"{SIDE_NAME[trade.side]} geschlossen: {sign}{trade.pnl_eur:.2f} €",
                        text, trade)]

        if not s["paused"] and trade.balance_after < cfg.pause_below:
            s["paused"] = True
            events.append(Event("pause", trade.exit_time, "Bot pausiert",
                                f"Konto {trade.balance_after:.2f} € unter {cfg.pause_below:.0f} €. "
                                "Keine neuen Trades, bis du ihn mit --resume freigibst."))
        return events
