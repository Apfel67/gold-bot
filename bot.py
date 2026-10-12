"""Gold-Bot: handelt die SHA+Kalman-Strategie auf einem Papierkonto.
 
Aufrufe:
    python bot.py                 normaler Lauf (stündlich per GitHub Actions)
    python bot.py --status        Kontostand und Lage als Push schicken
    python bot.py --test-push     Testnachricht aufs Handy
    python bot.py --resume        Pause aufheben (nach Unterschreiten der Grenze)
    python bot.py --backtest      dieselbe Engine über die ganze Historie laufen lassen
 
Offline mit eigenen Daten (z. B. TradingView-Export):
    python bot.py --backtest --h1-csv stunden.csv [--d1-csv tage.csv]
 
Zustand und Journal liegen im Ordner state/ (änderbar mit BOT_STATE_DIR).
"""
 
from __future__ import annotations
 
import argparse
import json
import os
import sys
import traceback
from dataclasses import asdict
from pathlib import Path
 
import pandas as pd
 
import data
import notifier
from broker import PaperBroker
from config import CONFIG, Config
from engine import SIDE_NAME, Engine
from strategy import prepare
 
STATE_DIR = Path(os.environ.get("BOT_STATE_DIR", "state"))
STATE_FILE = STATE_DIR / "state.json"
JOURNAL_FILE = STATE_DIR / "journal.csv"
ERROR_PUSH_HOURS = 6
 
 
# ───────────────────────── Zustand ─────────────────────────
 
def load_state() -> dict:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    return {}
 
 
def save_state(state: dict) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = STATE_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(STATE_FILE)
 
 
def append_journal(trades: list, path: Path | None = None) -> None:
    if not trades:
        return
    path = path or JOURNAL_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = pd.DataFrame([asdict(t) for t in trades])
    rows.insert(0, "side_name", rows["side"].map(SIDE_NAME))
    rows.to_csv(path, mode="a", header=not path.exists(), index=False)
 
 
# ───────────────────────── Daten ─────────────────────────
 
def load_frame(cfg: Config, args) -> tuple[pd.DataFrame, float]:
    if args.h1_csv:
        h1 = data.load_csv(args.h1_csv)
        if args.d1_csv:
            d1 = data.load_csv(args.d1_csv)
        else:
            from strategy import daily_from_hourly
            d1 = daily_from_hourly(h1)
        eurusd = cfg.eurusd_fallback
    else:
        h1, d1, eurusd = data.load_yahoo(cfg)
    return prepare(h1, d1, cfg), eurusd
 
 
# ───────────────────────── Texte ─────────────────────────
 
def status_text(engine: Engine, frame: pd.DataFrame) -> str:
    b = engine.broker
    last = frame.iloc[-1]
    s = engine.summary()
    lines = [f"Konto {s['balance']:.2f} € (Start {engine.cfg.start_balance:.0f} €)"]
    pos = b.position()
    if pos:
        upnl = (last["close"] - pos.entry_price) * pos.side * pos.qty / b.eurusd - pos.financing_eur
        lines.append(f"Offen: {SIDE_NAME[pos.side]} {pos.qty:.2f} oz ab {pos.entry_price:.2f}, "
                     f"Stop {pos.stop:.2f}, aktuell {upnl:+.2f} €")
    else:
        lines.append("Keine offene Position")
    filt = "nur Long" if last["d_long"] else "nur Short" if last["d_short"] else "kein Trade"
    lines.append(f"Tagesfilter: {filt} · Gold {last['close']:.2f}")
    if s["trades"]:
        lines.append(f"{s['trades']} Trades · Treffer {s['win_rate']:.0%} · Ø {s['avg_r']:+.2f} R")
    if s["paused"]:
        lines.append("PAUSIERT")
    return "\n".join(lines)
 
 
# ───────────────────────── Modi ─────────────────────────
 
def run_live(cfg: Config, args) -> int:
    state = load_state()
    if args.resume:
        state.setdefault("engine", {})["paused"] = False
        save_state(state)
        notifier.send("Bot wieder aktiv", "Pause aufgehoben, neue Signale werden gehandelt.", "info")
        return 0
 
    frame, eurusd = load_frame(cfg, args)
    broker = PaperBroker(cfg, state.get("broker"), eurusd=eurusd)
    engine = Engine(cfg, broker, state.get("engine"))
    latest = frame.index[-1]
 
    if args.status:
        notifier.send("Gold-Bot Status", status_text(engine, frame), "info")
        return 0
 
    events, trades = [], []
    last_bar = state.get("last_bar")
    if last_bar is None:
        notifier.send("Gold-Bot gestartet (Papierkonto)",
                      status_text(engine, frame) + "\nAb der nächsten Stundenkerze wird gehandelt.", "info")
    else:
        new = frame[frame.index > pd.Timestamp(last_bar)]
        if new.empty:
            # Keine neue abgeschlossene Kerze: nichts tun, Zustand unverändert lassen
            print(f"Keine neue Kerze seit {last_bar}.")
            return 0
        if len(new) > cfg.max_catchup_bars:
            notifier.send("Kerzen übersprungen",
                          f"{len(new) - cfg.max_catchup_bars} ältere Kerzen nicht nachgeholt.", "info")
            new = new.iloc[-cfg.max_catchup_bars:]
        for t, row in new.iterrows():
            for ev in engine.on_bar(t, row):
                events.append(ev)
                if ev.trade is not None:
                    trades.append(ev.trade)
 
    state.update({
        "last_bar": latest.isoformat(),
        "broker": broker.to_state(),
        "engine": engine.state,
    })
    append_journal(trades)
    save_state(state)
 
    for ev in events:
        notifier.send(ev.title, ev.text, ev.kind)
    print(f"OK · letzte Kerze {latest} · {len(events)} Ereignisse · Konto {broker.balance():.2f} €")
    return 0
 
 
def run_backtest(cfg: Config, args) -> int:
    frame, eurusd = load_frame(cfg, args)
    broker = PaperBroker(cfg, eurusd=eurusd)
    engine = Engine(cfg, broker)
    trades = []
    for t, row in frame.iterrows():
        for ev in engine.on_bar(t, row):
            if ev.trade is not None:
                trades.append(ev.trade)
    s = engine.summary()
    valid = frame[frame["valid"]]
    print(f"Zeitraum: {valid.index[0]} bis {valid.index[-1]} ({len(valid)} Stundenkerzen)")
    print(f"Trades: {s['trades']}")
    if s["trades"]:
        losses = [t.r for t in trades if t.r <= 0]
        wins = [t.r for t in trades if t.r > 0]
        print(f"Trefferquote: {s['win_rate']:.1%}")
        print(f"Ø R pro Trade (nach Kosten): {s['avg_r']:+.3f}")
        print(f"Ø Gewinn: {sum(wins) / max(len(wins), 1):+.2f} R | Ø Verlust: {sum(losses) / max(len(losses), 1):+.2f} R")
        print(f"Längste Verlustserie: {s['max_loss_streak']}")
        print(f"Konto: {cfg.start_balance:.2f} € → {s['balance']:.2f} € (tiefster Stand {s['min_balance']:.2f} €)")
        print(f"Pausiert: {'ja' if s['paused'] else 'nein'}")
    out = Path(args.out or "backtest_trades.csv")
    if out.exists():
        out.unlink()
    append_journal(trades, out)
    print(f"Trades gespeichert in {out}")
    return 0
 
 
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backtest", action="store_true")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--test-push", action="store_true")
    ap.add_argument("--h1-csv")
    ap.add_argument("--d1-csv")
    ap.add_argument("--out", help="Datei für Backtest-Trades")
    args = ap.parse_args(argv)
    cfg = CONFIG
 
    if args.test_push:
        ok = notifier.send("Gold-Bot Test", "Wenn du das liest, funktionieren die Push-Nachrichten.", "info")
        return 0 if ok else 1
 
    try:
        return run_backtest(cfg, args) if args.backtest else run_live(cfg, args)
    except Exception as e:
        traceback.print_exc()
        if not args.backtest:
            state = load_state()
            last = state.get("last_error_push")
            now = pd.Timestamp.now(tz="UTC")
            if last is None or now - pd.Timestamp(last) > pd.Timedelta(hours=ERROR_PUSH_HOURS):
                notifier.send("Gold-Bot Fehler", f"{type(e).__name__}: {e}", "error")
                state["last_error_push"] = now.isoformat()
                save_state(state)
        return 1
 
 
if __name__ == "__main__":
    sys.exit(main())
