"""Tests für Strategie, Papierkonto, Engine und Bot.

Ausführen mit:  python -m pytest -q
"""

import json
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

import bot
from broker import PaperBroker
from config import CONFIG
from engine import Engine
from strategy import daily_from_hourly, prepare


# ───────────────────────── Testdaten ─────────────────────────

def make_market(n: int = 6000, seed: int = 7) -> pd.DataFrame:
    """Stündlicher Goldkurs mit Trend- und Seitwärtsphasen (UTC)."""
    rng = np.random.default_rng(seed)
    drift = np.repeat(rng.normal(0, 1.2, n // 200 + 1), 200)[:n]
    close = 4000 + np.cumsum(drift + rng.normal(0, 6, n))
    open_ = np.r_[close[0], close[:-1]] + rng.normal(0, 1.5, n)
    high = np.maximum(open_, close) + np.abs(rng.normal(0, 4, n))
    low = np.minimum(open_, close) - np.abs(rng.normal(0, 4, n))
    idx = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close,
                         "volume": 1000.0}, index=idx)


@pytest.fixture(scope="module")
def market():
    return make_market()


@pytest.fixture(scope="module")
def frame(market):
    return prepare(market, daily_from_hourly(market), CONFIG)


def run_engine(frame, cfg=CONFIG):
    broker = PaperBroker(cfg)
    engine = Engine(cfg, broker)
    trades = []
    for t, row in frame.iterrows():
        trades += [e.trade for e in engine.on_bar(t, row) if e.trade is not None]
    return engine, trades


# ───────────────────────── Strategie ─────────────────────────

def test_daily_filter_has_no_lookahead(market):
    """Änderungen an späteren Tageskerzen dürfen frühere Signale nicht ändern."""
    d1 = daily_from_hourly(market)
    cut = d1.index[150]
    base = prepare(market, d1, CONFIG)
    d1_changed = d1.copy()
    d1_changed.loc[d1_changed.index >= cut, ["open", "high", "low", "close"]] *= 1.2
    changed = prepare(market, d1_changed, CONFIG)
    # Stundenkerzen, deren Handelstag spätestens 'cut' ist, sehen nur Tage vor 'cut'
    from strategy import trading_dates
    mask = trading_dates(market.index) <= cut
    cols = ["d_long", "d_short", "d_osc"]
    pd.testing.assert_frame_equal(base.loc[mask, cols], changed.loc[mask, cols])
    assert not base.loc[~mask, "d_osc"].equals(changed.loc[~mask, "d_osc"])


def test_hourly_signals_have_no_lookahead(market):
    """Signale bis Kerze k hängen nicht von Kerzen nach k ab."""
    k = 4000
    full = prepare(market, daily_from_hourly(market), CONFIG)
    part_h = market.iloc[:k]
    part = prepare(part_h, daily_from_hourly(market), CONFIG)
    cols = ["long_trig", "short_trig", "exit_long", "exit_short", "swing_lo", "swing_hi"]
    pd.testing.assert_frame_equal(full.iloc[:k][cols], part[cols])


def test_signals_exist(frame):
    v = frame[frame["valid"]]
    assert v["long_trig"].sum() > 5 and v["short_trig"].sum() > 5
    assert not (v["long_trig"] & v["short_trig"]).any()
    assert (v["d_long"] & v["d_short"]).sum() == 0


# ───────────────────────── Papierkonto ─────────────────────────

def bar(o, h, l, c):
    return {"open": o, "high": h, "low": l, "close": c}


T0 = pd.Timestamp("2026-10-12 10:00", tz="UTC")


def test_entry_and_exit_include_spread():
    b = PaperBroker(CONFIG)
    b.open(T0, 1, 1.0, 4000.0, 3970.0, 26.79)
    assert b.position().entry_price == pytest.approx(4000.2)
    tr = b.close(T0 + pd.Timedelta(hours=1), 4010.0, "test")
    assert tr.exit_price == pytest.approx(4009.8)
    assert tr.pnl_eur == pytest.approx(9.6 / 1.12)
    assert b.balance() == pytest.approx(575 + 9.6 / 1.12)


def test_stop_and_gap_fill():
    b = PaperBroker(CONFIG)
    b.open(T0, 1, 1.0, 4000.0, 3970.0, 26.79)
    assert b.sync(T0 + pd.Timedelta(hours=1), bar(3990, 3995, 3975, 3980)) is None
    tr = b.sync(T0 + pd.Timedelta(hours=2), bar(3950, 3960, 3940, 3955))  # Lücke unter den Stop
    assert tr.reason == "Stop"
    assert tr.exit_price == pytest.approx(3950 - 0.2)
    assert tr.r < -1  # Lücke macht den Verlust größer als geplant


def test_short_stop():
    b = PaperBroker(CONFIG)
    b.open(T0, -1, 0.5, 4000.0, 4030.0, 13.4)
    tr = b.sync(T0 + pd.Timedelta(hours=1), bar(4010, 4031, 4005, 4020))
    assert tr.exit_price == pytest.approx(4030 + 0.2)


def test_financing_per_night():
    b = PaperBroker(CONFIG)
    b.open(T0, 1, 1.0, 4000.0, 3900.0, 89.0)
    b.sync(T0 + pd.Timedelta(hours=5), bar(4000, 4001, 3999, 4000))        # gleicher Tag
    assert b.position().financing_eur == 0
    b.sync(pd.Timestamp("2026-10-15 08:00", tz="UTC"), bar(4000, 4001, 3999, 4000))  # 3 Nächte
    expected = 4000 / 1.12 * 0.06 / 365 * 3
    assert b.position().financing_eur == pytest.approx(expected)
    assert b.balance() == pytest.approx(575 - expected)


def test_state_roundtrip():
    b = PaperBroker(CONFIG)
    b.open(T0, -1, 0.75, 4000.0, 4030.0, 20.0)
    b2 = PaperBroker(CONFIG, json.loads(json.dumps(b.to_state())))
    assert b2.position() == b.position()
    assert b2.balance() == b.balance()


# ───────────────────────── Engine ─────────────────────────

def row(**kw):
    base = dict(valid=True, open=4000.0, high=4005.0, low=3995.0, close=4000.0,
                sha_bull=True, osc=30.0, exit_long=False, exit_short=False,
                long_trig=False, short_trig=False, d_long=True, d_short=False,
                swing_lo=3973.2, swing_hi=4030.0, in_session=True)
    base.update(kw)
    return pd.Series(base)


def test_position_size():
    """5 % von 575 € = 28,75 €; Stop 28,8 USD -> 1,11 oz bei EUR/USD 1,12."""
    eng = Engine(CONFIG, PaperBroker(CONFIG))
    ev = eng.on_bar(T0, row(long_trig=True))
    pos = eng.broker.position()
    assert ev[0].kind == "entry"
    assert pos.qty == pytest.approx(1.11)          # floor(28.75 * 1.12 / 28.8, 0.01)
    assert pos.stop == pytest.approx(3971.2)
    assert pos.risk_eur <= 28.75


def test_filter_blocks_wrong_direction():
    eng = Engine(CONFIG, PaperBroker(CONFIG))
    assert eng.on_bar(T0, row(short_trig=True, d_short=False)) == []
    assert eng.broker.position() is None


def test_daily_loss_limit():
    eng = Engine(CONFIG, PaperBroker(CONFIG))
    t = T0
    for _ in range(2):
        eng.on_bar(t, row(long_trig=True))
        t += pd.Timedelta(hours=1)
        eng.on_bar(t, row(low=3900.0, high=4001.0, open=4000.0))   # Stop
        t += pd.Timedelta(hours=1)
    assert eng.state["losses_today"] == 2
    ev = eng.on_bar(t, row(long_trig=True))
    assert ev[0].kind == "skip" and "Tageslimit" in ev[0].text
    # nächster Tag: wieder erlaubt
    ev = eng.on_bar(t + pd.Timedelta(days=1), row(long_trig=True))
    assert ev[0].kind == "entry"


def test_pause_below_limit():
    cfg = replace(CONFIG, start_balance=420.0)
    eng = Engine(cfg, PaperBroker(cfg))
    eng.on_bar(T0, row(long_trig=True))
    ev = eng.on_bar(T0 + pd.Timedelta(hours=1), row(low=3900.0, open=4000.0))
    assert any(e.kind == "pause" for e in ev)
    ev = eng.on_bar(T0 + pd.Timedelta(days=1), row(long_trig=True))
    assert ev[0].kind == "skip" and "pausiert" in ev[0].text


def test_signal_exit():
    eng = Engine(CONFIG, PaperBroker(CONFIG))
    eng.on_bar(T0, row(long_trig=True))
    ev = eng.on_bar(T0 + pd.Timedelta(hours=1), row(exit_long=True, sha_bull=False, close=4012.0))
    assert ev[0].kind == "exit" and ev[0].trade.reason == "SHA rot"
    assert ev[0].trade.pnl_eur > 0


def test_backtest_runs(frame):
    engine, trades = run_engine(frame)
    assert len(trades) > 10
    assert engine.broker.balance() == pytest.approx(trades[-1].balance_after)


# ───────────────────────── Bot (Ende zu Ende) ─────────────────────────

def test_hourly_runs_equal_backtest(tmp_path, market, monkeypatch, frame):
    """Stündliche Läufe mit gespeichertem Zustand = ein Durchlauf über alles."""
    _, bt_trades = run_engine(frame)

    monkeypatch.setattr(bot, "STATE_DIR", tmp_path)
    monkeypatch.setattr(bot, "STATE_FILE", tmp_path / "state.json")
    monkeypatch.setattr(bot, "JOURNAL_FILE", tmp_path / "journal.csv")
    monkeypatch.delenv("NTFY_TOPIC", raising=False)

    d1 = daily_from_hourly(market)
    start = 150  # erster Lauf direkt nach der Aufwärmphase
    full = prepare(market, d1, CONFIG)
    # Bot-Läufe simulieren: Datenstand wächst Kerze für Kerze (in 25er-Schritten)
    def fake_load(cfg, args, upto):
        return full.iloc[:upto], CONFIG.eurusd_fallback
    for upto in list(range(start, len(market), 25)) + [len(market)]:
        monkeypatch.setattr(bot, "load_frame", lambda cfg, args, u=upto: fake_load(cfg, args, u))
        assert bot.main([]) == 0

    journal = pd.read_csv(tmp_path / "journal.csv")
    bt = [t for t in bt_trades if pd.Timestamp(t.entry_time) > full.index[start - 1]]
    assert len(journal) == len(bt)
    np.testing.assert_allclose(journal["pnl_eur"].to_numpy(), [t.pnl_eur for t in bt], rtol=1e-6)


def test_csv_mode_and_status(tmp_path, market, monkeypatch, capsys):
    monkeypatch.setattr(bot, "STATE_DIR", tmp_path)
    monkeypatch.setattr(bot, "STATE_FILE", tmp_path / "state.json")
    monkeypatch.setattr(bot, "JOURNAL_FILE", tmp_path / "journal.csv")
    monkeypatch.delenv("NTFY_TOPIC", raising=False)
    csv = tmp_path / "h1.csv"
    out = market.copy()
    out.insert(0, "time", [int(t.timestamp()) for t in market.index])
    out.to_csv(csv, index=False)

    assert bot.main(["--h1-csv", str(csv)]) == 0          # Start
    assert "gestartet" in capsys.readouterr().out
    before = (tmp_path / "state.json").read_text()
    assert bot.main(["--h1-csv", str(csv)]) == 0          # nichts Neues
    assert "Keine neue Kerze" in capsys.readouterr().out
    assert (tmp_path / "state.json").read_text() == before  # kein unnötiger Commit
    assert bot.main(["--h1-csv", str(csv), "--status"]) == 0
    assert "Konto" in capsys.readouterr().out
    assert bot.main(["--backtest", "--h1-csv", str(csv), "--out", str(tmp_path / "bt.csv")]) == 0
    assert (tmp_path / "bt.csv").exists()
