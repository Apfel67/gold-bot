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


# ───────────────────────── Robustheit ─────────────────────────

def test_robustness_mode(tmp_path, market, monkeypatch, capsys):
    import data
    from strategy import daily_from_hourly

    def fake_download(symbol, interval, period, tries=3):
        h = make_market(4000, seed=abs(hash(symbol)) % 1000)
        if symbol == "EURUSD=X":       # anderer Preisbereich
            h[["open", "high", "low", "close"]] = h[["open", "high", "low", "close"]] / 3600
        return h if interval == "60m" else daily_from_hourly(h)

    monkeypatch.setattr(data, "_download", fake_download)
    monkeypatch.setattr(data, "drop_unfinished", lambda h1, now=None: h1)
    out = tmp_path / "rob.csv"
    assert bot.main(["--robustness", "--out", str(out)]) == 0
    table = pd.read_csv(out)
    assert len(table) == 19
    assert "Fehler" not in table or table["Fehler"].isna().all()
    assert (table["Trades"] > 0).all()
    assert table["Ø R"].abs().max() < 5
    assert "Basis (wie Live-Bot)" in capsys.readouterr().out


def test_robustness_baseline_equals_backtest(frame):
    """Die Basis-Variante rechnet in R genau wie der normale Backtest."""
    from dataclasses import replace
    from robustness import TEST_OVERRIDES, run_trades
    _, bt = run_engine(frame)
    cfg = replace(CONFIG, **TEST_OVERRIDES)
    rob = run_trades(frame, cfg)
    # R hängt nicht von der Positionsgröße ab: gleiche Trades -> gleiches R
    bt_r = {t.entry_time: t.r for t in bt}
    rob_r = {t.entry_time: t.r for t in rob}
    common = set(bt_r) & set(rob_r)
    assert len(common) >= 0.9 * len(bt)
    for k in common:
        assert rob_r[k] == pytest.approx(bt_r[k], rel=1e-6, abs=1e-9)


def test_delay_entry_shifts_signals(frame):
    from robustness import delay_entry
    d = delay_entry(frame)
    assert (d["long_trig"].iloc[1:].to_numpy() == frame["long_trig"].iloc[:-1].to_numpy()).all()
    assert not d["long_trig"].iloc[0]


# ───────────────────────── Alte Gold-Daten (history) ─────────────────────────

def make_history(start="2009-01-01", years=3, seed=11):
    n = int(years * 365 * 24)
    h = make_market(n, seed=seed)
    h.index = pd.date_range(start, periods=n, freq="h", tz="UTC")
    h[["open", "high", "low", "close"]] -= 2700          # Preisniveau um 1.300 USD
    # Wochenend-Kerzen ohne Handel wie bei Dukascopy
    weekend = h.index.dayofweek >= 5
    for col in ["open", "high", "low"]:
        h.loc[weekend, col] = h.loc[weekend, "close"]
    h.loc[weekend, "volume"] = 0.0
    return h


def test_clean_history_drops_flat_bars_and_checks_prices():
    from history import clean_history
    h = make_history(years=1)
    cleaned = clean_history(h)
    assert (cleaned.index.dayofweek < 5).all()
    bad = h.copy()
    bad[["open", "high", "low", "close"]] *= 1000       # falsches Preisformat
    with pytest.raises(RuntimeError, match="Preisformat"):
        clean_history(bad)


def test_history_mode(tmp_path, monkeypatch, capsys):
    import history
    from history import clean_history

    calls = {}

    def fake_load(start, end):
        calls["range"] = (start, end)
        return clean_history(make_history())

    monkeypatch.setattr(history, "load_dukascopy", fake_load)
    out = tmp_path / "hist"
    assert bot.main(["--history", "--von", "2010", "--bis", "2011", "--out", str(out)]) == 0
    assert calls["range"][0].year == 2009 and calls["range"][1].year == 2012

    years = pd.read_csv(out / "jahre.csv")
    assert set(years["Jahr"]) <= {2010, 2011} and len(years) >= 1
    variants = pd.read_csv(out / "varianten.csv")
    assert len(variants) == 13 and (variants["Trades"] > 0).all()
    trades = pd.read_csv(out / "trades.csv")
    assert (pd.to_datetime(trades["entry_time"], utc=True) >= pd.Timestamp("2010-01-01", tz="UTC")).all()
    assert len(trades) == variants.loc[0, "Trades"]
    assert "Ergebnis je Jahr" in capsys.readouterr().out


def test_year_range_validation():
    from history import year_range
    with pytest.raises(ValueError):
        year_range(2020, 2015)
