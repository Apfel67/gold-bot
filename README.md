# Gold-Bot (SHA + Kalman)

Ein Trading-Bot für Gold (XAU/USD), der die Strategie aus meinem
TradingView-Indikator „SHA + Kalman Trading-System“ automatisch umsetzt –
vorerst nur auf einem **Papierkonto mit Spielgeld**.

- **Signale:** Smoothed Heikin Ashi (Richtung) + Kalman Trend Strength
  Oscillator (Filter), Tageschart als Richtungsfilter, Einstieg im 1h-Chart
- **Risiko:** 5 % des Kontostands pro Trade, Stop hinter dem letzten Hoch/Tief,
  max. 2 Verlusttrades pro Tag, Pause unter 400 €
- **Kosten im Papierkonto:** Spread, Kurslücken beim Stop, Übernacht-Finanzierung
- **Benachrichtigung:** Push aufs Handy über [ntfy](https://ntfy.sh)
- **Betrieb:** stündlich und kostenlos über GitHub Actions

> Kein Anlagerat. Der Bot ist ein Lern- und Portfolio-Projekt.

## Aufbau

| Datei | Aufgabe |
|---|---|
| `indicators.py` | Exakte Python-Portierung beider TradingView-Indikatoren |
| `strategy.py` | Tagesfilter und Signale – identisch zum Pine-Skript, ohne Blick in die Zukunft |
| `broker.py` | Konto-Schnittstelle `Broker` und das Papierkonto `PaperBroker` |
| `engine.py` | Checkliste Kerze für Kerze: Stop, Ausstieg, Einstieg, Positionsgröße, Regeln |
| `bot.py` | Hauptprogramm: Daten laden, neue Kerzen verarbeiten, Zustand speichern, Push |
| `data.py` | Kursdaten von Yahoo Finance (Gold-Futures GC=F) oder aus CSV |
| `notifier.py` | Push-Nachrichten über ntfy |
| `config.py` | Alle Einstellungen an einer Stelle |

Backtest und Live-Bot nutzen **dieselbe Engine**. Ein Test stellt sicher,
dass stündliche Läufe mit gespeichertem Zustand exakt dieselben Trades
ergeben wie ein Durchlauf über die ganze Historie.

Für ein echtes Demokonto (z. B. Capital.com) wird nur eine neue Klasse
geschrieben, die `Broker` umsetzt. Strategie und Engine bleiben unverändert.

## Einrichtung

### 1. Push-Nachrichten aufs iPhone

1. App **ntfy** aus dem App Store installieren und Mitteilungen erlauben.
2. In der App auf **+** tippen und ein Thema abonnieren, z. B.
   `goldbot-` gefolgt von einer langen Zufallsfolge
   (Server: Standard `ntfy.sh`).
3. Den Themennamen geheim halten: Wer ihn kennt, kann mitlesen.

### 2. GitHub

1. Neues Repository anlegen und alle `.py`-Dateien, `requirements.txt`
   und diese README hochladen.
2. `gold-bot.yml` und `tests.yml` in den Ordner `.github/workflows/` legen.
3. Unter **Settings → Secrets and variables → Actions** ein Secret
   `NTFY_TOPIC` mit deinem Themennamen anlegen.
4. Unter **Actions → gold-bot → Run workflow** einmal `test-push` starten,
   dann einmal `run`. Der erste Lauf meldet „Gold-Bot gestartet“; ab der
   nächsten Stunde handelt er.

## Bedienung

Über **Actions → gold-bot → Run workflow**:

| Modus | Wirkung |
|---|---|
| `run` | normaler Lauf (passiert sonst automatisch jede Stunde) |
| `status` | Kontostand, offene Position und Tagesfilter als Push |
| `test-push` | Testnachricht |
| `resume` | Pause nach Unterschreiten der 400-€-Grenze aufheben |
| `backtest` | Engine über ca. 2 Jahre Stundendaten; Trades als Download |
| `robustness` | Backtest mit höheren Kosten, veränderten Parametern und auf 6 anderen Märkten; Tabelle als Download |

Zustand und Trade-Journal liegen in `state/state.json` und `state/journal.csv`.

Lokal:

```bash
pip install -r requirements.txt pytest
python -m pytest -q          # Tests
python bot.py --backtest     # Backtest mit Yahoo-Daten
python bot.py --backtest --h1-csv stunden.csv   # mit eigenen Daten
```

## Grenzen

- Yahoo liefert Gold-**Futures**, nicht den Spotkurs von FOREX.com im
  TradingView-Chart. Signale können sich daher um eine Kerze verschieben.
- GitHub startet geplante Läufe oft einige Minuten verspätet.
- Das Papierkonto schätzt die Ausführung. Echte Fills, Slippage und
  Finanzierungssätze kommen erst mit einem Demokonto beim Broker.

## Lizenz

Der Kalman-Teil ist abgeleitet von „Adaptive Kalman filter – Trend Strength
Oscillator“ © Zeiierman und steht wie das Original unter
[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).
Dieses Projekt steht unter derselben Lizenz.
