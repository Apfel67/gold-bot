"""Push-Nachrichten aufs Handy über ntfy (https://ntfy.sh).

Einrichtung iPhone: App "ntfy" aus dem App Store installieren, auf "+" tippen
und dasselbe Thema abonnieren, das als Umgebungsvariable NTFY_TOPIC gesetzt
ist. Das Thema wirkt wie ein Passwort: lang und zufällig wählen.

Ohne NTFY_TOPIC werden die Nachrichten nur in der Konsole ausgegeben.
"""

from __future__ import annotations

import json
import os
import urllib.request

PRIORITY = {"entry": 4, "exit": 4, "pause": 5, "error": 4, "info": 3, "skip": 2}
TAGS = {"entry": ["chart_with_upwards_trend"], "exit": ["checkered_flag"],
        "pause": ["warning"], "error": ["rotating_light"], "info": ["robot"],
        "skip": ["fast_forward"]}


def send(title: str, message: str, kind: str = "info") -> bool:
    topic = os.environ.get("NTFY_TOPIC", "").strip()
    print(f"[{kind}] {title}\n{message}\n")
    if not topic:
        return False
    server = os.environ.get("NTFY_SERVER", "https://ntfy.sh").rstrip("/")
    payload = {
        "topic": topic,
        "title": title,
        "message": message,
        "priority": PRIORITY.get(kind, 3),
        "tags": TAGS.get(kind, []),
    }
    req = urllib.request.Request(server, data=json.dumps(payload).encode("utf-8"),
                                 headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return 200 <= resp.status < 300
    except Exception as e:  # Push-Fehler dürfen den Bot nicht stoppen
        print(f"Push fehlgeschlagen: {e}")
        return False
