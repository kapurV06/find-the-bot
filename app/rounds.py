"""One JSON line per finished round, for measuring how well each bot persona fools people.

Real nicknames are never logged, only the round codenames. Full transcripts are opt-in
(LOG_TRANSCRIPTS=1) because they contain what your friends typed; they are what you need to
work out why a bot got caught.
"""
import json
import logging
import os
import time

log = logging.getLogger("findthebot.rounds")


def log_round(record: dict) -> None:
    try:
        rec = {"event": "round", "ts": round(time.time()), **record}
        if os.getenv("LOG_TRANSCRIPTS") != "1":
            rec.pop("transcript", None)
        line = json.dumps(rec, ensure_ascii=False, separators=(",", ":"))
        log.info("ROUND %s", line)
        path = os.getenv("ROUND_LOG")
        if path:
            with open(path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
    except Exception:  # logging must never break a game
        log.exception("could not log round")
