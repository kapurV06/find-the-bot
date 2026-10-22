import json

import pytest
from starlette.websockets import WebSocketDisconnect

import app.main as main
from tests.helpers import connect, kind, phase, recv_until


def test_full_round_with_reconnect_and_play_again(client, fake_llm):
    code = client.post("/api/rooms").json()["code"]

    with connect(client, code, "varish", "c1") as a:
        assert recv_until(a, kind("joined"))["you"] == "varish"
        assert recv_until(a, kind("players"))["host"] == "varish"
        with connect(client, code, "rahul", "c2") as b:
            recv_until(a, lambda m: m["type"] == "players" and len(m["players"]) == 2)

            # a different client cannot take a reserved nickname
            with connect(client, code, "rahul", "someone-else") as d:
                err = recv_until(d, kind("error"))
                assert err["fatal"]

            a.send_json({"type": "start"})
            pa = recv_until(a, phase("chat"))
            pb = recv_until(b, phase("chat"))
            assert pa["you"] != pb["you"] and pa["everyone"] == pb["everyone"]
            assert len(pa["everyone"]) == 3  # two humans and one bot
            assert "varish" not in pa["everyone"] and "rahul" not in pa["everyone"]
            bot_alias = (set(pa["everyone"]) - {pa["you"], pb["you"]}).pop()

            # typing pings reach other players, and the bot shows a typing indicator before it speaks
            b.send_json({"type": "typing"})
            recv_until(a, lambda m: m["type"] == "typing" and m["nick"] == pb["you"])
            recv_until(a, lambda m: m["type"] == "typing" and m["nick"] == bot_alias)
            recv_until(a, lambda m: m["type"] == "chat" and m["nick"] == bot_alias)

            b.send_json({"type": "chat", "text": "bhai hostel ka paneer"})
            recv_until(a, lambda m: m["type"] == "chat" and m["nick"] == pb["you"])

            va = recv_until(a, phase("vote"))
            recv_until(b, phase("vote"))
            assert va["your_vote"] is None

            # votes for yourself or for nobody are ignored; varish then catches the bot
            a.send_json({"type": "vote", "alias": va["you"]})
            a.send_json({"type": "vote", "alias": "Nobody"})
            a.send_json({"type": "vote", "alias": bot_alias})
            recv_until(b, lambda m: m["type"] == "vote_progress" and m["count"] == 1)
            b.close()  # rahul's connection drops mid-vote

        # The vote must NOT end just because a player dropped. Same client id -> same alias, state restored.
        with connect(client, code, "rahul", "c2") as b2:
            recv_until(b2, kind("joined"))
            recv_until(b2, kind("reset"))
            history = []
            while True:
                m = b2.receive_json()
                if m["type"] == "chat":
                    history.append(m)
                elif m.get("phase") == "vote":
                    assert m["you"] == pb["you"]
                    break
                else:
                    assert m.get("phase") != "result", "vote ended early when a player dropped"
            assert any(x["text"] == "bhai hostel ka paneer" for x in history)
            b2.send_json({"type": "vote", "alias": pa["you"]})

            res = recv_until(a, phase("result"))
            assert res["bots"][0]["alias"] == bot_alias
            assert res["humans"] == {pa["you"]: "varish", pb["you"]: "rahul"}
            assert res["caught"] == [pa["you"]]
            assert (res["bots"][0]["fooled"], res["bots"][0]["total"]) == (1, 2)
            assert main.PERSONA_STATS[res["bots"][0]["label"]]["rounds"] == 1

            b2.send_json({"type": "again"})
            recv_until(a, kind("lobby"))
            roster = recv_until(a, kind("players"))
            assert sorted(roster["players"]) == ["rahul", "varish"] and roster["host"] == "varish"
            a.send_json({"type": "start"})
            assert len(recv_until(a, phase("chat"))["everyone"]) == 3


def test_solo_round_against_the_bot(client, fake_llm):
    code = client.post("/api/rooms").json()["code"]
    with connect(client, code, "solo", "s1") as s:
        recv_until(s, kind("joined"))
        s.send_json({"type": "start"})
        p = recv_until(s, phase("chat"))
        assert len(p["everyone"]) == 2
        recv_until(s, phase("vote"))
        s.send_json({"type": "vote", "alias": next(x for x in p["everyone"] if x != p["you"])})
        assert recv_until(s, phase("result"))["caught"] == [p["you"]]


def test_unknown_room_is_refused(client):
    with connect(client, "ZZZZ", "y", "z") as e:
        err = recv_until(e, kind("error"))
        assert err["fatal"] and "not found" in err["text"].lower()


def test_round_log_never_contains_real_nicknames(client, fake_llm, monkeypatch, tmp_path):
    log_file = tmp_path / "rounds.jsonl"
    monkeypatch.setenv("ROUND_LOG", str(log_file))
    monkeypatch.setenv("LOG_TRANSCRIPTS", "1")
    code = client.post("/api/rooms").json()["code"]
    with connect(client, code, "uniquenick", "s1") as s:
        recv_until(s, kind("joined"))
        s.send_json({"type": "start"})
        p = recv_until(s, phase("chat"))
        s.send_json({"type": "chat", "text": "hello there"})
        recv_until(s, phase("vote"))
        s.send_json({"type": "vote", "alias": next(x for x in p["everyone"] if x != p["you"])})
        recv_until(s, phase("result"))

    line = log_file.read_text().strip()
    rec = json.loads(line)
    assert rec["event"] == "round" and rec["players"] == 1
    assert rec["bots"][0]["label"] and rec["bots"][0]["total"] == 1
    assert "uniquenick" not in line
    assert any(t["bot"] for t in rec["transcript"]) and any(not t["bot"] for t in rec["transcript"])


def test_transcripts_are_off_by_default(client, fake_llm, monkeypatch, tmp_path):
    log_file = tmp_path / "rounds.jsonl"
    monkeypatch.setenv("ROUND_LOG", str(log_file))
    monkeypatch.delenv("LOG_TRANSCRIPTS", raising=False)
    code = client.post("/api/rooms").json()["code"]
    with connect(client, code, "solo", "s1") as s:
        recv_until(s, kind("joined"))
        s.send_json({"type": "start"})
        p = recv_until(s, phase("chat"))
        recv_until(s, phase("vote"))
        s.send_json({"type": "vote", "alias": next(x for x in p["everyone"] if x != p["you"])})
        recv_until(s, phase("result"))
    assert "transcript" not in json.loads(log_file.read_text())


def test_health_and_install_files_are_served(client):
    h = client.get("/health").json()
    assert h["ok"] and h["bots"] is True
    assert client.get("/").status_code == 200
    manifest = client.get("/manifest.webmanifest")
    assert manifest.headers["content-type"].startswith("application/manifest+json")
    assert manifest.json()["display"] == "standalone"
    for icon in ("/icon-192.png", "/icon-512.png", "/icon-maskable-512.png", "/apple-touch-icon.png"):
        r = client.get(icon)
        assert r.status_code == 200 and r.headers["content-type"] == "image/png"
