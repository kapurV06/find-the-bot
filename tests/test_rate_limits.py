import pytest
from starlette.websockets import WebSocketDisconnect

import app.main as main
from app import bot as botmod
from app import limits
from tests.helpers import connect, kind, phase, recv_until


def test_room_creation_is_limited_per_ip(client, monkeypatch):
    monkeypatch.setattr(limits.ROOMS, "burst", 3)
    monkeypatch.setattr(limits.ROOMS, "rate", 0)
    assert [client.post("/api/rooms").status_code for _ in range(5)] == [200, 200, 200, 429, 429]
    assert "wait" in client.post("/api/rooms").json()["detail"].lower()


def test_server_stops_creating_rooms_when_full(client, monkeypatch):
    monkeypatch.setattr(limits, "MAX_ROOMS", 2)
    assert [client.post("/api/rooms").status_code for _ in range(3)] == [200, 200, 503]


def test_connection_attempts_are_limited_per_ip(client, monkeypatch):
    monkeypatch.setattr(limits.CONNECTS, "burst", 2)
    monkeypatch.setattr(limits.CONNECTS, "rate", 0)
    code = client.post("/api/rooms").json()["code"]
    for i in range(2):
        with connect(client, code, f"p{i}", f"c{i}") as ws:
            recv_until(ws, kind("joined"))
    with connect(client, code, "p3", "c3") as ws:
        err = recv_until(ws, kind("error"))
        assert err["fatal"] and "too many" in err["text"].lower()


def test_chat_spam_is_cut_off(client, monkeypatch):
    monkeypatch.setattr(botmod, "bot_count", lambda: 0)  # humans only, so only the spammer talks
    monkeypatch.setattr(limits.CHAT, "burst", 3)
    monkeypatch.setattr(limits.CHAT, "rate", 0)
    code = client.post("/api/rooms").json()["code"]
    with connect(client, code, "spammer", "s1") as ws:
        recv_until(ws, kind("joined"))
        ws.send_json({"type": "start"})
        recv_until(ws, phase("chat"))
        for i in range(8):
            ws.send_json({"type": "chat", "text": f"msg {i}"})
        delivered = [recv_until(ws, kind("chat"))["text"] for _ in range(3)]
        assert delivered == ["msg 0", "msg 1", "msg 2"]
        for _ in range(5):
            assert "slow down" in recv_until(ws, kind("error"))["text"].lower()


def test_oversized_messages_close_the_connection(client):
    code = client.post("/api/rooms").json()["code"]
    with connect(client, code, "big", "b1") as ws:
        recv_until(ws, kind("joined"))
        ws.send_text("x" * (limits.MAX_MESSAGE_CHARS + 1))
        with pytest.raises(WebSocketDisconnect):
            while True:
                ws.receive_json()


def test_garbage_input_does_not_crash_the_connection(client):
    code = client.post("/api/rooms").json()["code"]
    with connect(client, code, "weird", "w1") as ws:
        recv_until(ws, kind("joined"))
        for junk in ("not json", "[1,2,3]", '"string"', "null", '{"type": 5}', '{"type": "nope"}'):
            ws.send_text(junk)
        ws.send_json({"type": "start"})  # still alive and still the host
        assert recv_until(ws, phase("chat"))["you"]
