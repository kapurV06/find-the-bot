"""Websocket test helpers. recv_until buffers skipped messages so nothing is lost between waits."""

BUF: dict[int, list] = {}


def recv_until(ws, pred, limit=400):
    buf = BUF.setdefault(id(ws), [])
    for i, m in enumerate(buf):
        if pred(m):
            del buf[i]
            return m
    for _ in range(limit):
        m = ws.receive_json()
        if pred(m):
            return m
        buf.append(m)
    raise AssertionError("expected message never arrived")


def phase(name):
    return lambda m: m.get("phase") == name


def kind(name):
    return lambda m: m.get("type") == name


def connect(client, code, nick, cid):
    return client.websocket_connect(f"/ws/{code}?nick={nick}&cid={cid}")
