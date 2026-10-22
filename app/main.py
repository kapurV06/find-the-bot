import asyncio
import json
import logging
import os
import random
import string
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from . import bot as botmod
from . import limits, rounds

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("findthebot")
if not botmod.bots_enabled():
    log.warning("GROQ_API_KEY not set: rooms will have humans only, no bot")

app = FastAPI(title="Find the Bot")
STATIC = Path(__file__).resolve().parent.parent / "static"

CHAT_SECONDS = int(os.getenv("CHAT_SECONDS", "180"))
VOTE_SECONDS = int(os.getenv("VOTE_SECONDS", "25"))
ROOM_GRACE = int(os.getenv("ROOM_GRACE", "60"))  # keep an emptied room alive for reconnects
MAX_PLAYERS = 7

PROMPTS = [
    "Describe your worst hostel food experience.",
    "What's the dumbest reason you've bunked a class?",
    "Defend your most unpopular food opinion.",
    "What did you do the night before your last exam?",
    "Pitch a startup idea that is clearly a bad idea.",
    "What's the most useless thing you've bought this semester?",
    "Which teacher would win in a fight, and why?",
]

# Everyone (humans and bots) gets a random alias each round, so nobody can spot the bot
# just because its name isn't in the friend group.
ALIASES = [
    "Mango", "Falcon", "Pixel", "Biscuit", "Comet", "Noodle", "Panda", "Ginger", "Rocket", "Waffle",
    "Otter", "Cactus", "Turbo", "Pepper", "Zebra", "Nimbus", "Tofu", "Maverick", "Sundae", "Jelly",
    "Ember", "Pickle", "Koala", "Blaze", "Mocha", "Orbit", "Sprout", "Yeti", "Marble", "Fudge",
]

# How often each bot persona fools people (in memory, resets on restart).
PERSONA_STATS: dict[str, dict] = {}


def min_players() -> int:
    return 1 if botmod.bots_enabled() else 2  # solo testing against the bot is allowed


class Room:
    def __init__(self, code: str):
        self.code = code
        self.players: dict[str, WebSocket] = {}  # real nickname -> live socket
        self.cids: dict[str, str] = {}           # real nickname -> client id (reserves the name)
        self.host: str | None = None
        self.phase = "lobby"  # lobby -> chat -> vote -> result -> lobby
        self.prompt = ""
        self.ends_at = 0.0
        self.history: list[dict] = []
        self.alias: dict[str, str] = {}   # real nickname -> round alias
        self.everyone: list[str] = []     # all aliases, shuffled
        self.votes: dict[str, str] = {}   # real nickname -> alias voted for
        self.result: dict | None = None
        self.bots: list[botmod.Bot] = []
        self.timer: asyncio.Task | None = None
        self.bot_tasks: list[asyncio.Task] = []
        self.cleanup: asyncio.Task | None = None

    # ----- messaging -----

    def roster(self) -> dict:
        return {"type": "players", "players": list(self.players), "host": self.host}

    async def broadcast(self, msg: dict, skip: str | None = None) -> None:
        for nick, ws in list(self.players.items()):
            if nick == skip:
                continue
            try:
                await ws.send_json(msg)
            except Exception:
                pass  # the disconnect handler cleans up

    async def post(self, alias: str, text: str) -> None:
        msg = {"type": "chat", "nick": alias, "text": text, "ts": time.time()}
        self.history.append(msg)
        await self.broadcast(msg)
        for b in self.bots:
            if b.alias != alias:
                b.wake.set()

    async def typing(self, alias: str, skip: str | None = None) -> None:
        await self.broadcast({"type": "typing", "nick": alias}, skip=skip)

    def _remaining(self) -> float:
        return max(0.0, round(self.ends_at - time.time(), 1))

    def chat_msg(self, nick: str) -> dict:
        return {
            "type": "phase", "phase": "chat", "prompt": self.prompt, "seconds": self._remaining(),
            "total": CHAT_SECONDS, "you": self.alias[nick], "everyone": self.everyone,
        }

    def vote_msg(self, nick: str) -> dict:
        return {
            "type": "phase", "phase": "vote", "seconds": self._remaining(), "total": VOTE_SECONDS,
            "you": self.alias[nick], "everyone": self.everyone, "your_vote": self.votes.get(nick),
            "count": len(self.votes), "voters": len(self.alias),
        }

    async def send_state(self, nick: str, ws: WebSocket) -> None:
        """Bring a (re)connected player up to date."""
        await ws.send_json({"type": "reset"})
        if self.phase == "lobby":
            await ws.send_json(self.roster())
            return
        if self.phase == "chat":
            await ws.send_json(self.chat_msg(nick))
        for m in self.history:
            await ws.send_json(m)
        if self.phase == "vote":
            await ws.send_json(self.vote_msg(nick))
        elif self.phase == "result" and self.result:
            await ws.send_json({**self.result, "instant": True})

    # ----- round lifecycle -----

    async def start_round(self) -> None:
        humans = list(self.players)
        n_bots = botmod.bot_count()
        pool = random.sample(ALIASES, len(humans) + n_bots)
        self.alias = dict(zip(humans, pool))
        self.bots = [botmod.Bot(a) for a in pool[len(humans):]]
        self.everyone = pool[:]
        random.shuffle(self.everyone)
        self.history, self.votes, self.result = [], {}, None
        self.phase = "chat"
        self.prompt = random.choice(PROMPTS)
        self.ends_at = time.time() + CHAT_SECONDS
        for nick, ws in list(self.players.items()):
            try:
                await ws.send_json(self.chat_msg(nick))
            except Exception:
                pass
        self.timer = asyncio.create_task(self._chat_timeout())
        self.bot_tasks = [asyncio.create_task(b.run(self)) for b in self.bots]

    async def _chat_timeout(self) -> None:
        await asyncio.sleep(CHAT_SECONDS)
        await self.start_vote()

    async def start_vote(self) -> None:
        if self.phase != "chat":
            return
        self.phase = "vote"
        for t in self.bot_tasks:
            t.cancel()
        self.ends_at = time.time() + VOTE_SECONDS
        self.votes = {}
        for nick, ws in list(self.players.items()):
            try:
                await ws.send_json(self.vote_msg(nick))
            except Exception:
                pass
        self.timer = asyncio.create_task(self._vote_timeout())

    async def _vote_timeout(self) -> None:
        await asyncio.sleep(VOTE_SECONDS)
        await self.finish_vote()

    async def cast_vote(self, nick: str, target: str) -> None:
        if self.phase != "vote" or nick in self.votes:
            return
        if target not in self.everyone or target == self.alias.get(nick):
            return
        self.votes[nick] = target
        await self.broadcast({"type": "vote_progress", "count": len(self.votes), "voters": len(self.alias)})
        # Wait for every human in the round, connected or not: a dropped phone gets until the timer to rejoin.
        if all(n in self.votes for n in self.alias):
            await self.finish_vote()

    async def finish_vote(self) -> None:
        if self.phase != "vote":
            return
        self.phase = "result"
        if self.timer and self.timer is not asyncio.current_task():
            self.timer.cancel()
        bot_aliases = {b.alias for b in self.bots}
        humans = {alias: nick for nick, alias in self.alias.items()}
        votes = {self.alias[n]: t for n, t in self.votes.items()}
        caught = [a for a, t in votes.items() if t in bot_aliases]
        total = len(self.alias)
        bots_out = []
        for b in self.bots:
            fooled = total - sum(1 for t in votes.values() if t == b.alias)
            st = PERSONA_STATS.setdefault(b.label, {"rounds": 0, "fooled": 0, "total": 0})
            st["rounds"] += 1
            st["fooled"] += fooled
            st["total"] += total
            bots_out.append({
                "alias": b.alias, "label": b.label, "fooled": fooled, "total": total,
                "all_time": {"rounds": st["rounds"], "rate": round(100 * st["fooled"] / st["total"])},
            })
        rounds.log_round({
            "room": self.code, "players": total, "voted": len(votes), "caught_by": len(caught),
            "messages": len(self.history), "bot_messages": sum(1 for m in self.history if m["nick"] in bot_aliases),
            "chat_seconds": CHAT_SECONDS, "model": botmod.MODEL,
            "bots": [{k: b[k] for k in ("label", "alias", "fooled", "total")} for b in bots_out],
            "transcript": [{"who": m["nick"], "bot": m["nick"] in bot_aliases, "text": m["text"]} for m in self.history],
        })
        self.result = {
            "type": "phase", "phase": "result", "bots": bots_out, "humans": humans,
            "votes": votes, "caught": caught, "everyone": self.everyone,
        }
        await self.broadcast(self.result)

    async def play_again(self) -> None:
        if self.phase != "result":
            return
        self.stop()
        self.cids = {n: c for n, c in self.cids.items() if n in self.players}
        self.phase = "lobby"
        self.history, self.alias, self.everyone = [], {}, []
        self.bots, self.votes, self.result = [], {}, None
        if self.host not in self.players:
            self.host = next(iter(self.players), None)
        await self.broadcast({"type": "lobby"})
        await self.broadcast(self.roster())

    def stop(self) -> None:
        if self.timer and self.timer is not asyncio.current_task():
            self.timer.cancel()
        for t in self.bot_tasks:
            t.cancel()
        self.bot_tasks = []

    def schedule_cleanup(self) -> None:
        async def later():
            await asyncio.sleep(ROOM_GRACE)
            if not self.players:
                self.stop()
                rooms.pop(self.code, None)
        if self.cleanup:
            self.cleanup.cancel()
        self.cleanup = asyncio.create_task(later())


rooms: dict[str, Room] = {}


def new_code() -> str:
    while True:
        code = "".join(random.choices(string.ascii_uppercase, k=4))
        if code not in rooms:
            return code


def client_ip(conn) -> str:
    return conn.client.host if conn.client else "unknown"


@app.post("/api/rooms")
async def create_room(request: Request):
    if not limits.ROOMS.allow(client_ip(request)):
        raise HTTPException(429, "Too many rooms created. Wait a minute and try again.")
    if len(rooms) >= limits.MAX_ROOMS:
        raise HTTPException(503, "The server is busy. Try again in a few minutes.")
    code = new_code()
    rooms[code] = Room(code)
    return {"code": code}


@app.get("/health")
async def health():
    return {
        "ok": True, "rooms": len(rooms), "bots": botmod.bots_enabled(),
        "model": botmod.MODEL, "min_players": min_players(), "persona_stats": PERSONA_STATS,
    }


@app.get("/")
async def index():
    return FileResponse(STATIC / "index.html")


async def _refuse(ws: WebSocket, text: str) -> None:
    await ws.send_json({"type": "error", "text": text, "fatal": True})
    await ws.close()


@app.websocket("/ws/{code}")
async def ws_room(ws: WebSocket, code: str, nick: str = "", cid: str = ""):
    code = code.upper()
    nick = nick.strip()[:16]
    cid = cid.strip()[:64] or uuid.uuid4().hex
    await ws.accept()

    if not limits.CONNECTS.allow(client_ip(ws)):
        return await _refuse(ws, "Too many connection attempts. Wait a minute and try again.")

    room = rooms.get(code)
    if room is None:
        return await _refuse(ws, "Room not found. Check the code and try again.")
    if not nick:
        return await _refuse(ws, "Pick a nickname first.")

    rejoin = False
    if nick in room.cids:
        if room.cids[nick] != cid:
            return await _refuse(ws, "That nickname is taken in this room. Pick another one.")
        rejoin = True
    elif room.phase != "lobby":
        return await _refuse(ws, "This round already started. Ask the host for the next one.")
    elif len(room.players) >= MAX_PLAYERS:
        return await _refuse(ws, "This room is full.")

    if rejoin:
        old = room.players.get(nick)
        if old is not None and old is not ws:
            try:
                await old.close()
            except Exception:
                pass
    else:
        room.cids[nick] = cid
    room.players[nick] = ws
    if room.cleanup:
        room.cleanup.cancel()
        room.cleanup = None
    if room.host is None:
        room.host = nick

    await ws.send_json({"type": "joined", "code": code, "you": nick})
    if rejoin:
        await room.send_state(nick, ws)
    else:
        await room.broadcast(room.roster())

    try:
        while True:
            try:
                raw = await ws.receive_text()
            except KeyError:  # binary frame, not part of the protocol
                continue
            if len(raw) > limits.MAX_MESSAGE_CHARS:
                await ws.close(code=1009)
                break
            try:
                data = json.loads(raw)
            except ValueError:
                continue
            if not isinstance(data, dict):
                continue
            kind = data.get("type")

            key = f"{code}:{nick}"
            if kind == "chat":
                if not limits.CHAT.allow(key):
                    await ws.send_json({"type": "error", "text": "Slow down a little."})
                    continue
            elif kind == "typing":
                if not limits.TYPING.allow(key):
                    continue
            elif not limits.ACTIONS.allow(key):
                continue

            if kind == "start" and nick == room.host and room.phase == "lobby":
                if len(room.players) < min_players():
                    await ws.send_json({"type": "error", "text": f"Need at least {min_players()} players."})
                    continue
                await room.start_round()

            elif kind == "chat" and room.phase == "chat":
                text = str(data.get("text", "")).strip()[:300]
                alias = room.alias.get(nick)
                if text and alias:
                    await room.post(alias, text)

            elif kind == "typing" and room.phase == "chat":
                alias = room.alias.get(nick)
                if alias:
                    await room.typing(alias, skip=nick)

            elif kind == "vote":
                await room.cast_vote(nick, str(data.get("alias", "")))

            elif kind == "again":
                await room.play_again()

    except WebSocketDisconnect:
        pass
    finally:
        if room.players.get(nick) is ws:  # a replaced socket must not evict its successor
            room.players.pop(nick, None)
            if room.phase == "lobby":
                room.cids.pop(nick, None)
                if room.host == nick:
                    room.host = next(iter(room.players), None)
                await room.broadcast(room.roster())
            if not room.players:
                room.schedule_cleanup()


# Install-as-app files live at the site root so the manifest scope covers the whole game.
def _root_file(filename: str, media_type: str):
    async def handler():
        return FileResponse(STATIC / filename, media_type=media_type)
    return handler


for _path, _file, _type in [
    ("/manifest.webmanifest", "manifest.webmanifest", "application/manifest+json"),
    ("/icon-192.png", "icon-192.png", "image/png"),
    ("/icon-512.png", "icon-512.png", "image/png"),
    ("/icon-maskable-512.png", "icon-maskable-512.png", "image/png"),
    ("/apple-touch-icon.png", "apple-touch-icon.png", "image/png"),
    ("/favicon.svg", "favicon.svg", "image/svg+xml"),
]:
    app.add_api_route(_path, _root_file(_file, _type), methods=["GET"], include_in_schema=False)

app.mount("/static", StaticFiles(directory=STATIC), name="static")
