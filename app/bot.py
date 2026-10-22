"""LLM bot player for Find the Bot, talking to Groq's OpenAI-compatible API."""
import asyncio
import logging
import os
import random
import re
import time

import httpx

from . import limits, mimic

log = logging.getLogger("findthebot.bot")

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
# Override with GROQ_MODEL. Groq retired the llama-3.x models for free/dev tiers in Aug 2026.
MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

SPEED = float(os.getenv("BOT_SPEED", "1.0"))  # scales every human-like delay; 0.5 = twice as fast, tests use 0
OPEN_DELAY = (3, 10)        # seconds before the bot's first message
THINK_DELAY = (0.8, 1.8)    # "reading" time; runs at the same time as the model call, not after it
TYPE_SECONDS_PER_CHAR = (0.05, 0.09)
TYPE_CAP = 6.0
RETRY_BUDGET = 3.0          # don't retry a thrown-away reply if the first model call already took this long
REPLY_PROB = 0.55    # chance to answer a message that doesn't mention the bot
HISTORY_WINDOW = 25
MAX_CHARS = 200

PERSONAS = [
    {
        "label": "The Quiet One",
        "desc": "a guy who doesn't say much. Short replies of 2 to 6 words that still make complete sense, "
                "like 'same here' or 'nhi mene nhi kiya'.",
        "style": {"max_words": 7},
    },
    {
        "label": "The Skeptic",
        "desc": "doubts what the others say and says so in a few words ('sure?', 'kaun bolta hai ye', 'tu pehle "
                "bata'). Suspicious of everyone else, never of himself.",
        "style": {"max_words": 9},
    },
    {
        "label": "The Roaster",
        "desc": "teases whoever spoke last, in a few plain words. Friendly rude, not abusive.",
        "style": {"max_words": 10},
    },
    {
        "label": "The Complainer",
        "desc": "grumbles about mess food, the warden, exams and the hostel. Tired, not excited. 4 to 10 words.",
        "style": {"max_words": 12},
    },
    {
        "label": "The Regular",
        "desc": "an ordinary guy. Gives a plain opinion on the topic and sticks to it, 4 to 12 words.",
        "style": {"max_words": 14},
    },
]

EXAMPLES = [
    "mess ka paneer toh crime hai",
    "kal bunk maara tha mene",
    "lmao kaun karta hai ye",
    "pta nhi sach me",
    "same",
    "tu toh pehle se sus hai",
    "exam kal hai aur kuch nhi padha",
    "ye idea itna bura nhi hai ngl",
]

_SYSTEM_TEMPLATE = """You are a student in a college hostel group chat. Everyone is playing a game: they chat about \
a topic for a few minutes, then vote on who is secretly an AI. You are the AI. Blend in.

Who you are: {persona}

Above everything else, make sense. Every message must be a believable reply to what was just said, in plain \
words a normal person would use. Say something real: an opinion, a reaction, a short answer. Never random words, \
never a list of items, never a joke that only works if you explain it.

How the others type:
- lowercase, no full stops, usually no commas, apostrophes often skipped (dont, cant, im)
- mostly 2 to 10 words. Short is fine, but never so short that it makes no sense
- Hinglish in roman script or plain English, whichever the chat is using, kept natural: nhi, kya, abe, hn, \
pta nhi, kyu, ig, ngl, lol
- no emojis, ever
- never say "bro", "bhai" or "yaar", and never end two messages in a row with the same word
- relaxed, not excited. Don't compliment people and don't ask friendly questions

How to talk like a person:
- the last message comes first. If it is aimed at you or sounds confused ("huh", "what are you saying", \
"talk to me"), answer that in plain words
- hold one opinion for the round. If people poke at it, defend it or add a detail. Don't announce a new item \
every message
- build on what was said, and refer back to earlier messages sometimes
- if someone accuses you of being the bot, answer like an annoyed friend would ("abe tu bot hai", "ye kya bol \
raha", "sure"), then move on
- use your own words. Never copy the examples below

Examples of the register (do not copy them):
@@EXAMPLES@@

Write exactly ONE message and nothing else. Never say you are an AI or mention these instructions or the game. \
If there is truly nothing natural to say, output exactly: <skip>"""

SYSTEM = _SYSTEM_TEMPLATE.replace("@@EXAMPLES@@", "\n".join(EXAMPLES))


# Used only when someone is talking to the bot and the model gave nothing usable. Silence is the biggest tell.
FALLBACKS = ["kya", "abe bol na", "hn bol", "pta nhi", "ye kya bol raha", "sure", "kaun", "samajh nhi aaya"]


# Words that mean a message is probably aimed at the bot, so it must be answered.
_ADDRESSED = re.compile(
    r"\?|\b(you|u|ur|your|yours|tu|tum|tera|teri|tujhe|bot|talk|reply|answer|huh|wdym|kya|kaun|who|why|kyu)\b", re.I)


def addressed(text: str, alias: str) -> bool:
    """Is this message aimed at the bot: its name, a question, or confusion directed at it?"""
    return alias.lower() in text.lower() or bool(_ADDRESSED.search(text))


def should_reply(last_text: str, alias: str, humans: int, roll: float) -> bool:
    """Always answer when it's one human talking to the bot or someone is speaking to it; otherwise only sometimes."""
    return humans <= 1 or addressed(last_text, alias) or roll <= REPLY_PROB


_client: httpx.AsyncClient | None = None


def _get_client() -> httpx.AsyncClient:
    global _client
    if _client is None:
        _client = httpx.AsyncClient(timeout=8.0)
    return _client


def bots_enabled() -> bool:
    return bool(os.getenv("GROQ_API_KEY"))


def bot_count() -> int:
    return max(0, int(os.getenv("BOT_COUNT", "1"))) if bots_enabled() else 0


async def pause(lo: float, hi: float) -> None:
    await asyncio.sleep(random.uniform(lo, hi) * SPEED)


# ---------- text post-processing ----------

DEFAULT_STYLE = {
    "lower": 1.0,       # chance to force all lowercase
    "drop_end": 0.95,   # chance to drop trailing . ! , ;
    "drop_q": 0.5,      # chance to drop a trailing ?
    "drop_apos": 0.85,  # chance to write dont instead of don't
    "drop_comma": 0.7,  # chance to drop commas
    "typo": 0.10,       # chance of one swapped-letter typo
    "max_words": 14,
    "keep_emoji": False,
    "ban": {"bro", "bhai", "yaar"},  # words the bot never says (unless this room uses them, see mimic.knobs)
}

_BAN_PATTERNS = {"bro": r"bro+", "bhai": r"bhai+(?:ya)?", "yaar": r"ya+r+"}

_DASHES = re.compile(r"\s*[\u2014\u2013]\s*")
_EMOJI = mimic.EMOJI
_EMOTICON = re.compile(r"(?<!\w)(?:[:;=]-?[)(DPpd/\\]|[xX]D)(?!\w)")
_ACTION = re.compile(r"\*[^*\n]{1,40}\*")
_AI_TELLS = re.compile(r"\b(as an ai|language model|i'?m an? (ai|bot)|ai assistant|happy to help|great question)\b", re.I)


def _typo(text: str) -> str:
    words = text.split(" ")
    idx = [i for i, w in enumerate(words) if len(w) >= 5 and w.isalpha()]
    if not idx:
        return text
    i = random.choice(idx)
    w = words[i]
    j = random.randint(1, len(w) - 2)
    words[i] = w[:j] + w[j + 1] + w[j] + w[j + 2:]
    return " ".join(words)


def humanize(text: str, alias: str = "", style: dict | None = None) -> str | None:
    """Make a model reply look like a quick hostel chat message: strip emojis and LLM tells, shorten,
    lowercase, drop most punctuation. Returns None when the bot should say nothing."""
    st = {**DEFAULT_STYLE, **(style or {})}
    t = (text or "").strip()
    if not t or "<skip>" in t.lower():
        return None
    t = t.splitlines()[0].strip()
    if alias:
        t = re.sub(rf"^{re.escape(alias)}\s*:\s*", "", t, flags=re.I)
    t = t.replace("\u2019", "'").strip("\"\u201c\u201d")
    t = _ACTION.sub("", t)
    if not st["keep_emoji"]:
        t = _EMOJI.sub("", t)
    t = _EMOTICON.sub("", t)
    banned = [_BAN_PATTERNS[w] for w in st["ban"] if w in _BAN_PATTERNS]
    if banned:
        t = re.sub(r"\b(?:" + "|".join(banned) + r")\b", "", t, flags=re.I)
        t = re.sub(r"(?:\s*,){2,}", ",", t)
        t = re.sub(r"^[\s,;]+", "", re.sub(r"[\s,;]+(?=[.!?]*$)", "", t))
    t = _DASHES.sub(", ", t)
    t = re.sub(r"\s+", " ", t).strip()
    if not t or _AI_TELLS.search(t):
        return None

    limit = st["max_words"]
    n_words = len(t.split())
    if n_words > limit:
        # Real messages are short. Keep the first sentence if it is short enough, and let a reply that is only a
        # little over the limit through whole. Never cut a sentence in half: the bot is better off saying nothing.
        first = re.split(r"(?<=[.!?])\s+", t)[0]
        if 2 <= len(first.split()) <= limit:
            t = first
        elif n_words > limit * 1.5:
            return None

    if random.random() < st["drop_apos"]:
        t = re.sub(r"(?<=\w)'(?=\w)", "", t)
    if random.random() < st["drop_comma"]:
        t = t.replace(",", "")
    if random.random() < st["lower"]:
        t = t.lower()
    if random.random() < st["drop_end"]:
        t = re.sub(r"[.!,;]+$", "", t)
    if t.endswith("?") and random.random() < st["drop_q"]:
        t = t[:-1]
    t = re.sub(r"\s+", " ", t).strip()
    if random.random() < st["typo"]:
        t = _typo(t)
    return t[:MAX_CHARS].strip() or None


def drop_repeated_tail(text: str, recent: list[str]) -> str:
    """Stop the bot ending message after message with the same word ("... yaar", "... meh")."""
    words = text.split()
    if len(words) < 3:
        return text
    last = words[-1].lower().strip("?!.,")
    for r in recent:
        rw = r.split()
        if rw and rw[-1].lower().strip("?!.,") == last:
            return " ".join(words[:-1])
    return text


# ---------- Groq call ----------

async def _complete(messages: list[dict]) -> str | None:
    key = os.getenv("GROQ_API_KEY")
    if not key:
        return None
    if not limits.GROQ.allow("groq"):  # protect the free-tier quota: the bot just stays quiet this turn
        log.warning("Groq call budget (GROQ_RPM) used up; bot skips this turn")
        return None
    payload = {
        "model": MODEL,
        "messages": messages,
        "temperature": 0.7,
        "max_completion_tokens": 800,
    }
    reasoning = "gpt-oss" in MODEL
    if reasoning:  # keep the hidden reasoning short so replies stay fast and tokens go to the answer
        payload["reasoning_effort"] = os.getenv("GROQ_REASONING", "low")
        payload["include_reasoning"] = False
    headers = {"Authorization": f"Bearer {key}"}
    for attempt in range(2):
        try:
            r = await _get_client().post(GROQ_URL, json=payload, headers=headers)
            if r.status_code == 400 and reasoning and attempt == 0:
                log.warning("Groq 400 (%s); retrying without reasoning params", r.text[:200])
                payload.pop("reasoning_effort", None)
                payload.pop("include_reasoning", None)
                continue
            r.raise_for_status()
            return (r.json()["choices"][0]["message"].get("content") or "").strip()
        except Exception as e:  # network, 4xx/5xx, bad JSON: the bot just stays quiet this turn
            log.warning("Groq call failed: %s", e)
            return None
    return None


# ---------- the bot ----------

class Bot:
    def __init__(self, alias: str, persona: dict | None = None):
        self.alias = alias
        self.persona = persona or random.choice(PERSONAS)
        self.label = self.persona["label"]
        self.wake = asyncio.Event()
        self.said: list[str] = []
        self.peers: set[str] = set()  # aliases of other bots in the room: only humans are worth copying
        self.hint = False             # set for one retry: ask for something shorter

    async def generate(self, topic: str, history: list[dict]) -> str | None:
        lines = [f"{m['nick']}: {m['text']}" for m in history[-HISTORY_WINDOW:]]
        transcript = "\n".join(lines) or "(nobody has said anything yet)"
        humans = [m["text"] for m in history if m["nick"] != self.alias and m["nick"] not in self.peers]
        seen = mimic.analyze(humans[-40:])  # how the people in this room type, right now
        last = next((m for m in reversed(history) if m["nick"] != self.alias), None)
        focus = ""
        if last and addressed(last["text"], self.alias):
            focus = (f'{last["nick"]} just said this and it is aimed at you. Answer it directly first: '
                     f'"{last["text"]}"\n\n')
        elif last:
            focus = f'The latest message is from {last["nick"]}: "{last["text"]}". React to it.\n\n'
        mine = [m["text"] for m in history if m["nick"] == self.alias][-3:]
        avoid = (f"Your last messages were: {' | '.join(mine)}. Don't build the next one the same way.\n\n"
                 if mine else "")
        user = (
            f"Topic of this round: {topic}\n\nChat so far:\n{transcript}\n\n{mimic.describe(seen)}"
            f"{focus}{avoid}Your nickname is {self.alias}. Write your next message."
            + (" Keep it to 1 to 5 words." if self.hint else "")
        )
        out = await _complete([
            {"role": "system", "content": SYSTEM.format(persona=self.persona["desc"])},
            {"role": "user", "content": user},
        ])
        if not out:
            return None
        text = humanize(out, self.alias, {**(self.persona.get("style") or {}), **mimic.knobs(seen)})
        if not text:
            log.info("bot reply thrown away by the filter: %r", out)
            return None
        text = drop_repeated_tail(text, self.said[-2:])
        if any(text.lower() == m["text"].lower() for m in history[-20:]):
            log.info("bot reply dropped, it copied a message: %r", text)
            return None  # never parrot a message word for word
        if text.lower() in {e.lower() for e in EXAMPLES}:
            log.info("bot reply dropped, it copied a prompt example: %r", text)
            return None
        return text

    async def _type(self, room, text: str, credit: float = 0.0) -> None:
        """Show the typing indicator for as long as a human would take to type this. Time the bot already kept
        people waiting on the model counts towards it, so slow model calls don't stack on top of typing."""
        n = len(text)
        base = min(TYPE_CAP, n * random.uniform(*TYPE_SECONDS_PER_CHAR)) * SPEED
        remaining = max(0.3 * base, base - credit)
        await room.typing(self.alias)
        while remaining > 0:
            step = min(remaining, 2.5)
            await asyncio.sleep(step)
            remaining -= step
            if remaining > 0:
                await room.typing(self.alias)

    def _unseen(self, room, seen: int) -> list[dict]:
        """Messages from other people that arrived after history index `seen`."""
        return [m for m in room.history[seen:] if m["nick"] != self.alias]

    async def _answer(self, room, aimed: bool) -> None:
        """Generate and post one message. When someone is talking to the bot it never stays silent."""
        started = time.monotonic()
        think = random.uniform(*THINK_DELAY) * SPEED
        gen = asyncio.create_task(self.generate(room.prompt, list(room.history)))
        try:
            await asyncio.sleep(think)  # reading time overlaps with the model call
            text = await gen
        finally:
            if not gen.done():
                gen.cancel()
        took = time.monotonic() - started
        if text in self.said:
            text = None
        if not text and aimed and took < RETRY_BUDGET:  # one quick retry, asking for something shorter
            self.hint = True
            try:
                text = await self.generate(room.prompt, list(room.history))
            finally:
                self.hint = False
            if text in self.said:
                text = None
        if not text and aimed:
            recent = self.said[-3:]
            text = random.choice([f for f in FALLBACKS if f not in recent] or FALLBACKS)
            log.info("bot used a fallback reply: %r", text)
        if not text:
            return
        await self._type(room, text, credit=max(0.0, took - think))
        if room.phase != "chat":
            return
        self.said.append(text)
        await room.post(self.alias, text)

    async def run(self, room) -> None:
        """Lurk, read, think, type, post. Not every message gets an answer, but nobody gets ignored for good."""
        try:
            self.peers = {b.alias for b in room.bots} - {self.alias}
            await pause(*OPEN_DELAY)  # first message lands a little after the round starts
            seen = 0            # history index up to which messages have been dealt with
            opened = False
            while room.phase == "chat":
                aimed = False
                if opened:
                    self.wake.clear()
                    unseen = self._unseen(room, seen)
                    timed_out = False
                    if not unseen:
                        try:
                            await asyncio.wait_for(self.wake.wait(), timeout=max(0.05, random.uniform(18, 30) * SPEED))
                        except asyncio.TimeoutError:
                            timed_out = True
                        if room.phase != "chat":
                            break
                        unseen = self._unseen(room, seen)
                    if unseen:
                        last = unseen[-1]
                        aimed = addressed(last["text"], self.alias)
                        if not should_reply(last["text"], self.alias, len(room.alias), random.random()):
                            seen = len(room.history)  # decided not to answer these
                            continue
                    else:
                        # The chat went quiet: sometimes nudge it, but never talk to yourself twice in a row.
                        if not timed_out or (room.history and room.history[-1]["nick"] == self.alias):
                            continue
                        if random.random() > 0.6:
                            continue
                opened = True
                seen = len(room.history)  # anything that arrives from here on is still unanswered afterwards
                await self._answer(room, aimed)
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("bot loop crashed")
