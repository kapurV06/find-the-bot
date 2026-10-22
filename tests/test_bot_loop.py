"""The bot's reply loop, driven against a fake room: does it actually answer the people talking to it?"""
import asyncio

from app import bot as botmod


class FakeRoom:
    def __init__(self, humans=1):
        self.phase = "chat"
        self.prompt = "defend your most unpopular food opinion"
        self.history = []
        self.bots = []
        self.alias = {f"human{i}": f"H{i}" for i in range(humans)}

    async def typing(self, alias):
        pass

    async def post(self, alias, text):
        self.history.append({"nick": alias, "text": text, "ts": 0})
        for b in self.bots:
            if b.alias != alias:
                b.wake.set()

    def say(self, who, text):
        return self.post(who, text)


def run_scenario(scenario, monkeypatch, generate, humans=1, rand=None):
    """Start one bot in a fake room, run `scenario(room, bot)`, then stop the bot."""
    monkeypatch.setattr(botmod.Bot, "generate", generate)
    if rand is not None:
        monkeypatch.setattr(botmod.random, "random", lambda: rand)

    async def main():
        room = FakeRoom(humans)
        bot = botmod.Bot("Falcon", persona=botmod.PERSONAS[0])
        room.bots = [bot]
        task = asyncio.create_task(bot.run(room))
        try:
            await scenario(room, bot)
        finally:
            room.phase = "ended"
            task.cancel()
        return room

    return asyncio.run(main())


def bot_lines(room):
    return [m["text"] for m in room.history if m["nick"] == "Falcon"]


def test_a_message_sent_while_the_bot_is_thinking_still_gets_answered(monkeypatch):
    calls = []

    async def generate(self, topic, history):
        calls.append([m["text"] for m in history if m["nick"] != "Falcon"])
        await asyncio.sleep(0.25)  # the model is slow
        return f"reply {len(calls)}"

    async def scenario(room, bot):
        await asyncio.sleep(0.1)       # the bot opens first
        await room.say("H0", "what are you saying")
        await asyncio.sleep(0.05)
        await room.say("H0", "Huh")    # arrives while the bot is still generating a reply to the first one
        await asyncio.sleep(0.05)
        await room.say("H0", "talk to me")
        await asyncio.sleep(1.8)

    room = run_scenario(scenario, monkeypatch, generate)
    assert "talk to me" in calls[-1], f"the bot never saw your last message: {calls}"
    assert room.history[-1]["nick"] == "Falcon", "your last message was left unanswered"


def test_the_bot_is_never_silent_when_spoken_to_even_if_the_model_fails(monkeypatch):
    async def generate(self, topic, history):
        return None  # Groq down, rate limited, or the filter threw the reply away

    async def scenario(room, bot):
        await asyncio.sleep(0.1)
        await room.say("H0", "who tf are you talking to ?")
        await asyncio.sleep(0.8)

    room = run_scenario(scenario, monkeypatch, generate)
    answers = [m for m in room.history if m["nick"] == "Falcon"]
    assert answers, "the bot ignored a direct question"
    assert answers[-1]["text"] in botmod.FALLBACKS


def test_in_a_group_it_does_not_answer_every_unaddressed_message(monkeypatch):
    async def generate(self, topic, history):
        return "opening line"

    async def scenario(room, bot):
        await asyncio.sleep(0.1)
        await room.say("H0", "mess ka khana crime hai")
        await room.say("H1", "paneer is overrated")
        await asyncio.sleep(0.6)

    room = run_scenario(scenario, monkeypatch, generate, humans=3, rand=0.99)
    assert bot_lines(room) == ["opening line"]


def test_the_bot_never_talks_to_itself_when_the_chat_is_quiet(monkeypatch):
    async def generate(self, topic, history):
        return "hello"

    async def scenario(room, bot):
        await asyncio.sleep(0.8)  # nobody says anything

    room = run_scenario(scenario, monkeypatch, generate, rand=0.0)
    assert bot_lines(room) == ["hello"]


def test_fallback_replies_are_safe():
    assert botmod.FALLBACKS
    for line in botmod.FALLBACKS:
        assert "bro" not in line and "bhai" not in line and line == line.lower()
        assert not botmod._EMOJI.search(line)


def test_reading_time_overlaps_the_model_call_instead_of_adding_to_it(monkeypatch):
    import time
    monkeypatch.setattr(botmod, "SPEED", 1.0)
    monkeypatch.setattr(botmod, "OPEN_DELAY", (0, 0))
    monkeypatch.setattr(botmod, "THINK_DELAY", (0.4, 0.4))
    monkeypatch.setattr(botmod, "TYPE_SECONDS_PER_CHAR", (0, 0))
    marks = {}

    counter = iter(range(100))

    async def generate(self, topic, history):
        await asyncio.sleep(0.4)  # model latency equal to the reading time
        return f"reply {next(counter)}"

    async def scenario(room, bot):
        await asyncio.sleep(0.9)  # let the opening line go out
        marks["sent"] = time.monotonic()
        await room.say("H0", "what are you saying")
        while len(bot_lines(room)) < 2:
            await asyncio.sleep(0.02)
        marks["answered"] = time.monotonic()

    run_scenario(scenario, monkeypatch, generate)
    assert marks["answered"] - marks["sent"] < 0.7, "reading time and model call are running one after the other"


def test_a_slow_model_call_is_not_retried_just_to_find_a_fallback(monkeypatch):
    monkeypatch.setattr(botmod, "RETRY_BUDGET", 0.2)
    calls = []

    async def generate(self, topic, history):
        calls.append(1)
        await asyncio.sleep(0.3)
        return None

    async def scenario(room, bot):
        await asyncio.sleep(0.5)
        await room.say("H0", "talk to me")
        await asyncio.sleep(1.0)

    room = run_scenario(scenario, monkeypatch, generate, rand=0.99)
    assert len(calls) == 2  # the opening line plus one try for the question, no retry
    assert bot_lines(room)[-1] in botmod.FALLBACKS
