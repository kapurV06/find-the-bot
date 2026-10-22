import asyncio
import json

import httpx

from app import bot as botmod
from app import limits


def test_humanize_strips_llm_tells():
    for _ in range(200):
        out = botmod.humanize('Mango: "Bro that’s wild — honestly."', "Mango")
        assert out and "—" not in out and not out.startswith("Mango") and '"' not in out


def test_humanize_skips_and_takes_first_line():
    assert botmod.humanize("<skip>") is None
    assert botmod.humanize("") is None
    assert botmod.humanize("line one\nline two").startswith("line one")


def _bot():
    return botmod.Bot("Falcon", persona={"label": "Test", "desc": "test persona"})


def _generate(handler):
    async def run():
        botmod._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        try:
            return await _bot().generate("worst hostel food", [{"nick": "Mango", "text": "hostel paneer is a crime"}])
        finally:
            await botmod._client.aclose()
            botmod._client = None
    return asyncio.run(run())


def test_groq_request_shape_and_reasoning_fallback():
    calls = []

    def handler(req):
        body = json.loads(req.content)
        calls.append(body)
        assert req.headers["authorization"] == "Bearer test-key"
        if "reasoning_effort" in body:
            return httpx.Response(400, json={"error": "unsupported"})
        return httpx.Response(200, json={"choices": [{"message": {"content": "falcon: lol same bro."}}]})

    text = _generate(handler)
    assert text == "lol same"  # "bro" is banned, so it is stripped
    assert len(calls) == 2 and "reasoning_effort" not in calls[1]
    assert "test persona" in calls[1]["messages"][0]["content"]
    assert "Mango: hostel paneer is a crime" in calls[1]["messages"][1]["content"]


def test_groq_failure_means_the_bot_stays_quiet():
    assert _generate(lambda req: httpx.Response(429, json={"error": "rate limited"})) is None


def test_groq_call_budget_stops_calls_before_they_are_made(monkeypatch):
    monkeypatch.setattr(limits.GROQ, "burst", 1)
    monkeypatch.setattr(limits.GROQ, "rate", 0)
    limits.GROQ.reset()
    calls = []

    def handler(req):
        calls.append(1)
        return httpx.Response(200, json={"choices": [{"message": {"content": "hey"}}]})

    assert _generate(handler) == "hey"
    assert _generate(handler) is None
    assert len(calls) == 1


# ---------- sounding like a person, not an assistant ----------

SURE = {"lower": 1.0, "drop_end": 1.0, "drop_q": 1.0, "drop_apos": 1.0, "drop_comma": 1.0, "typo": 0.0, "max_words": 12}


def test_emojis_and_emoticons_never_survive():
    for raw in ("tu bhi na, bro \U0001F602", "lol that's wild \U0001F480\U0001F525", "same :) xD", "nice ;D", "abe :joy:".replace(":joy:", "")):
        out = botmod.humanize(raw, style=SURE)
        assert out and not botmod._EMOJI.search(out) and not botmod._EMOTICON.search(out), (raw, out)


def test_messages_look_typed_not_written():
    out = botmod.humanize("Honestly, I don't think that's right. Why would anyone do that?", style=SURE)
    assert out == out.lower() and "'" not in out and "," not in out
    assert not out.endswith((".", "!", "?"))
    assert botmod.humanize("Abe, kya bol raha hai tu!!", style=SURE) == "abe kya bol raha hai tu"


def test_long_replies_are_cut_down():
    out = botmod.humanize("this is the first short bit. and here goes a much longer second sentence that nobody types", style=SURE)
    assert out == "this is the first short bit"
    # far too long and no short first sentence: dropped, never posted cut off mid-sentence
    assert botmod.humanize(" ".join(["word"] * 40), style={**SURE, "max_words": 6}) is None
    # only a little over the limit: kept whole, a slightly long message beats a mangled one
    assert botmod.humanize("honestly the mess food has been getting worse lately", style={**SURE, "max_words": 7}) == \
        "honestly the mess food has been getting worse lately"


def test_assistant_phrases_make_the_bot_stay_quiet():
    for raw in ("As an AI I can't really say", "Great question, honestly", "I'm a bot lol", "<skip>"):
        assert botmod.humanize(raw, style=SURE) is None


def test_every_persona_is_usable():
    assert len({p["label"] for p in botmod.PERSONAS}) == len(botmod.PERSONAS)
    for p in botmod.PERSONAS:
        prompt = botmod.SYSTEM.format(persona=p["desc"])
        assert "no emojis" in prompt and p["desc"] in prompt


# ---------- no verbal tics, no nonsense ----------

def test_yaar_is_stripped_like_bro_and_bhai():
    assert botmod.humanize("nhi yaar kal bunk maara tha", style=SURE) == "nhi kal bunk maara tha"
    assert botmod.humanize("ye toh theek hai yaar", style=SURE) == "ye toh theek hai"
    assert botmod.humanize("arre yaaar kya bol raha hai", style=SURE) == "arre kya bol raha hai"


def test_the_bot_cannot_end_every_message_with_the_same_word():
    assert botmod.drop_repeated_tail("ye toh theek hai meh", ["cheese toast meh"]) == "ye toh theek hai"
    assert botmod.drop_repeated_tail("ye toh theek hai meh", ["something else entirely"]) == "ye toh theek hai meh"
    assert botmod.drop_repeated_tail("same", ["same"]) == "same"  # too short to touch


def test_the_prompt_itself_never_teaches_the_tics():
    banned = ("bro", "bhai", "yaar")
    text = botmod.SYSTEM.replace('"bro", "bhai" or "yaar"', "")
    for p in botmod.PERSONAS:
        text += p["desc"]
    for line in botmod.EXAMPLES + botmod.FALLBACKS:
        assert not any(w in line.lower().split() for w in banned), line
    assert not any(w in text.lower() for w in banned)
    assert "unhinged" not in text and "random" not in botmod.PERSONAS[0]["desc"]


def test_copying_a_prompt_example_is_dropped():
    def handler(req):
        return httpx.Response(200, json={"choices": [{"message": {"content": botmod.EXAMPLES[0]}}]})

    assert _generate(handler) is None


def test_sampling_is_calm_enough_to_stay_coherent():
    seen = {}

    def handler(req):
        seen.update(json.loads(req.content))
        return httpx.Response(200, json={"choices": [{"message": {"content": "same"}}]})

    _generate(handler)
    assert seen["temperature"] <= 0.8
