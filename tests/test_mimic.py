import asyncio
import json

import httpx

from app import bot as botmod
from app import mimic

SURE = {"lower": 1.0, "drop_end": 1.0, "drop_q": 1.0, "drop_apos": 1.0, "drop_comma": 1.0, "typo": 0.0, "max_words": 12}

LOWER_HINGLISH = ["abe kya scene hai", "nhi yaar kal tha na", "lol sus", "pta nhi kya hua", "abe sach bol raha hu", "same lol"]
PROPER_ENGLISH = [
    "I think the mess food is fine, honestly.", "Well, that's one way to look at it.",
    "No, I was in the library all evening.", "Why would anyone say that, though?", "That's not what happened at all.",
]


# ---------- bro and bhai are banned ----------

def test_bro_and_bhai_are_removed_from_replies():
    assert botmod.humanize("tu bhi na, bro", style=SURE) == "tu bhi na"
    assert botmod.humanize("Bhai kya kar raha hai tu", style=SURE) == "kya kar raha hai tu"
    assert botmod.humanize("abe bhaiya chal", style=SURE) == "abe chal"
    assert botmod.humanize("BROOO what", style=SURE) == "what"
    assert botmod.humanize("bro", style=SURE) is None


def test_the_ban_is_lifted_only_when_the_room_really_says_it():
    quiet_room = mimic.knobs(mimic.analyze(LOWER_HINGLISH))
    assert quiet_room["ban"] == {"bro", "bhai", "yaar"}
    bro_room = mimic.knobs(mimic.analyze(["bro what", "nah bro", "same lol", "abe bhai sun"]))
    assert bro_room["ban"] == {"bhai", "yaar"}  # two people said bro, only one said bhai, nobody said yaar


# ---------- learning how the room types ----------

def test_not_enough_messages_means_no_copying_yet():
    assert mimic.knobs(mimic.analyze(["hi", "yo"])) == {}
    assert mimic.describe(mimic.analyze(["hi", "yo"])) == ""


def test_a_lowercase_hinglish_room_gets_lowercase_hinglish_settings():
    cs = mimic.analyze(LOWER_HINGLISH)
    k = mimic.knobs(cs)
    assert cs.hinglish_share > 0.5 and cs.lower_share == 1.0
    assert k["lower"] == 1.0 and k["drop_end"] > 0.9 and k["keep_emoji"] is False
    assert k["max_words"] <= 10
    text = mimic.describe(cs)
    assert "lowercase" in text and "Hindi" in text and "abe" in text and "nobody uses them" in text


def test_a_proper_english_room_gets_capitals_and_full_stops():
    cs = mimic.analyze(PROPER_ENGLISH)
    k = mimic.knobs(cs)
    assert k["lower"] < 0.3 and k["drop_end"] < 0.5 and k["drop_apos"] < 0.5
    assert "almost entirely in English" in mimic.describe(cs)
    out = botmod.humanize("I think the mess food is fine, honestly.", style={**SURE, **k, "typo": 0.0})
    assert out is not None  # the filter no longer flattens it to chat-speak


def test_emojis_are_allowed_only_if_the_room_uses_them():
    emoji_room = mimic.analyze(["lol \U0001F602", "same \U0001F480", "abe kya", "sus \U0001F602"])
    k = mimic.knobs(emoji_room)
    assert k["keep_emoji"] is True
    assert "\U0001F602" in botmod.humanize("sus \U0001F602", style={**SURE, "keep_emoji": True})
    assert "\U0001F602" not in botmod.humanize("sus \U0001F602", style=SURE)


def test_message_length_follows_the_room():
    short = mimic.knobs(mimic.analyze(["ok", "same", "lol", "nhi", "abe"]))
    chatty = mimic.knobs(mimic.analyze(["this is a fairly long message about the mess food today"] * 4))
    assert short["max_words"] < chatty["max_words"]


# ---------- the bot actually uses it ----------

def _prompt_sent_for(history, peers=()):
    seen = {}

    def handler(req):
        seen["user"] = json.loads(req.content)["messages"][1]["content"]
        return httpx.Response(200, json={"choices": [{"message": {"content": "abe sach bol raha hu"}}]})

    async def run():
        botmod._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        try:
            bot = botmod.Bot("Falcon", persona={"label": "T", "desc": "test", "style": {}})
            bot.peers = set(peers)
            return await bot.generate("topic", history)
        finally:
            await botmod._client.aclose()
            botmod._client = None

    return asyncio.run(run()), seen["user"]


def _chat(*texts, who="Mango"):
    return [{"nick": who, "text": t} for t in texts]


def test_prompt_describes_the_room_once_there_is_enough_to_copy():
    _, user = _prompt_sent_for(_chat(*LOWER_HINGLISH))
    assert "How the humans in this chat type right now" in user and "lowercase" in user
    _, early = _prompt_sent_for(_chat("abe", "kya"))
    assert "How the humans in this chat type" not in early


def test_the_bot_learns_from_humans_only_not_from_itself_or_other_bots():
    history = _chat("I will explain this properly now.", "Absolutely, that is correct.", "Yes, of course.", who="Falcon")
    history += _chat("Capitalised sentence number one.", who="Panda")  # another bot
    _, user = _prompt_sent_for(history, peers={"Panda"})
    assert "How the humans in this chat type" not in user  # no human messages to learn from yet


def test_the_bot_never_parrots_a_message_word_for_word():
    history = _chat("abe", "kya", "lol", "abe sach bol raha hu")
    text, _ = _prompt_sent_for(history)
    assert text is None


# ---------- it has to actually talk to the people talking to it ----------

def test_questions_and_confusion_are_recognised_as_aimed_at_the_bot():
    for text in ("what are you saying", "Huh", "talk to me", "Who tf are you talking to ?", "Falcon you there"):
        assert botmod.addressed(text, "Falcon"), text
    for text in ("paneer biryani legit", "lol same", "mess ka khana crime hai"):
        assert not botmod.addressed(text, "Falcon"), text


def test_a_lone_human_always_gets_an_answer_and_a_group_only_sometimes():
    assert botmod.should_reply("lol same", "Falcon", humans=1, roll=0.99)
    assert botmod.should_reply("what are you saying", "Falcon", humans=4, roll=0.99)
    assert not botmod.should_reply("lol same", "Falcon", humans=4, roll=0.99)
    assert botmod.should_reply("lol same", "Falcon", humans=4, roll=0.0)


def test_prompt_puts_the_question_first_so_it_cannot_be_ignored():
    history = _chat("paneer biryani legit", who="Falcon") + _chat("what are you saying", who="Ember")
    _, user = _prompt_sent_for(history)
    assert "aimed at you" in user and '"what are you saying"' in user

    _, casual = _prompt_sent_for(_chat("mess ka khana crime hai", who="Ember"))
    assert "aimed at you" not in casual and "The latest message is from Ember" in casual


def test_prompt_warns_the_bot_off_repeating_its_own_pattern():
    history = (_chat("cheese toast meh", "pasta with ketchup weird", who="Falcon")
               + _chat("talk to me", who="Ember"))
    _, user = _prompt_sent_for(history)
    assert "Your last messages were: cheese toast meh | pasta with ketchup weird" in user


def test_nothing_in_the_instructions_tells_the_bot_to_ignore_people():
    assert "doesn't answer the question" not in botmod.SYSTEM
    for p in botmod.PERSONAS:
        assert "different question" not in p["desc"]
    assert "aimed at you" in botmod.SYSTEM and "the last message comes first" in botmod.SYSTEM
