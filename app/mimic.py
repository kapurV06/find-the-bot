"""Learn how the humans in a room actually type, so the bot can blend in with them.

analyze() measures the human messages seen so far, knobs() turns that into settings for the reply filter
in bot.humanize(), and describe() writes the same facts into the prompt so the model copies them too.
"""
import re
import statistics
from collections import Counter
from dataclasses import dataclass, field

MIN_MESSAGES = 3  # below this the sample is too small to copy, so the persona defaults apply

EMOJI = re.compile("[\U0001F000-\U0001FAFF☀-➿⬀-⯿‍️]+")

# Common roman-script Hindi words, used to estimate how much Hinglish the room speaks.
HINDI = {
    "kya", "nhi", "nahi", "ni", "hai", "h", "hn", "haan", "abe", "arre", "yaar", "yar", "kyu", "kyun",
    "kaise", "kaun", "kab", "kahan", "mera", "meri", "tera", "teri", "tu", "tum", "hum", "ye", "yeh", "wo", "woh",
    "tha", "thi", "kar", "kr", "karna", "raha", "rha", "rahi", "bol", "bola", "pta", "pata", "kal",
    "aaj", "bc", "mc", "saale", "sale", "accha", "achha", "theek", "thik", "matlab", "bas", "sab", "kuch", "koi",
    "aur", "toh", "chal", "chalo", "bhi", "na", "mai", "mujhe", "tujhe", "isko", "usko",
}
STOP = {
    "the", "is", "a", "an", "and", "to", "of", "it", "you", "i", "in", "that", "this", "for", "on", "with", "was",
    "are", "be", "so", "but", "or", "my", "me", "we", "do", "not", "if", "at", "as", "have", "its",
}
_BRO = re.compile(r"\bbro+\b", re.I)
_BHAI = re.compile(r"\bbhai+(?:ya)?\b", re.I)
_YAAR = re.compile(r"\bya+r+\b", re.I)


@dataclass
class ChatStyle:
    n: int = 0
    median_words: float = 0
    p90_words: int = 0
    lower_share: float = 1.0     # messages with no capital letters
    stop_share: float = 0.0      # messages ending in . ! , or ;
    apostrophe_share: float = 0.0
    comma_share: float = 0.0     # among messages of 5+ words
    emoji_share: float = 0.0
    hinglish_share: float = 0.0
    bro_msgs: int = 0
    bhai_msgs: int = 0
    yaar_msgs: int = 0
    vocab: list = field(default_factory=list)


def _share(flags) -> float:
    flags = list(flags)
    return sum(flags) / len(flags) if flags else 0.0


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def analyze(messages) -> ChatStyle:
    msgs = [m.strip() for m in messages if m and m.strip()]
    if not msgs:
        return ChatStyle()
    words = sorted(len(m.split()) for m in msgs)
    n = len(msgs)
    longish = [m for m in msgs if len(m.split()) >= 5]
    tokens = [set(re.findall(r"[a-z']{2,}", m.lower())) for m in msgs]
    counts = Counter(t for toks in tokens for t in toks if t not in STOP)
    return ChatStyle(
        n=n,
        median_words=statistics.median(words),
        p90_words=words[min(n - 1, int(0.9 * n))],
        lower_share=_share(m == m.lower() for m in msgs),
        stop_share=_share(bool(re.search(r"[.!;,]$", m)) for m in msgs),
        apostrophe_share=_share(("'" in m or "’" in m) for m in msgs),
        comma_share=_share("," in m for m in longish),
        emoji_share=_share(bool(EMOJI.search(m)) for m in msgs),
        hinglish_share=_share(bool(toks & HINDI) for toks in tokens),
        bro_msgs=sum(bool(_BRO.search(m)) for m in msgs),
        bhai_msgs=sum(bool(_BHAI.search(m)) for m in msgs),
        yaar_msgs=sum(bool(_YAAR.search(m)) for m in msgs),
        vocab=[w for w, c in counts.most_common(8) if c >= 2],
    )


def knobs(cs: ChatStyle) -> dict:
    """Reply-filter settings that copy this room. Empty until there is enough to copy."""
    if cs.n < MIN_MESSAGES:
        return {}
    return {
        "lower": _clamp(cs.lower_share, 0.05, 1.0),
        "drop_end": _clamp(1 - cs.stop_share, 0.05, 0.98),
        "drop_apos": 0.25 if cs.apostrophe_share >= 0.15 else 0.92,
        "drop_comma": _clamp(1 - cs.comma_share, 0.2, 0.95),
        "max_words": int(_clamp(round(cs.p90_words) + 4, 6, 18)),
        "keep_emoji": cs.emoji_share >= 0.25,
        # nobody talks like "bro", "bhai" or "yaar" unless this room actually does
        "ban": {w for w, c in (("bro", cs.bro_msgs), ("bhai", cs.bhai_msgs), ("yaar", cs.yaar_msgs)) if c < 2},
    }


def describe(cs: ChatStyle) -> str:
    """Prompt text describing how the humans type. Empty until there is enough to copy."""
    if cs.n < MIN_MESSAGES:
        return ""
    k = knobs(cs)
    lower = ("almost always in lowercase" if cs.lower_share >= 0.8
             else "usually capitalised" if cs.lower_share <= 0.3 else "a mix of lowercase and capitals")
    stops = ("almost never end a message with a full stop or comma" if cs.stop_share <= 0.2
             else "usually end messages with a full stop" if cs.stop_share >= 0.6
             else "sometimes end messages with a full stop")
    hindi = (f"about {round(100 * cs.hinglish_share)}% of their messages mix in Hindi words"
             if cs.hinglish_share >= 0.15 else "they write almost entirely in English")
    lines = [
        "How the humans in this chat type right now (blend in with exactly this):",
        f"- messages are typically {round(cs.median_words)} words, and almost none are over {cs.p90_words}",
        f"- they write {lower}",
        f"- they {stops}",
        f"- {hindi}",
    ]
    if cs.vocab:
        lines.append(f"- words they keep repeating: {', '.join(cs.vocab)}")
    lines.append("- emojis: they use some, so one now and then is fine" if k["keep_emoji"]
                 else "- emojis: nobody uses them, so you don't either")
    lines.append("Copy their length, casing, punctuation and slang. Reuse a word or spelling someone just used. "
                 "Never use slang nobody here has used. Never copy a message word for word."
                 + (f" Do not say {' or '.join(sorted(k['ban']))}." if k["ban"] else ""))
    return "\n".join(lines) + "\n\n"
