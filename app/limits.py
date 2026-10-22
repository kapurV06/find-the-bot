"""Small in-memory token-bucket rate limiters.

Single process only, like the rest of the room state. Limits are deliberately generous per IP,
because a whole hostel floor can share one public address.
"""
import os
import time
from collections import OrderedDict


class RateLimiter:
    """Allow `rate` events per second on average, with bursts up to `burst`, tracked per key."""

    def __init__(self, rate: float, burst: float, max_keys: int = 5000, clock=time.monotonic):
        self.rate = rate
        self.burst = burst
        self.max_keys = max_keys
        self.clock = clock
        self._buckets: OrderedDict[str, tuple[float, float]] = OrderedDict()

    def allow(self, key: str, cost: float = 1.0) -> bool:
        now = self.clock()
        tokens, stamp = self._buckets.get(key, (self.burst, now))
        tokens = min(self.burst, tokens + (now - stamp) * self.rate)
        ok = tokens >= cost
        if ok:
            tokens -= cost
        self._buckets[key] = (tokens, now)
        self._buckets.move_to_end(key)
        while len(self._buckets) > self.max_keys:  # forget the least recently seen keys
            self._buckets.popitem(last=False)
        return ok

    def reset(self) -> None:
        self._buckets.clear()


def _per_minute(env_name: str, default: int, burst: float | None = None) -> RateLimiter:
    n = float(os.getenv(env_name, default))
    return RateLimiter(rate=n / 60.0, burst=burst if burst is not None else n)


MAX_MESSAGE_CHARS = 4096                      # anything bigger is not a real game message
MAX_ROOMS = int(os.getenv("MAX_ROOMS", "200"))

ROOMS = _per_minute("ROOMS_PER_MIN", 10)       # create-room calls, per IP
CONNECTS = _per_minute("CONNECTS_PER_MIN", 60)  # websocket connection attempts, per IP
CHAT = RateLimiter(rate=1.2, burst=5)          # chat messages, per room + player
TYPING = RateLimiter(rate=0.5, burst=2)        # typing pings, per room + player (dropped silently)
ACTIONS = RateLimiter(rate=1.0, burst=5)       # start / vote / again, per room + player
_groq_rpm = float(os.getenv("GROQ_RPM", "25"))
GROQ = RateLimiter(rate=_groq_rpm / 60.0, burst=min(_groq_rpm, 10))  # LLM calls, whole server

ALL = [ROOMS, CONNECTS, CHAT, TYPING, ACTIONS, GROQ]


def reset_all() -> None:
    for limiter in ALL:
        limiter.reset()
