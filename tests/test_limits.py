from app.limits import RateLimiter


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def test_burst_then_blocked():
    clock = Clock()
    rl = RateLimiter(rate=1, burst=3, clock=clock)
    assert [rl.allow("a") for _ in range(5)] == [True, True, True, False, False]


def test_tokens_refill_over_time_but_never_past_the_burst():
    clock = Clock()
    rl = RateLimiter(rate=1, burst=3, clock=clock)
    for _ in range(3):
        rl.allow("a")
    assert not rl.allow("a")
    clock.t += 2
    assert rl.allow("a") and rl.allow("a") and not rl.allow("a")
    clock.t += 1000
    assert [rl.allow("a") for _ in range(4)] == [True, True, True, False]


def test_keys_are_independent():
    rl = RateLimiter(rate=0, burst=1, clock=Clock())
    assert rl.allow("a") and not rl.allow("a")
    assert rl.allow("b")


def test_memory_is_bounded():
    rl = RateLimiter(rate=1, burst=1, max_keys=100, clock=Clock())
    for i in range(1000):
        rl.allow(str(i))
    assert len(rl._buckets) == 100
