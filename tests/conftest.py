import asyncio
import os
import sys
from pathlib import Path

os.environ["GROQ_API_KEY"] = "test-key"  # must be set before the app is imported
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from starlette.testclient import TestClient

import app.main as main
from app import bot as botmod
from app import limits
from tests import helpers


@pytest.fixture(autouse=True)
def clean_state(monkeypatch):
    limits.reset_all()
    main.PERSONA_STATS.clear()
    main.rooms.clear()
    helpers.BUF.clear()
    monkeypatch.setattr(botmod, "SPEED", 0.0)       # no human-like delays in tests
    monkeypatch.setattr(main, "CHAT_SECONDS", 1.5)
    monkeypatch.setattr(main, "VOTE_SECONDS", 5)
    yield
    main.rooms.clear()


@pytest.fixture
def client():
    # One portal for the whole test, so room timers and bot tasks share a single event loop like in production.
    with TestClient(main.app) as c:
        yield c


@pytest.fixture
def fake_llm(monkeypatch):
    """Bots reply from a fixed list instead of calling Groq."""
    replies = iter(["lol same", "nah that's mid", "bruh", "wait what", "true", "ok fair"] * 30)

    async def fake_generate(self, topic, history):
        await asyncio.sleep(0.15)
        return next(replies)

    monkeypatch.setattr(botmod.Bot, "generate", fake_generate)
