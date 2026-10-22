# Find the Bot

A party game for a friend group. Everyone chats about a prompt for 3 minutes. One player is an LLM.
Then you vote on who the bot was, and the reveal shows who fooled whom.

Everyone (humans and bot) gets a random codename each round, so the bot can't be spotted by an unfamiliar name.

## Run locally

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export GROQ_API_KEY=gsk_...        # without this the game runs humans-only (2+ players)
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000`. Friends on the same WiFi use your LAN IP. For anyone else, `cloudflared tunnel --url http://localhost:8000`.

## Deploy (Fly.io)

```bash
fly auth login
# edit `app` in fly.toml to a unique name first
fly launch --no-deploy --copy-config
fly secrets set GROQ_API_KEY=gsk_...
fly deploy
fly scale count 1
```

Share `https://<your-app>.fly.dev`. Keep exactly one machine: room state is in memory, so scaling out
splits players across servers. Move state to Redis before going past one instance. The Dockerfile also
works on Railway, Render, or any VPS. Never commit your API key.

## Config (env vars)

| Var | Default | What it does |
| --- | --- | --- |
| `GROQ_API_KEY` | none | Enables the bot. Without it, rooms are humans-only. |
| `GROQ_MODEL` | `openai/gpt-oss-120b` | Groq retired `llama-3.1-8b-instant` and `llama-3.3-70b-versatile` for free/dev tiers on 2026-08-16. `openai/gpt-oss-20b` is cheaper and faster. |
| `BOT_COUNT` | `1` | Bots per round. |
| `CHAT_SECONDS` / `VOTE_SECONDS` | `180` / `25` | Phase lengths. |
| `GROQ_RPM` | `25` | Most LLM calls per minute for the whole server. When it's used up the bot stays quiet for that turn, which protects your free quota. |
| `ROOMS_PER_MIN` / `CONNECTS_PER_MIN` | `10` / `60` | Create-room calls and websocket connections per IP. Generous on purpose: a whole hostel can share one public IP. |
| `MAX_ROOMS` | `200` | The server refuses new rooms past this. |
| `ROUND_LOG` | none | Path of a file that gets one JSON line per finished round. |
| `LOG_TRANSCRIPTS` | off | Set to `1` to include the full chat in the round log. Use it to see why a bot got caught. |

`GET /health` shows whether the bot is on and each persona's fool rate so far (resets on restart).

## How it works

```
app/main.py        rooms, websocket protocol, phases: lobby -> chat -> vote -> result
app/bot.py         personas, Groq call, human-like pacing (reading delay, typing indicator, typos, skips)
static/index.html  the whole frontend, no build step
```

Bots are written as bored hostel guys, not helpful assistants. The prompt asks for short, lowercase,
nonchalant Hinglish with no emojis, and every reply is then filtered: emojis and emoticons are stripped, replies are
cut to a few words, capitals, apostrophes and most punctuation are dropped, and anything that sounds like an assistant
is thrown away so the bot stays quiet instead. Bots also wait before replying, show a typing indicator for as long as a
person would take to type, skip some messages, and answer faster when named. The personas and prompt live at the top
of `app/bot.py`: that is the file to edit when playtests show what still gives the bot away.

**The bot copies the room.** Before every reply it measures how the humans in that room type (message length,
lowercase or capitals, full stops, how much Hindi they mix in, slang they repeat, emojis) and both tells the model
and filters the reply to match. Until three human messages exist it uses the persona defaults. It never says "bro" or
"bhai" unless two people in the room have said it, never copies a message word for word, and never learns from its own
messages or other bots. This lives in `app/mimic.py`.

Players who drop (phone lock, bad signal) rejoin the same room under the same nickname and pick up where
they were. Voting waits for everyone until the timer ends.

## Limits and logging

Per player: 5 chat messages in a burst, then about 1 per second. Oversized or malformed messages are
ignored or disconnected. Every finished round writes one JSON line to the server log (and to `ROUND_LOG`
if set) with the bot persona, how many players it fooled, and message counts. Real nicknames are never
logged, only the round codenames.

On Render's free tier the filesystem is wiped on restart, so a log file does not survive. Read the lines
in the Render logs, or add a hosted database (see Next).

## Tests

```bash
pip install -r requirements-dev.txt
pytest -q
```

Covers the rate limiter, the bot and Groq call handling (mocked, no key needed), a full round with a
dropped and reconnected player, the rate limits, and the install files. The same suite runs on every
push via GitHub Actions.

## Next

- [ ] Hosted Postgres (Neon) for permanent round stats and a persona leaderboard
- [ ] Redis for room state (needed for more than one instance)
- [ ] Scoreboard across rounds for the humans
- [ ] More prompts, and a prompt pack per friend group
