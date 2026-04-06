# Evil Gary — RAG-Powered Discord Bot

> *"The verisimilitude of a proper dungeon cannot be achieved through rules-light thaumaturgy alone."*
> — Col_Pladoh

Evil Gary replaces a legacy Markov chain processor with a Retrieval-Augmented
Generation (RAG) pipeline, grounding every response in Gary Gygax's actual
words from `corpus.txt`.

---

## Architecture

```
corpus.txt
    │
    ▼
ingest_corpus.py  ──►  ChromaDB (.chromadb/)
                              │
Discord user  ──►  bot.py  ──►  rag_engine.py  ──►  OpenRouter
                              ▲                        │
                         similarity                 Gary's
                          search                   response
                              │
                         token_logger.py  ──►  logs/token_usage.jsonl
```

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env with your DISCORD_TOKEN and OPENROUTER_API_KEY
```

### 3. Ingest the corpus

Place `corpus.txt` in the `data/` directory, then:

```bash
python ingest_corpus.py
# or specify paths explicitly:
python ingest_corpus.py --corpus data/corpus.txt --db-path .chromadb
```

This is a one-time operation (plus reruns whenever the corpus changes).
It uses a local embedding model (sentence-transformers), which runs freely on your machine.

### 4. Run the bot

```bash
python bot.py
```

---

## Discord Commands

| Command | Description |
|---|---|
| `/ask <question>` | Ask Gary a direct question |
| `/admin usage` | View token consumption (admin only) |

**Passive mode:** Add channel IDs to `PASSIVE_CHANNEL_IDS` in `.env`.
Gary will respond whenever he is @mentioned or his name appears in a message.

---

## File Structure

```
evil-gary/
├── .agent/skills/gygax-voice/SKILL.md   # Persona definition
├── data/corpus.txt                       # Source corpus (you provide this)
├── logs/
│   ├── evil_gary.log                    # Runtime log
│   └── token_usage.jsonl                # Per-call token ledger
├── .chromadb/                           # ChromaDB persistence (auto-created)
├── bot.py                               # Discord interface (Phase 3)
├── rag_engine.py                        # RAG logic (Phase 2)
├── ingest_corpus.py                     # Data ingestion (Phase 1)
├── token_logger.py                      # Cost tracking
├── requirements.txt
├── .env.example
└── README.md
```

---

## Performance Notes

- Similarity search targets **< 500 ms** to honour Discord's heartbeat.
- Retrieval typically completes in 80–200 ms on commodity hardware.
- If retrieval exceeds 400 ms a warning is emitted in the log.

---

## Constraints & Guardrails

- **No hallucinations:** If the corpus lacks an answer Gary says so in character.
- **Token logging:** Every query's token usage is recorded to `logs/token_usage.jsonl`.
- **Graceful errors:** All failures produce in-character messages, never raw stack traces.

---

*Cheers, Gary*
