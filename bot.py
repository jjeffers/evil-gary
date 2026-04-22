"""
bot.py — The Tabletop
~~~~~~~~~~~~~~~~~~~~~~~
Discord bot interface for Evil Gary.  Exposes:

  /ask <question>  — Direct, synchronous query to the RAG engine.
  Passive mode     — Listens in configured channels and responds to
                     messages addressed to Gary (mention or keyword).
  /admin usage     — Shows token consumption from the ledger.

All error messages are delivered in character.  The dice may not always
favour the petitioner, but Gary's dignity remains intact.

Usage:
    python bot.py

Environment variables required:
    DISCORD_TOKEN      — Your bot's secret token
    OPENROUTER_API_KEY — OpenRouter API key
    PASSIVE_CHANNEL_IDS — Comma-separated Discord channel IDs for passive mode
                          (optional; e.g. "123456789,987654321")

Col_Pladoh
"""

from __future__ import annotations

import asyncio
import datetime
import logging
import os
import random
import sys
from pathlib import Path

import discord
from discord import app_commands
from discord.ext import commands, tasks

from rag_engine import GaryRAGEngine
from token_logger import TokenLogger

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
Path("logs").mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/evil_gary.log", encoding="utf-8"),
    ],
)
log = logging.getLogger("evil_gary.bot")

# ---------------------------------------------------------------------------
# Configuration from environment
# ---------------------------------------------------------------------------
DISCORD_TOKEN: str = os.environ.get("DISCORD_TOKEN", "")
if not DISCORD_TOKEN:
    log.critical(
        "DISCORD_TOKEN not set.  The portal to the Discord realm is barred.  "
        "Set this variable and try again, young myrmidon."
    )
    sys.exit(1)

_raw_channels = os.environ.get("PASSIVE_CHANNEL_IDS", "")
PASSIVE_CHANNELS: set[int] = set()
if _raw_channels:
    for cid in _raw_channels.split(","):
        cid = cid.strip()
        if cid.isdigit():
            PASSIVE_CHANNELS.add(int(cid))

# ---------------------------------------------------------------------------
# In-character error / status messages
# ---------------------------------------------------------------------------
ERRORS = {
    "api_unreachable": (
        "The dice favour you not this day; the API is unreachable.  "
        "Perhaps try again once the stars realign.  Col_Pladoh"
    ),
    "timeout": (
        "I find myself detained by forces beyond the Prime Material Plane.  "
        "The response took overlong and was cut short.  Cheers, Gary"
    ),
    "no_corpus": (
        "That particular arcane lore was lost in the fires of Lake Geneva.  "
        "I cannot speak to this matter with the verisimilitude it deserves.  "
        "Col_Pladoh"
    ),
    "generic": (
        "An unknown perturbation has disrupted my thaumaturgy.  "
        "Pray, submit your query once more.  Cheers, Gary"
    ),
}

GARY_KEYWORDS = {"gary", "gygax", "col_pladoh", "evil gary"}

# ---------------------------------------------------------------------------
# Bot setup
# ---------------------------------------------------------------------------
intents = discord.Intents.default()
intents.message_content = True   # required for passive mode

bot = commands.Bot(command_prefix="!", intents=intents)
tree = bot.tree

# Lazily initialised on ready so we have an event loop
_engine: GaryRAGEngine | None = None
_logger: TokenLogger | None = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _engine_ready() -> bool:
    return _engine is not None


def _format_response(rag_answer: str, retrieval_ms: float) -> str:
    """
    Wrap Gary's answer with a subtle performance footer (DM-visible only if
    you are curious; invisible to the casual reader).
    """
    # Discord messages cap at 2 000 characters; trim gracefully
    MAX = 1900
    if len(rag_answer) > MAX:
        rag_answer = rag_answer[:MAX] + "… *(the parchment runs out)*"
    return rag_answer


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------

@bot.event
async def on_ready() -> None:
    global _engine, _logger
    log.info("Evil Gary awakens as %s (ID: %s).", bot.user, bot.user.id)

    _logger = TokenLogger()

    try:
        _engine = GaryRAGEngine()
    except Exception as exc:
        log.exception(
            "Failed to raise the Lich's Tomb: %s.  "
            "Ensure ingest_corpus.py has been run.", exc
        )
        # Bot stays online but /ask will return an in-character error

    try:
        synced = await tree.sync()
        log.info("Synced %d slash commands.", len(synced))
    except Exception as exc:
        log.exception("Slash command sync failed: %s", exc)

    if PASSIVE_CHANNELS:
        log.info("Passive listening enabled in channels: %s", PASSIVE_CHANNELS)
        if not auto_ingest_discord.is_running():
            auto_ingest_discord.start()
    log.info("Gary is ready to adjudicate.  Cheers, Gary.")


@bot.event
async def on_message(message: discord.Message) -> None:
    """
    Passive listener.  Responds in configured channels when Gary is addressed
    by name or by @mention.
    """
    # Ignore self
    if message.author == bot.user:
        return

    # Always process prefix commands
    await bot.process_commands(message)

    # Passive mode guard
    if message.channel.id not in PASSIVE_CHANNELS:
        return
    if not _engine_ready():
        return

    content_lower = message.content.lower()
    mentioned = bot.user in message.mentions
    keyword_match = any(kw in content_lower for kw in GARY_KEYWORDS)
    
    interject = random.random() < 0.05

    if not (mentioned or keyword_match or interject):
        return

    # If it's purely a random interjection, 50% chance to just react with an emoji quietly
    if interject and not (mentioned or keyword_match):
        if random.random() < 0.5:
            # Emojis: d20, dragon, dagger, scroll, shield, skull
            emojis = ["\U0001F3B2", "\U0001F409", "\U0001F5E1\U0000FE0F", "\U0001F4DC", "\U0001F6E1\U0000FE0F", "\U0001F480"]
            try:
                await message.add_reaction(random.choice(emojis))
                log.info("Interjecting with a Gygaxian emoji.")
            except Exception as e:
                log.warning("Failed to add reaction: %s", e)
            return

    # Strip the mention prefix if present so Gary doesn't answer "who are you"
    query = message.content
    if bot.user.mention in query:
        query = query.replace(bot.user.mention, "").strip()
    if not query:
        query = "Greet the assembled adventurers in your customary fashion."

    # Give Gary some context if he is voluntarily chiming in
    if interject and not (mentioned or keyword_match):
        query = f"(System Note: You are unsolicitedly chiming in on this conversation. Be brief and opinionated.) {query}"

    async with message.channel.typing():
        await _handle_query(message.channel, query,
                            user_id=message.author.id,
                            guild_id=message.guild.id if message.guild else None)


# ---------------------------------------------------------------------------
# Slash commands
# ---------------------------------------------------------------------------

@tree.command(name="ask", description="Put a question to Gary Gygax himself.")
@app_commands.describe(question="Your question for the Master of Dungeons & Dragons.")
async def ask_command(interaction: discord.Interaction, question: str) -> None:
    """
    /ask <question> — The primary interface for consulting the Archmage.
    """
    if not _engine_ready():
        await interaction.response.send_message(ERRORS["api_unreachable"], ephemeral=True)
        return

    await interaction.response.defer(thinking=True)

    try:
        result = await _engine.ask(question)
        if _logger:
            _logger.record(
                model=_engine.chat_model,
                prompt_tokens=result.prompt_tokens,
                completion_tokens=result.completion_tokens,
                user_id=interaction.user.id,
                guild_id=interaction.guild_id,
                query_preview=question,
            )
        answer = _format_response(result.answer, result.retrieval_ms)
        await interaction.followup.send(answer)

    except TimeoutError:
        await interaction.followup.send(ERRORS["timeout"])
    except Exception as exc:
        log.exception("Error processing /ask: %s", exc)
        await interaction.followup.send(ERRORS["generic"])


@tree.command(name="admin", description="Administrative commands for the guild master.")
@app_commands.describe(subcommand="Subcommand: 'usage' to view token consumption.")
@app_commands.checks.has_permissions(administrator=True)
async def admin_command(interaction: discord.Interaction, subcommand: str) -> None:
    """
    /admin usage — Display the token ledger summary.
    Only guild administrators may consult the guild treasury.
    """
    if subcommand.lower() == "usage":
        if _logger is None:
            await interaction.response.send_message(
                "The ledger has not yet been opened.  Pray, wait for Gary to awaken fully.",
                ephemeral=True,
            )
            return
        stats = _logger.summary()
        msg = (
            f"**Token Ledger — Guild Treasury Report**\n"
            f"Total API calls: `{stats['calls']:,}`\n"
            f"Total tokens consumed: `{stats['total_tokens']:,}`\n\n"
            f"*Spend your gold pieces wisely, young myrmidon.  Col_Pladoh*"
        )
        await interaction.response.send_message(msg, ephemeral=True)
    else:
        await interaction.response.send_message(
            f"Unknown subcommand '{subcommand}'.  Available: `usage`.", ephemeral=True
        )


# ---------------------------------------------------------------------------
# Shared query handler (used by both passive and slash flows)
# ---------------------------------------------------------------------------

async def _handle_query(
    channel: discord.abc.Messageable,
    query: str,
    *,
    user_id: int | None = None,
    guild_id: int | None = None,
) -> None:
    try:
        assert _engine is not None
        result = await _engine.ask(query)
        if _logger:
            _logger.record(
                model=_engine.chat_model,
                prompt_tokens=result.prompt_tokens,
                completion_tokens=result.completion_tokens,
                user_id=user_id,
                guild_id=guild_id,
                query_preview=query,
            )
        answer = _format_response(result.answer, result.retrieval_ms)
        await channel.send(answer)  # type: ignore[arg-type]

    except TimeoutError:
        await channel.send(ERRORS["timeout"])  # type: ignore[arg-type]
    except Exception as exc:
        log.exception("Error in passive handler: %s", exc)
        await channel.send(ERRORS["generic"])  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Background Tasks
# ---------------------------------------------------------------------------

@tasks.loop(hours=24)
async def auto_ingest_discord() -> None:
    """
    Daily incremental ingestion of Discord chat logs.
    Re-uses the existing RAG engine's database and sentence-transformer model
    to avoid memory swap thrashing.
    """
    if not _engine_ready():
        return
        
    log.info("Starting daily auto-ingestion of Discord logs...")
    
    assert _engine is not None
    supabase = _engine._supabase
    model = _engine._model

    try:
        # Query Supabase for the most recent chat log timestamp
        response = supabase.table("gary_knowledge").select("metadata").eq("metadata->>type", "chat_log").order("metadata->timestamp", desc=True).limit(1).execute()
        if response.data and response.data[0].get("metadata", {}).get("timestamp"):
            last_timestamp_str = response.data[0]["metadata"]["timestamp"]
            cutoff_date = datetime.datetime.fromisoformat(last_timestamp_str)
            log.info(f"Last ingested message timestamp found: {cutoff_date}")
        else:
            cutoff_date = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=2)
            log.info("No previous chat logs found. Defaulting cutoff to 2 days ago.")
    except Exception as e:
        log.warning(f"Failed to fetch last timestamp from Supabase: {e}. Defaulting to 2 days ago.")
        cutoff_date = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=2)
    
    BATCH_SIZE = 100
    
    os.makedirs("dumps", exist_ok=True)
    
    for ch_id in PASSIVE_CHANNELS:
        channel = bot.get_channel(ch_id)
        if channel is None:
            try:
                channel = await bot.fetch_channel(ch_id)
            except Exception as e:
                log.warning(f"Could not access channel {ch_id} for ingestion: {e}")
                continue
                
        log.info(f"Auto-ingesting channel: '{channel.name}' from {cutoff_date} forwards...")
        
        safe_name = "".join(c for c in channel.name if c.isalnum() or c in ("-", "_")).rstrip()
        dump_file = open(f"dumps/{safe_name}_{ch_id}_daily.txt", "w", encoding="utf-8")
        
        messages_batch = []
        total_upserted = 0
        
        try:
            async for msg in channel.history(limit=None, after=cutoff_date, oldest_first=True):
                if msg.author.bot or not msg.content.strip():
                    continue
                    
                content = msg.content.strip()
                if len(content) < 15:
                    continue
                    
                formatted_text = f"User '{msg.author.display_name}' said: {content}"
                dump_file.write(f"[{msg.created_at.isoformat()}] {formatted_text}\n")
                
                messages_batch.append({
                    "id": f"discord_{msg.id}",
                    "content": formatted_text,
                    "metadata": {
                        "source": f"discord:{channel.name}",
                        "author": msg.author.display_name,
                        "timestamp": msg.created_at.isoformat(),
                        "type": "chat_log"
                    }
                })
                
                if len(messages_batch) >= BATCH_SIZE:
                    texts = [b["content"] for b in messages_batch]
                    embeddings = await asyncio.to_thread(model.encode, texts)
                    for i, emb in enumerate(embeddings.tolist()):
                        messages_batch[i]["embedding"] = emb
                        
                    def _upsert(b=messages_batch):
                        supabase.table("gary_knowledge").upsert(b).execute()
                        
                    await asyncio.to_thread(_upsert)
                    total_upserted += len(messages_batch)
                    messages_batch = []
                    
            if messages_batch:
                texts = [b["content"] for b in messages_batch]
                embeddings = await asyncio.to_thread(model.encode, texts)
                for i, emb in enumerate(embeddings.tolist()):
                    messages_batch[i]["embedding"] = emb
                    
                def _upsert(b=messages_batch):
                    supabase.table("gary_knowledge").upsert(b).execute()
                    
                await asyncio.to_thread(_upsert)
                total_upserted += len(messages_batch)
                
            log.info(f"Finished auto-ingest for '{channel.name}'. Upserted {total_upserted} messages.")
            dump_file.close()
            
        except Exception as e:
            log.exception(f"Error auto-ingesting history for channel {ch_id}: {e}")
            dump_file.close()
            
    log.info("Daily auto-ingestion completed.")

@auto_ingest_discord.before_loop
async def before_auto_ingest_discord():
    await bot.wait_until_ready()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    bot.run(DISCORD_TOKEN, log_handler=None)  # log_handler=None = use our config
