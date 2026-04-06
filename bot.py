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

import logging
import os
import sys
from pathlib import Path

import discord
from discord import app_commands
from discord.ext import commands

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

    if not (mentioned or keyword_match):
        return

    # Strip the mention prefix if present so Gary doesn't answer "who are you"
    query = message.content
    if bot.user.mention in query:
        query = query.replace(bot.user.mention, "").strip()
    if not query:
        query = "Greet the assembled adventurers in your customary fashion."

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
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    bot.run(DISCORD_TOKEN, log_handler=None)  # log_handler=None = use our config
