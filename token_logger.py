"""
token_logger.py — The Ledger of Lake Geneva
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Tracks OpenAI API token consumption so that profligate spending does not
drain the guild treasury before the fortnight is out.

Writes a JSON-lines log to logs/token_usage.jsonl and provides a
simple summary method for /admin commands.

Col_Pladoh
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

log = logging.getLogger(__name__)

LOG_PATH = Path("logs/token_usage.jsonl")

# Pricing is no longer hardcoded as OpenRouter supports hundreds of models
# with dynamic pricing. The ledger now tracks tokens only.


class TokenLogger:
    """
    Appends a JSON record for every API call and exposes aggregate stats.
    Thread-safe for a single process; not suitable for multi-process shards.
    """

    def __init__(self, log_path: Path = LOG_PATH) -> None:
        self._path = log_path
        self._path.parent.mkdir(parents=True, exist_ok=True)
        log.info("Token ledger opened at '%s'.", self._path)

    # ── Write ─────────────────────────────────────────────────────────────────

    def record(
        self,
        *,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        user_id: int | None = None,
        guild_id: int | None = None,
        query_preview: str = "",
    ) -> None:
        """Append one usage record to the ledger."""
        # OpenRouter pricing is dynamic; we track raw usage only

        record = {
            "ts": datetime.now(tz=timezone.utc).isoformat(),
            "model": model,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "user_id": user_id,
            "guild_id": guild_id,
            "query_preview": query_preview[:120],
        }

        with self._path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")

    # ── Read / summarise ──────────────────────────────────────────────────────

    def summary(self) -> dict:
        """
        Return aggregate stats across all recorded API calls.
        Suitable for an /admin usage command.
        """
        if not self._path.exists():
            return {"calls": 0, "total_tokens": 0}

        calls = 0
        total_tokens = 0

        with self._path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                    calls += 1
                    total_tokens += row.get("total_tokens", 0)
                except json.JSONDecodeError:
                    pass  # corrupted line — press on, brave myrmidon

        return {
            "calls": calls,
            "total_tokens": total_tokens,
        }
