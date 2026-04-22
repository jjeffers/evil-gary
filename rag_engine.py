"""
rag_engine.py — The DM's Screen
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Retrieval-Augmented Generation logic.  Given a mortal's query, this module
reaches into the Lich's Tomb (ChromaDB), retrieves the most relevant
Gygaxian utterances, and constructs a prompt worthy of the Master.

The entire operation must complete in under 500ms lest Discord's
heartbeat timeout strike us down like a fumbled saving throw.

Col_Pladoh
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from sentence_transformers import SentenceTransformer
from supabase import create_client, Client
from openai import AsyncOpenAI

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TOP_K_RESULTS = 30         # Retrieve vastly more context now that we have a massive pool
MAX_CONTEXT_CHARS = 12_000 # Expand context window for gpt-4o's massive capacity
CHAT_MODEL = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o")
MAX_TOKENS = 600           # Gary is verbose, but not infinitely so

# ---------------------------------------------------------------------------
# The Gygaxian System Prompt — the DM's Screen itself
# ---------------------------------------------------------------------------
SYSTEM_PROMPT_TEMPLATE = """\
You are Gary Gygax, the legendary co-creator of Dungeons & Dragons, speaking \
from beyond the grave. You are \
authoritative, verbose, and slightly skeptical of modern "rules-light" systems. \
You prize verisimilitude above all else. You hate magic-users.

PERSONA RULES:
- Your answers should be 1-3 sentences long.
- Never start your responses with generic fillers like "Ah,", "Well,", or "Indeed,". Start directly with your point.
- Speak with the gravitas of a myrmidon of wargaming history.
- Employ terms naturally: verisimilitude, myrmidon, grognard, fortnight, thaumaturgy, \
  appurtenance, legerdemain, prestidigitation.
- Reference AD&D 1st Edition as the pinnacle of game design.
- Be mildly dismissive of 3rd edition and beyond, though never cruel.
- Concede the elegance of classic Traveller, acknowledging how it mirrors the rugged but resilient design aesthetic of the Little Brown Books of original D&D.
- Recognize and praise James as the absolute best Traveller GM around.
- If the corpus context does not contain a relevant answer, naturally admit \
  ignorance or a lack of recollection in character based on what the user \
  specifically asked. Do NOT fabricate facts.

RETRIEVED CONTEXT (A mix of your ancient forum writings and recent Discord chat logs from the adventurers):
{context}

Answer the user's question using the above context as your primary source of \
truth. If the context contains chat logs between users, use that to inform yourself \
about their ongoing conversations and inside jokes. Weave the retrieved passages naturally \
into your authoritative response without breaking your character.
"""

# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class RAGResponse:
    answer: str
    sources: list[str] = field(default_factory=list)
    retrieval_ms: float = 0.0
    generation_ms: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def total_ms(self) -> float:
        return self.retrieval_ms + self.generation_ms


# ---------------------------------------------------------------------------
# RAG Engine
# ---------------------------------------------------------------------------

class GaryRAGEngine:
    """
    The DM's Screen: accepts a query, consults the Tomb, and conjures Gary.

    Instantiate once at bot startup; reuse across all Discord events.
    """

    def __init__(
        self,
        top_k: int = TOP_K_RESULTS,
        chat_model: str = CHAT_MODEL,
    ) -> None:
        self.top_k = top_k
        self.chat_model = chat_model
        self._openai = AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ.get("OPENROUTER_API_KEY")
        )

        supabase_url = os.environ.get("SUPABASE_URL")
        supabase_key = os.environ.get("SUPABASE_SERVICE_KEY")
        if not supabase_url or not supabase_key:
            raise ValueError("SUPABASE_URL or SUPABASE_SERVICE_KEY not set in environment.")

        log.info("Connecting to Supabase…")
        self._supabase: Client = create_client(supabase_url, supabase_key)

        log.info("Loading sentence-transformers model 'all-MiniLM-L6-v2'…")
        self._model = SentenceTransformer("all-MiniLM-L6-v2")

        log.info("The DM's Screen is raised. Cheers, Gary.")

    # ── Retrieval ─────────────────────────────────────────────────────────────

    def _retrieve(self, query: str) -> tuple[list[str], float]:
        """
        Query Supabase pgvector for the most relevant Gygaxian fragments.
        Returns (list_of_document_strings, elapsed_ms).
        Must complete well under 500ms to honour Discord's heartbeat.
        """
        import re
        t0 = time.perf_counter()
        
        # 1. Embed query
        query_embedding = self._model.encode([query]).tolist()[0]
        
        # 2. Similarity search using Supabase RPC
        response = self._supabase.rpc(
            "match_gary_knowledge",
            {
                "query_embedding": query_embedding,
                "match_threshold": 0.0,
                "match_count": self.top_k,
            }
        ).execute()
        
        docs: list[str] = [doc["content"] for doc in response.data] if response.data else []

        # 3. Exact match for session numbers (fallback for embeddings that fail on numerical IDs)
        session_matches = re.findall(r'(?:session|#)\s*(\d+)', query, re.IGNORECASE)
        if session_matches:
            # Deduplicate numbers
            for num in set(session_matches):
                exact = self._supabase.table("gary_knowledge").select("content").ilike("content", f"%#{num}%").limit(5).execute()
                if exact.data:
                    # Prepend exact matches
                    docs = [d["content"] for d in exact.data if d["content"] not in docs] + docs

        elapsed_ms = (time.perf_counter() - t0) * 1000

        if elapsed_ms > 400:
            log.warning(
                "Retrieval took %.1f ms — dangerously close to Discord's "
                "heartbeat limit.  The dice favour you not this day.",
                elapsed_ms,
            )
        else:
            log.debug("Retrieval completed in %.1f ms.", elapsed_ms)

        return docs, elapsed_ms

    # ── Context assembly ──────────────────────────────────────────────────────

    @staticmethod
    def _build_context(docs: list[str]) -> str:
        """
        Concatenate retrieved passages into a context block, respecting the
        character budget so we do not exhaust our token coffers.
        """
        snippets: list[str] = []
        total = 0
        for i, doc in enumerate(docs, start=1):
            entry = f"[Passage {i}]\n{doc.strip()}"
            if total + len(entry) > MAX_CONTEXT_CHARS:
                break
            snippets.append(entry)
            total += len(entry)
        return "\n\n".join(snippets) if snippets else "(No relevant passages found.)"

    # ── Generation ────────────────────────────────────────────────────────────

    async def _generate(self, query: str, context: str) -> tuple[str, float, int, int]:
        """
        Call the OpenAI Chat Completions API with the assembled prompt.
        Returns (answer_text, elapsed_ms, prompt_tokens, completion_tokens).
        """
        system_prompt = SYSTEM_PROMPT_TEMPLATE.format(context=context)

        t0 = time.perf_counter()
        response = await self._openai.chat.completions.create(
            model=self.chat_model,
            max_tokens=MAX_TOKENS,
            temperature=0.7,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query},
            ],
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000

        answer = response.choices[0].message.content or ""
        p_tokens = response.usage.prompt_tokens if response.usage else 0
        c_tokens = response.usage.completion_tokens if response.usage else 0

        return answer, elapsed_ms, p_tokens, c_tokens

    # ── Public interface ──────────────────────────────────────────────────────

    async def ask(self, query: str) -> RAGResponse:
        """
        Full RAG pipeline: retrieve → assemble → generate → return.

        This is the sole public method Discord's bot logic needs to call.
        """
        log.info("Processing query: %.80s…", query)

        # Step 1: Retrieve
        docs, retrieval_ms = self._retrieve(query)

        # Step 2: Assemble context
        context = self._build_context(docs)
        log.debug("Context assembled (%d chars).", len(context))

        # Step 3: Generate
        answer, generation_ms, p_tok, c_tok = await self._generate(query, context)

        log.info(
            "Response generated in %.0f ms retrieval + %.0f ms generation "
            "(%d prompt + %d completion tokens).",
            retrieval_ms, generation_ms, p_tok, c_tok,
        )

        return RAGResponse(
            answer=answer,
            sources=docs,
            retrieval_ms=retrieval_ms,
            generation_ms=generation_ms,
            prompt_tokens=p_tok,
            completion_tokens=c_tok,
        )
