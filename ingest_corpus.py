"""
ingest_corpus.py — The Lich's Tomb
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Reads corpus.txt (Gary Gygax forum posts), scrubs the dross of forum
metadata, and entombs the resulting wisdom within a ChromaDB vector
collection for future thaumaturgy.

Usage:
    python ingest_corpus.py [--corpus data/corpus.txt]

Col_Pladoh
"""

import argparse
import logging
import os
import re
import sys
import time
from pathlib import Path

from sentence_transformers import SentenceTransformer
from supabase import create_client, Client

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DEFAULT_CORPUS = Path("data/corpus.txt")
COLLECTION_NAME = "evil_gary"
CHUNK_SIZE = 500          # characters per chunk — optimal for verisimilitude
CHUNK_OVERLAP = 75        # overlap to preserve contextual continuity
BATCH_SIZE = 100          # upsert batch size; too large and ChromaDB weeps
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Text cleaning — strip the forum chaff, keep the Gygaxian grain
# ---------------------------------------------------------------------------

# Patterns that constitute noise unworthy of a learned myrmidon
_NOISE_PATTERNS = [
    re.compile(r"\[QUOTE[^\]]*\].*?\[/QUOTE\]", re.DOTALL | re.IGNORECASE),  # nested quotes
    re.compile(r"\[/?[A-Z]+[^\]]*\]", re.IGNORECASE),   # BBCode tags
    re.compile(r"https?://\S+"),                          # URLs
    re.compile(r"^>.+$", re.MULTILINE),                   # markdown quotes
    re.compile(r"^-{3,}$", re.MULTILINE),                 # horizontal rules
    re.compile(r"\d{1,2}[/-]\d{1,2}[/-]\d{2,4}"),        # timestamps
    re.compile(r"\d{1,2}:\d{2}(?::\d{2})?\s*(?:AM|PM)?", re.IGNORECASE),  # time
    re.compile(r"^(?:Posted by|Originally posted|Join Date|Posts:|Location:).+$",
               re.MULTILINE | re.IGNORECASE),              # forum metadata
    re.compile(r"\s{3,}", re.MULTILINE),                   # excessive whitespace
]


def clean_text(raw: str) -> str:
    """
    Purge forum detritus from raw corpus text.
    Returns text worthy of the Dungeon Master's scrutiny.
    """
    text = raw
    for pattern in _NOISE_PATTERNS:
        text = pattern.sub(" ", text)
    # Collapse runs of blank lines into a single paragraph break
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Chunking — divide the tome into manageable scrolls
# ---------------------------------------------------------------------------

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """
    Divide text into overlapping chunks of `chunk_size` characters.
    Overlap preserves contextual continuity across the seams — much as a
    skilled DM bridges session transitions with a brief recap.
    """
    chunks: list[str] = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = min(start + chunk_size, text_len)
        chunk = text[start:end].strip()
        if len(chunk) > 50:   # discard trivially small fragments
            chunks.append(chunk)
        start += chunk_size - overlap

    return chunks


# ---------------------------------------------------------------------------
# Ingestion orchestrator
# ---------------------------------------------------------------------------

def ingest(corpus_path: Path) -> None:
    """
    Main ingestion pipeline.  Reads, cleans, chunks, embeds, and stores.
    The entire ritual completes faster than a grognard can dispute THAC0.
    """
    # ── 1. Verify the corpus scroll exists ──────────────────────────────────
    if not corpus_path.exists():
        log.error(
            "Corpus not found at '%s'. "
            "That particular arcane lore was lost in the fires of Lake Geneva.",
            corpus_path,
        )
        sys.exit(1)

    log.info("Opening the Lich's Tomb: %s", corpus_path)
    raw_text = corpus_path.read_text(encoding="utf-8", errors="replace")
    log.info("Corpus loaded: %d characters across %d lines.",
             len(raw_text), raw_text.count("\n"))

    # ── 2. Clean ─────────────────────────────────────────────────────────────
    log.info("Purging forum dross — separating wheat from chaff…")
    clean = clean_text(raw_text)
    log.info("Cleaned corpus: %d characters remain.", len(clean))

    # ── 3. Chunk ─────────────────────────────────────────────────────────────
    log.info("Dividing the tome into %d-character scrolls (overlap: %d)…",
             CHUNK_SIZE, CHUNK_OVERLAP)
    chunks = chunk_text(clean, CHUNK_SIZE, CHUNK_OVERLAP)
    log.info("Produced %d chunks.", len(chunks))

    # ── 4. Initialise Supabase & Model ───────────────────────────────────────
    from dotenv import load_dotenv
    load_dotenv()
    
    supabase_url = os.environ.get("SUPABASE_URL")
    supabase_key = os.environ.get("SUPABASE_SERVICE_KEY")
    if not supabase_url or not supabase_key:
        log.error("Missing SUPABASE_URL or SUPABASE_SERVICE_KEY in environment.")
        sys.exit(1)

    log.info("Connecting to Supabase at '%s'…", supabase_url)
    supabase: Client = create_client(supabase_url, supabase_key)

    log.info("Loading sentence-transformers model 'all-MiniLM-L6-v2'…")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # ── 5. Batch upsert ───────────────────────────────────────────────────────
    log.info("Entombing %d chunks into Supabase in batches of %d…",
             len(chunks), BATCH_SIZE)
    t0 = time.perf_counter()
    total_upserted = 0

    for batch_start in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[batch_start: batch_start + BATCH_SIZE]
        
        # Generate embeddings directly using sentence-transformers
        embeddings = model.encode(batch).tolist()
        
        records = []
        for i, chunk in enumerate(batch):
            records.append({
                "id": f"gary_{batch_start + i:06d}",
                "content": chunk,
                "metadata": {
                    "chunk_index": batch_start + i, 
                    "char_count": len(chunk), 
                    "source": "corpus.txt"
                },
                "embedding": embeddings[i]
            })

        supabase.table("gary_knowledge").upsert(records).execute()
        total_upserted += len(batch)
        log.info("  Upserted batch ending at chunk %d / %d",
                 batch_start + len(batch), len(chunks))

    elapsed = time.perf_counter() - t0
    log.info(
        "Ingestion complete.  %d chunks entombed in %.1f seconds.  "
        "The Lich's Tomb is sealed and ready for consultation.  Cheers, Gary.",
        total_upserted, elapsed,
    )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Ingest the Evil Gary corpus into ChromaDB.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS,
                   help=f"Path to corpus.txt (default: {DEFAULT_CORPUS})")
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    ingest(corpus_path=args.corpus)
