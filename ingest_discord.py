"""
ingest_discord.py — The Chronicler
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
[IMPORTANT] 
DO NOT run this script as a daily cron job! 
Daily incremental syncs are now handled automatically by bot.py in the background.
This script should ONLY be used manually for one-off deep historical backfills.

Connects to Discord, downloads messaging history from PASSIVE_CHANNEL_IDS
going back to a specific cutoff date, and embeds the chat logs into the
Supabase vector database.

Usage:
    python ingest_discord.py
"""

import argparse
import asyncio
import datetime
import logging
import os
import sys

import discord
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from supabase import create_client, Client

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
# The great cataclysm before which we do not search
DEFAULT_CUTOFF_DATE = datetime.datetime(2020, 8, 18, tzinfo=datetime.timezone.utc)
BATCH_SIZE = 100

def parse_channels() -> list[int]:
    raw = os.environ.get("PASSIVE_CHANNEL_IDS", "")
    chans = []
    for cid in raw.split(","):
        cid = cid.strip()
        if cid.isdigit():
            chans.append(int(cid))
    return chans

class IngestClient(discord.Client):
    def __init__(self, channels: list[int], cutoff_date: datetime.datetime):
        intents = discord.Intents.default()
        intents.message_content = True
        super().__init__(intents=intents)
        self.target_channels = channels
        self.cutoff_date = cutoff_date

    async def on_ready(self):
        log.info(f"Logged in as {self.user} (ID: {self.user.id})")
        log.info(f"Targeting channel IDs: {self.target_channels}")
        
        supabase_url = os.environ.get("SUPABASE_URL")
        supabase_key = os.environ.get("SUPABASE_SERVICE_KEY")
        if not supabase_url or not supabase_key:
            log.error("Missing SUPABASE_URL or SUPABASE_SERVICE_KEY in environment.")
            await self.close()
            return

        log.info("Connecting to Supabase...")
        supabase: Client = create_client(supabase_url, supabase_key)
        
        log.info("Loading sentence-transformers model 'all-MiniLM-L6-v2'...")
        # Load model entirely synchronously since its initialization is heavy
        model = await asyncio.to_thread(SentenceTransformer, "all-MiniLM-L6-v2")
        
        log.info("Fetching existing message IDs to prevent re-ingestion...")
        existing_ids = set()
        limit = 1000
        offset = 0
        while True:
            resp = supabase.table("gary_knowledge").select("id").like("id", "discord_%").range(offset, offset + limit - 1).execute()
            if not resp.data:
                break
            for row in resp.data:
                existing_ids.add(row["id"])
            offset += limit
            
        log.info(f"Found {len(existing_ids)} existing Discord messages in the database.")
        
        os.makedirs("dumps", exist_ok=True)
        
        for ch_id in self.target_channels:
            channel = self.get_channel(ch_id)
            if channel is None:
                try:
                    channel = await self.fetch_channel(ch_id)
                except Exception as e:
                    log.warning(f"Could not access channel {ch_id} (does the bot have access?): {e}")
                    continue
                
            log.info(f"Beginning ingestion for channel: '{channel.name}' from {self.cutoff_date} forwards...")
            
            # Open a text dump file for the channel
            safe_name = "".join(c for c in channel.name if c.isalnum() or c in ("-", "_")).rstrip()
            dump_file = open(f"dumps/{safe_name}_{ch_id}.txt", "w", encoding="utf-8")
            
            messages_batch = []
            total_upserted = 0
            
            try:
                # Iterate from the oldest messages forwards
                async for msg in channel.history(limit=None, after=self.cutoff_date, oldest_first=True):
                    # Ignore bot messages (including Gary himself) and empty messages
                    if msg.author.bot or not msg.content.strip():
                        continue
                        
                    content = msg.content.strip()
                    
                    # Skip extremely short/meaningless messages to save vector space
                    if len(content) < 15:
                        continue
                        
                    msg_id_str = f"discord_{msg.id}"
                    if msg_id_str in existing_ids:
                        continue
                        
                    # Format text specifically so Gary knows WHO said what
                    formatted_text = f"User '{msg.author.display_name}' said: {content}"
                    
                    # Write to the local text dump
                    dump_file.write(f"[{msg.created_at.isoformat()}] {formatted_text}\n")
                    
                    messages_batch.append({
                        "id": msg_id_str,
                        "content": formatted_text,
                        "metadata": {
                            "source": f"discord:{channel.name}",
                            "author": msg.author.display_name,
                            "timestamp": msg.created_at.isoformat(),
                            "type": "chat_log"
                        }
                    })
                    
                    if len(messages_batch) >= BATCH_SIZE:
                        await self.process_batch(supabase, model, messages_batch)
                        total_upserted += len(messages_batch)
                        messages_batch = []
                        
                # Flush the remains of the batch
                if messages_batch:
                    await self.process_batch(supabase, model, messages_batch)
                    total_upserted += len(messages_batch)
                    messages_batch = []
                    
                log.info(f"Finished channel '{channel.name}'. Totally upserted {total_upserted} substantive messages.")
                dump_file.close()
                
            except Exception as e:
                log.exception(f"Error fetching history for channel {ch_id}: {e}")
                dump_file.close()
            
        log.info("All historical ingestion across targeted channels is complete! Closing connection.")
        await self.close()
        
    async def process_batch(self, supabase, model, batch):
        texts = [b["content"] for b in batch]
        
        # Offload CPU-heavy sentence embedding to avoid blocking connection
        embeddings = await asyncio.to_thread(model.encode, texts)
        
        for i, emb in enumerate(embeddings.tolist()):
            batch[i]["embedding"] = emb
            
        def _upsert():
            import time
            for attempt in range(5):
                try:
                    supabase.table("gary_knowledge").upsert(batch).execute()
                    return
                except Exception as e:
                    if attempt == 4:
                        raise e
                    log.warning(f"Upsert failed, retrying in 2 seconds... ({e})")
                    time.sleep(2)
            
        # Offload sync I/O upsert
        await asyncio.to_thread(_upsert)
        log.info(f"  --> Upserted {len(batch)} message chunks into the Supabase vector store...")

def main():
    parser = argparse.ArgumentParser(description="Ingest Discord chat logs.")
    parser.add_argument("--days-back", type=int, default=3, help="Number of days back to fetch. Defaults to 3 for fast cron runs.")
    parser.add_argument("--full-history", action="store_true", help="Fetch all history back to 2020.")
    args = parser.parse_args()

    load_dotenv()
    token = os.environ.get("DISCORD_TOKEN")
    if not token:
        log.error("DISCORD_TOKEN environment variable not set. Please populate your .env file.")
        sys.exit(1)
        
    channels = parse_channels()
    if not channels:
        log.error("PASSIVE_CHANNEL_IDS not set or invalid in .env. There are no channels to index.")
        sys.exit(1)
        
    if args.full_history:
        cutoff = DEFAULT_CUTOFF_DATE
    else:
        cutoff = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=args.days_back)
        
    client = IngestClient(channels, cutoff_date=cutoff)
    log.info("Starting historical client...")
    client.run(token, log_handler=None)

if __name__ == "__main__":
    main()
