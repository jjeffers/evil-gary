-- Enable the pgvector extension for similarity search
CREATE EXTENSION IF NOT EXISTS vector;

-- Create the knowledge base table
CREATE TABLE gary_knowledge (
    id TEXT PRIMARY KEY,
    content TEXT NOT NULL,
    metadata JSONB,
    embedding VECTOR(384) -- 384 dimensions for all-MiniLM-L6-v2
);

-- Create an index to speed up vector similarity searches (optional but recommended for large datasets)
-- CREATE INDEX ON gary_knowledge USING hnsw (embedding vector_cosine_ops);

-- Create the RPC function to perform similarity search
-- This function is called by rag_engine.py
CREATE OR REPLACE FUNCTION match_gary_knowledge (
  query_embedding VECTOR(384),
  match_threshold FLOAT,
  match_count INT
)
RETURNS TABLE (
  id TEXT,
  content TEXT,
  metadata JSONB,
  similarity FLOAT
)
LANGUAGE plpgsql
AS $$
BEGIN
  RETURN QUERY
  SELECT
    gary_knowledge.id,
    gary_knowledge.content,
    gary_knowledge.metadata,
    1 - (gary_knowledge.embedding <=> query_embedding) AS similarity
  FROM gary_knowledge
  WHERE 1 - (gary_knowledge.embedding <=> query_embedding) > match_threshold
  ORDER BY gary_knowledge.embedding <=> query_embedding
  LIMIT match_count;
END;
$$;
