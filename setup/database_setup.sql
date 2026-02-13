-- Create postgres database if not exists
SELECT 'CREATE DATABASE roman_history'
WHERE NOT EXISTS (
    SELECT 1 FROM pg_database WHERE datname = 'roman_history'
);

\gexec

\c roman_history;

-- ===========================================
-- 0) Extensions
-- ===========================================
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS unaccent;
CREATE EXTENSION IF NOT EXISTS pg_trgm;


-- Create tables    

CREATE TABLE IF NOT EXISTS history(
    chunk_id SERIAL PRIMARY KEY, 
    source_id UUID NOT NULL,
    document_id UUID NOT NULL,
    raw_doc TEXT NOT NULL,
    norm_doc TEXT NOT NULL,
    compressed_doc TEXT NOT NULL,
    norm_embedding VECTOR(1024), 
    compressed_embedding VECTOR(1024),
    ner_entities TEXT[] NOT NULL,
    metadata JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);  
