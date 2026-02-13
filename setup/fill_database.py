import psycopg
import json
import uuid
from pgvector.psycopg import register_vector


with open('./database_data.jsonl', 'r') as f:
    data = f.readlines()
    data = [json.loads(x) for x in data]

formatted_data = [
    (
        uuid.UUID(x['source_id']),
        uuid.UUID(x['document_id']),
        x['raw_doc'],
        x['norm_doc'],
        x['compressed_doc'],
        x['norm_embedding'],
        x['compressed_embedding'],
        x['ner_entities'], 
        json.loads(x['metadata'])
    ) for x in data
]


populate_string = """\
insert into history (source_id, document_id, raw_doc, norm_doc, compressed_doc, norm_embedding, compressed_embedding, \
ner_entities, metadata) values (%s, %s, %s, %s, %s, %s, %s, %s, %s);"""

#FILL DATABASE
with psycopg.connect("dbname=roman_history") as conn:
    register_vector(conn)
    with conn.cursor() as cur:
        with conn.pipeline():
            cur.executemany(populate_string, formatted_data)


