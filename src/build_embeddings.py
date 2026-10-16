"""Build the optional dense vector index used alongside the SQLite FTS index."""
import argparse
import importlib.metadata
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from fastembed import TextEmbedding

MODEL = 'BAAI/bge-small-en-v1.5'
DIMENSIONS = 384
MAX_DOCUMENT_CHARACTERS = 600


def build_embeddings(database, cache_dir, model_name=MODEL, batch_size=48, threads=1, parallel=2):
    """Create normalized float32 passage vectors, transactionally replacing prior vectors."""
    database = Path(database).resolve()
    if not database.is_file():
        raise ValueError(f'Index does not exist: {database}')
    embedder = TextEmbedding(model_name=model_name, cache_dir=str(cache_dir), threads=threads)
    con = sqlite3.connect(database, timeout=120)
    try:
        schema = con.execute("SELECT value FROM meta WHERE key='schema'").fetchone()
        if not schema or schema[0] not in ('hpc-2', 'hpc-3'):
            raise ValueError('Unsupported index schema; rebuild it with the current ingest.py')

        con.execute('BEGIN IMMEDIATE')
        con.execute('CREATE INDEX IF NOT EXISTS units_scope ON units(library,version,kind,id)')
        con.execute('CREATE INDEX IF NOT EXISTS units_parent ON units(parent,ordinal)')
        con.execute('CREATE TABLE IF NOT EXISTS vectors(unit_id TEXT PRIMARY KEY,model TEXT NOT NULL,vector BLOB NOT NULL)')
        con.execute('CREATE TABLE IF NOT EXISTS embedding_meta(key TEXT PRIMARY KEY,value TEXT NOT NULL)')
        con.execute('DELETE FROM vectors')
        con.execute('DELETE FROM embedding_meta')
        total = con.execute('SELECT count(*) FROM units').fetchone()[0]
        count = 0
        id_cursor = con.execute('SELECT id FROM units ORDER BY id')
        text_cursor = con.execute('SELECT title,text FROM units ORDER BY id')
        passages = (f'{title}\n{text[:MAX_DOCUMENT_CHARACTERS]}' for title, text in text_cursor)
        vectors = embedder.passage_embed(passages, batch_size=batch_size, parallel=parallel)
        records = []
        for (unit_id,), vector in zip(id_cursor, vectors):
            vector = np.asarray(vector, dtype=np.float32)
            norm = float(np.linalg.norm(vector))
            if vector.size != DIMENSIONS or not norm:
                raise ValueError(f'Unexpected embedding shape or zero vector for unit {unit_id}')
            vector /= norm
            records.append((unit_id, model_name, vector.tobytes()))
            if len(records) == batch_size:
                con.executemany('INSERT INTO vectors(unit_id,model,vector) VALUES (?,?,?)', records)
                count += len(records)
                records.clear()
                if count % (batch_size * 20) == 0:
                    print(f'Embedded {count:,}/{total:,} units', flush=True)
        if records:
            con.executemany('INSERT INTO vectors(unit_id,model,vector) VALUES (?,?,?)', records)
            count += len(records)
        if count != total:
            raise ValueError(f'Embedding model returned {count} vectors for {total} units')
        print(f'Embedded {count:,}/{total:,} units', flush=True)

        metadata = {
            'model': model_name,
            'dimensions': str(DIMENSIONS),
            'dtype': 'float32',
            'normalization': 'l2',
            'document_text': f'title plus first {MAX_DOCUMENT_CHARACTERS} characters',
            'encoder': f'fastembed {importlib.metadata.version("fastembed")}',
            'unit_count': str(count),
            'created_at': datetime.now(timezone.utc).isoformat(),
        }
        con.executemany('INSERT INTO embedding_meta(key,value) VALUES (?,?)', metadata.items())
        con.commit()
        return metadata
    except Exception:
        con.rollback()
        raise
    finally:
        con.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--db', type=Path, default=Path(__file__).resolve().parent.parent / 'data' / 'nuopc.sqlite3')
    parser.add_argument('--cache-dir', type=Path, default=Path(__file__).resolve().parent.parent / 'data' / 'model-cache')
    parser.add_argument('--batch-size', type=int, default=48)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--parallel', type=int, default=2)
    args = parser.parse_args()
    if not 1 <= args.batch_size <= 256:
        parser.error('--batch-size must be from 1 to 256')
    if not 1 <= args.threads <= 8:
        parser.error('--threads must be from 1 to 8')
    if not 1 <= args.parallel <= 8:
        parser.error('--parallel must be from 1 to 8')
    print(json.dumps(build_embeddings(args.db, args.cache_dir, batch_size=args.batch_size,
                                      threads=args.threads,parallel=args.parallel), indent=2))


if __name__ == '__main__':
    main()
