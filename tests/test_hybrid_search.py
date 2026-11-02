"""Offline tests for the hybrid lexical+semantic search path and the embedding build.

The regression index has no vectors, so without these tests the semantic half of
Knowledge.search and all of build_embeddings.py never run. Everything here uses
hand-made vectors and a stub encoder: no network and no model download.
"""
import contextlib
import io
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

import build_embeddings
from knowledge import Knowledge, build

DOC = '''Alpha
=====

The widget handles requests.

Beta
====

The gadget handles responses.

Gamma
=====

The sprocket handles retries.
'''
MODEL = 'stub-model'
DIMENSIONS = 4
BASIS = {'Alpha': 0, 'Beta': 1, 'Gamma': 2}


def basis_vector(index, size=DIMENSIONS):
    vector = np.zeros(size, dtype=np.float32)
    vector[index] = 1.0
    return vector


def build_index(root, libraries):
    """Index DOC once per (library, version) and return the database path."""
    (root / 'doc.rst').write_text(DOC)
    entries = [{'name': 'doc', 'path': 'doc.rst', 'library': library, 'version': version, 'kind': 'documentation'}
               for library, version in libraries]
    manifest = root / 'corpus.json'
    manifest.write_text(json.dumps({'sources': entries}))
    database = root / 'index.sqlite3'
    build(manifest, database)
    return database


class StubEmbedder:
    """Stands in for fastembed's query encoder."""

    def __init__(self, vector):
        self.vector = vector

    def query_embed(self, query):
        return iter([self.vector])


class ForbiddenEmbedder:
    def query_embed(self, query):
        raise AssertionError('the query encoder must not run without a usable vector index')


class HybridSearchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        # A neighbour with the same version and a sibling with the same library, so each
        # scoping filter (library, version) is exercised on its own.
        self.db = build_index(self.root, [('kokkos', 'snapshot-test'), ('kokkos', 'snapshot-other'),
                                          ('kokkos-kernels', 'snapshot-test')])
        con = sqlite3.connect(self.db)
        self.ids = {}
        for unit_id, library, version, title in con.execute('SELECT id,library,version,title FROM units'):
            self.ids.setdefault((library, version), {})[title] = unit_id
        con.close()

    def tearDown(self):
        self.temp.cleanup()

    def add_vectors(self, scope=('kokkos', 'snapshot-test'), model=MODEL, dimensions=DIMENSIONS):
        """Store a basis vector for each section, the way build_embeddings would."""
        rows = [(self.ids[scope][title], model, basis_vector(index, dimensions).tobytes())
                for title, index in BASIS.items()]
        con = sqlite3.connect(self.db)
        con.executemany('INSERT OR REPLACE INTO vectors(unit_id,model,vector) VALUES (?,?,?)', rows)
        con.executemany('INSERT OR REPLACE INTO embedding_meta(key,value) VALUES (?,?)',
                        [('model', model), ('dimensions', str(dimensions))])
        con.commit()
        con.close()

    def execute(self, statement, *parameters):
        con = sqlite3.connect(self.db)
        con.execute(statement, parameters)
        con.commit()
        con.close()

    def search(self, query, encoder, library='kokkos', version='snapshot-test'):
        store = Knowledge(self.db)  # a fresh store per call: the vector cache is per instance
        store._embedder = encoder
        return store.search(query, 'documentation', version=version, library=library)

    def test_index_without_embedding_metadata_is_lexical_only(self):
        hits = self.search('widget', ForbiddenEmbedder())
        self.assertEqual([hit['title'] for hit in hits], ['Alpha'])
        self.assertEqual(hits[0]['retrieved_by'], ['lexical'])

    def test_index_without_vector_tables_is_lexical_only(self):
        # Older hpc-2 indexes have neither table.
        self.execute('DROP TABLE vectors')
        self.execute('DROP TABLE embedding_meta')
        hits = self.search('widget', ForbiddenEmbedder())
        self.assertEqual([hit['title'] for hit in hits], ['Alpha'])
        self.assertEqual(hits[0]['retrieved_by'], ['lexical'])

    def test_semantic_only_unit_is_surfaced_after_the_lexical_match(self):
        self.add_vectors()
        # "widget" matches only Alpha lexically, but the query vector points at Gamma.
        hits = self.search('widget', StubEmbedder(basis_vector(BASIS['Gamma'])))
        by_title = {hit['title']: hit for hit in hits}
        self.assertEqual(by_title['Gamma']['retrieved_by'], ['semantic'])
        self.assertEqual(by_title['Alpha']['retrieved_by'], ['lexical', 'semantic'])
        titles = [hit['title'] for hit in hits]
        self.assertLess(titles.index('Alpha'), titles.index('Gamma'))

    def test_agreeing_channels_rank_the_shared_hit_first(self):
        self.add_vectors()
        hits = self.search('widget', StubEmbedder(basis_vector(BASIS['Alpha'])))
        self.assertEqual(hits[0]['title'], 'Alpha')
        self.assertEqual(hits[0]['retrieved_by'], ['lexical', 'semantic'])

    def test_unusable_query_vector_falls_back_to_lexical(self):
        self.add_vectors()
        unusable = {'wrong dimension': np.ones(DIMENSIONS - 1, dtype=np.float32),
                    'zero vector': np.zeros(DIMENSIONS, dtype=np.float32)}
        for label, vector in unusable.items():
            with self.subTest(label):
                hits = self.search('widget', StubEmbedder(vector))
                self.assertEqual([hit['title'] for hit in hits], ['Alpha'])
                self.assertEqual(hits[0]['retrieved_by'], ['lexical'])

    def test_vector_of_the_wrong_size_is_an_error(self):
        self.add_vectors()
        self.execute('UPDATE vectors SET vector=? WHERE unit_id=?',
                     basis_vector(0, DIMENSIONS - 1).tobytes(), self.ids[('kokkos', 'snapshot-test')]['Alpha'])
        with self.assertRaisesRegex(ValueError, 'invalid size'):
            self.search('widget', StubEmbedder(basis_vector(0)))

    def test_vectors_of_other_libraries_never_leak_into_a_scoped_search(self):
        self.add_vectors(('kokkos-kernels', 'snapshot-test'))
        hits = self.search('widget', StubEmbedder(basis_vector(BASIS['Gamma'])))
        self.assertTrue(hits)
        self.assertTrue(all(hit['library'] == 'kokkos' for hit in hits))
        self.assertTrue(all(hit['retrieved_by'] == ['lexical'] for hit in hits))

    def test_vectors_of_other_versions_never_leak_into_a_scoped_search(self):
        self.add_vectors(('kokkos', 'snapshot-other'))
        hits = self.search('widget', StubEmbedder(basis_vector(BASIS['Gamma'])))
        self.assertTrue(hits)
        self.assertTrue(all(hit['version'] == 'snapshot-test' for hit in hits))
        self.assertTrue(all(hit['retrieved_by'] == ['lexical'] for hit in hits))

    def test_vectors_from_another_model_are_ignored(self):
        self.add_vectors(model='some-other-model')
        self.execute("UPDATE embedding_meta SET value=? WHERE key='model'", MODEL)
        hits = self.search('widget', StubEmbedder(basis_vector(BASIS['Gamma'])))
        self.assertEqual([hit['title'] for hit in hits], ['Alpha'])
        self.assertEqual(hits[0]['retrieved_by'], ['lexical'])


class FakeTextEmbedding:
    """Replaces fastembed.TextEmbedding; yields one basis-style vector per passage."""
    dimension = build_embeddings.DIMENSIONS
    withheld = 0

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def passage_embed(self, passages, batch_size, parallel):
        vectors = []
        for index, _ in enumerate(passages):
            vector = np.zeros(self.dimension, dtype=np.float32)
            vector[index % self.dimension] = index + 1  # deliberately not unit length
            vectors.append(vector)
        return iter(vectors[:len(vectors) - self.withheld])


class ShortTextEmbedding(FakeTextEmbedding):
    withheld = 1


class NarrowTextEmbedding(FakeTextEmbedding):
    dimension = 8


class BuildEmbeddingsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.db = build_index(self.root, [('kokkos', 'snapshot-test')])
        self.units = self.query('SELECT count(*) FROM units')[0][0]

    def tearDown(self):
        self.temp.cleanup()

    def query(self, statement, *parameters):
        con = sqlite3.connect(self.db)
        try:
            return con.execute(statement, parameters).fetchall()
        finally:
            con.close()

    def execute(self, statement):
        con = sqlite3.connect(self.db)
        con.execute(statement)
        con.commit()
        con.close()

    def run_build(self, embedder=FakeTextEmbedding):
        # build_embeddings prints progress; keep the test log clean.
        with mock.patch.object(build_embeddings, 'TextEmbedding', embedder), contextlib.redirect_stdout(io.StringIO()):
            return build_embeddings.build_embeddings(self.db, self.root / 'model-cache')

    def snapshot(self):
        return (self.query('SELECT unit_id,model,vector FROM vectors ORDER BY unit_id'),
                self.query('SELECT key,value FROM embedding_meta ORDER BY key'))

    def test_stores_one_normalised_vector_per_unit_with_metadata(self):
        metadata = self.run_build()
        rows = self.query('SELECT unit_id,model,vector FROM vectors')
        self.assertEqual(len(rows), self.units)
        for _, model, blob in rows:
            self.assertEqual(model, build_embeddings.MODEL)
            vector = np.frombuffer(blob, dtype=np.float32)
            self.assertEqual(vector.size, build_embeddings.DIMENSIONS)
            self.assertAlmostEqual(float(np.linalg.norm(vector)), 1.0, places=5)
        stored = dict(self.query('SELECT key,value FROM embedding_meta'))
        self.assertEqual(stored['model'], build_embeddings.MODEL)
        self.assertEqual(stored['dimensions'], str(build_embeddings.DIMENSIONS))
        self.assertEqual(stored['unit_count'], str(self.units))
        self.assertEqual(metadata['unit_count'], str(self.units))

    def test_rebuilding_replaces_rather_than_duplicates(self):
        self.run_build()
        self.run_build()
        self.assertEqual(self.query('SELECT count(*) FROM vectors')[0][0], self.units)

    def test_vector_count_mismatch_rolls_back_and_keeps_previous_vectors(self):
        self.run_build()
        before = self.snapshot()
        with self.assertRaisesRegex(ValueError, 'vectors for'):
            self.run_build(ShortTextEmbedding)
        self.assertEqual(self.snapshot(), before)

    def test_wrong_embedding_dimension_is_rejected_and_rolled_back(self):
        with self.assertRaisesRegex(ValueError, 'Unexpected embedding shape'):
            self.run_build(NarrowTextEmbedding)
        self.assertEqual(self.query('SELECT count(*) FROM vectors')[0][0], 0)

    def test_unsupported_schema_is_rejected(self):
        self.execute("UPDATE meta SET value='hpc-1' WHERE key='schema'")
        with self.assertRaisesRegex(ValueError, 'Unsupported index schema'):
            self.run_build()

    def test_missing_database_is_rejected(self):
        with mock.patch.object(build_embeddings, 'TextEmbedding', FakeTextEmbedding):
            with self.assertRaisesRegex(ValueError, 'Index does not exist'):
                build_embeddings.build_embeddings(self.root / 'absent.sqlite3', self.root / 'model-cache')


if __name__ == '__main__':
    unittest.main()
