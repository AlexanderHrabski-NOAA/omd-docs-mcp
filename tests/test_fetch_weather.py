"""Offline unit tests for the weather/workflow documentation imports."""
import json
import tempfile
import unittest
from unittest import mock
from pathlib import Path

from build_manifest import FETCHED_LIBRARIES
from knowledge import LIBRARIES, build, markup_sections


WEATHER_LIBRARIES = ('dask', 'dask-jobqueue', 'ecflow', 'ufs-weather-model',
                     'mom6', 'cice', 'fv3', 'mpas')


class WeatherRegistrationTests(unittest.TestCase):
    def test_libraries_are_registered(self):
        for library in WEATHER_LIBRARIES:
            with self.subTest(library=library):
                self.assertIn(library, LIBRARIES)

    def test_mom6_is_fetched_at_build_time(self):
        import fetch_weather

        self.assertIn('mom6', FETCHED_LIBRARIES)
        self.assertIn('mom6', fetch_weather.MOM6_SITES)
        self.assertEqual(fetch_weather.MOM6_SITES['mom6']['base_url'],
                         'https://mom6.readthedocs.io/en/main/')
        self.assertEqual(fetch_weather.MOM6_SITES['mom6']['start_pages'], ('index.html',))
        self.assertEqual(fetch_weather.MOM6_SITES['mom6']['version'], 'snapshot-main')

    def test_snapshot_pins_cover_seven_repositories(self):
        import fetch_weather

        self.assertEqual(set(fetch_weather.SNAPSHOT),
                         set(WEATHER_LIBRARIES) - {'mom6'})
        for library, config in fetch_weather.SNAPSHOT.items():
            with self.subTest(library=library):
                for key in ('repo', 'ref', 'commit', 'version', 'doc_dir', 'exts', 'licenses'):
                    self.assertIn(key, config)
                self.assertRegex(config['commit'], r'^[0-9a-f]{40}$')
                self.assertTrue(config['version'])

    def test_rolling_snapshot_labels(self):
        import fetch_weather

        # MPAS docs repo has no release tags: rolling snapshot label, release stays informational.
        mpas = fetch_weather.SNAPSHOT['mpas']
        self.assertTrue(mpas['version'].startswith('snapshot-'))
        self.assertEqual(mpas['target_release'], '8.4.2')
        self.assertEqual(mpas['licenses'], [])

    def test_tag_to_version_transforms(self):
        import fetch_weather

        cases = {
            'dask': ('2026.8.0', '2026.8.0'),
            'dask-jobqueue': ('0.9.0', '0.9.0'),
            'ecflow': ('5.19.0', '5.19.0'),
            'ufs-weather-model': ('Release/P8b', 'P8b'),
            'cice': ('CICE6.6.3', '6.6.3'),
            'fv3': ('FV3-202604-public', '202604'),
        }
        for library, (tag, version) in cases.items():
            with self.subTest(library=library):
                self.assertEqual(fetch_weather.tag_version(library, tag), version)


class CopyDocsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        source = self.root / 'docs'
        (source / 'guide').mkdir(parents=True)
        (source / '_static').mkdir()
        (source / 'index.rst').write_text('Title\n=====\nBody text.\n')
        (source / 'guide' / 'usage.rst').write_text('Usage\n=====\nDetails.\n')
        (source / 'conf.py').write_text('project = "x"\n')
        (source / 'theme.css').write_text('body{}\n')
        (source / '_static' / 'logo.rst').write_text('should be skipped\n')
        (source / 'README').write_text('Doc overview.\n')
        self.target = self.root / 'out'

    def tearDown(self):
        self.temp.cleanup()

    def test_copies_matching_types_preserving_relative_paths(self):
        import fetch_weather

        count = fetch_weather.copy_docs(self.root / 'docs', self.target, ('.rst',))
        self.assertEqual(count, 3)  # index.rst, guide/usage.rst, README (name-qualified)
        self.assertEqual((self.target / 'guide' / 'usage.rst').read_text(), 'Usage\n=====\nDetails.\n')
        self.assertFalse((self.target / 'conf.py').exists())
        self.assertFalse((self.target / '_static' / 'logo.rst').exists())
        self.assertFalse((self.target / 'theme.css').exists())


class LatexSectionTests(unittest.TestCase):
    TEXT = r"""\documentclass{report}
\title{Model Guide}
\begin{document}
Preamble paragraph.
\chapter{Ocean}
First chapter text with a brace \{group\} and nested {brace}.
\section{Dynamics}
Dynamics text.
\subsection{Vertical}
Vertical text.
\section*{Acknowledgments}
Thanks.
\end{document}
"""

    def test_tex_headings_create_hierarchical_units(self):
        units = markup_sections(self.TEXT, 'guide.tex', 'https://example.test/guide.tex', '.tex')
        titles = [unit['title'] for unit in units]
        self.assertEqual(titles, ['Document preamble', 'Ocean', 'Dynamics', 'Vertical', 'Acknowledgments'])
        ocean, dynamics, vertical = units[1], units[2], units[3]
        self.assertEqual(dynamics['parent'], ocean['id'])
        self.assertEqual(vertical['parent'], dynamics['id'])
        self.assertEqual(units[4]['parent'], ocean['id'])
        self.assertIn('nested {brace}', ocean['text'])
        self.assertIn('Dynamics text', dynamics['text'])

    def test_tex_without_headings_falls_back_to_whole_file(self):
        units = markup_sections('\\newcommand{\\x}{1}\n', 'macros.tex', '', '.tex')
        self.assertEqual(len(units), 1)
        self.assertTrue(units[0]['title'].startswith('Full source file'))

    def test_tex_sections_are_searchable_through_build(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'mpas.tex').write_text(self.TEXT, encoding='utf-8')
            manifest = root / 'corpus.json'
            manifest.write_text(json.dumps({'sources': [{
                'name': 'mpas-docs', 'path': 'mpas.tex', 'library': 'mpas',
                'version': 'snapshot-test', 'kind': 'documentation'}]}))
            database = root / 'index.sqlite3'
            result = build(manifest, database)
            self.assertEqual(result['collections'], [('mpas', 'snapshot-test')])
            from knowledge import Knowledge

            store = Knowledge(database)
            hits = store.search('Dynamics', 'documentation', library='mpas')
            self.assertTrue(hits)
            self.assertEqual(hits[0]['title'], 'Dynamics')


class CrawlerReuseTests(unittest.TestCase):
    def test_crawl_site_accepts_an_override_registry(self):
        import fetch_pytorch

        sites = {'demo': {'version': '1', 'base_url': 'https://example.invalid/docs/1/',
                          'start_pages': ('index.html',), 'path': 'corpus/demo/source'}}
        self.assertIn('demo', sites)
        # Signature compatibility only; no network in unit tests.
        import inspect

        parameters = inspect.signature(fetch_pytorch.crawl_site).parameters
        self.assertIn('sites', parameters)

    def test_htmlzip_extraction_drops_top_directory_and_keeps_pages(self):
        import io
        import tempfile
        import zipfile as zf

        import fetch_pytorch

        buffer = io.BytesIO()
        with zf.ZipFile(buffer, 'w') as archive:
            archive.writestr('mom6-main/index.html', '<h1>MOM6</h1>')
            archive.writestr('mom6-main/api/mom.html', '<h1>API</h1>')
            archive.writestr('mom6-main/_static/theme.css', 'body{}')
            archive.writestr('mom6-main/_sources/index.rst.txt', 'raw')
            archive.writestr('mom6-main/searchindex.js', 'data')
            archive.writestr('mom6-main/', '')
        buffer.seek(0)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            stored = fetch_pytorch.extract_html_zip(buffer, output)
            self.assertEqual(stored, {'index.html', 'api/mom.html'})
            self.assertEqual((output / 'api' / 'mom.html').read_text(), '<h1>API</h1>')
            self.assertFalse((output / '_static').exists())

    def test_htmlzip_extraction_rejects_unsafe_paths(self):
        import io
        import zipfile as zf

        import fetch_pytorch

        buffer = io.BytesIO()
        with zf.ZipFile(buffer, 'w') as archive:
            archive.writestr('mom6-main/../../escape.html', 'bad')
        buffer.seek(0)
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                fetch_pytorch.extract_html_zip(buffer, Path(directory))


class IdempotentSkipTests(unittest.TestCase):
    """A refresh must reuse an existing acquisition instead of re-fetching."""

    def setUp(self):
        import fetch_weather

        self.module = fetch_weather
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.patch = unittest.mock.patch.object(fetch_weather, 'ROOT', self.root)
        self.patch.start()
        self.entry = [{'name': 'mom6-docs', 'library': 'mom6', 'managed': True}]
        self.lock = {'mom6': {'version': 'snapshot-main'}}

    def tearDown(self):
        self.patch.stop()
        self.temp.cleanup()

    def _make_corpus(self):
        source = self.root / 'corpus' / 'mom6' / 'source'
        source.mkdir(parents=True)
        (source / 'index.html').write_text('<h1>MOM6</h1>')

    def test_requires_lock_entry(self):
        self._make_corpus()
        self.assertFalse(self.module.already_acquired('mom6', self.entry, {}))

    def test_requires_catalog_entry(self):
        self._make_corpus()
        self.assertFalse(self.module.already_acquired('mom6', [], self.lock))

    def test_requires_nonempty_corpus(self):
        (self.root / 'corpus' / 'mom6' / 'source').mkdir(parents=True)
        self.assertFalse(self.module.already_acquired('mom6', self.entry, self.lock))

    def test_true_when_all_present(self):
        self._make_corpus()
        self.assertTrue(self.module.already_acquired('mom6', self.entry, self.lock))

    def test_crawl_site_accepts_a_pacing_delay(self):
        import inspect

        import fetch_pytorch

        self.assertIn('delay', inspect.signature(fetch_pytorch.crawl_site).parameters)


if __name__ == '__main__':
    unittest.main()
