"""Guard that the places a collection is registered stay in agreement.

Adding a library touches knowledge.LIBRARIES, the catalog, a lock file, the search
tool's description and the README, and nothing else notices when one is missed.
These checks read only committed files, so they need no network or fetched corpus.
"""
import json
import unittest
from pathlib import Path

import server
from build_manifest import FETCHED_LIBRARIES
from knowledge import LIBRARIES

ROOT = Path(__file__).resolve().parents[1]
LOCK_FILES = ('weather-lock.json', 'standards-lock.json', 'kokkos-lock.json')
# In LIBRARIES and the catalog, but pinned by no lock file: ESMF is the hard-coded
# VERSION, and the PyTorch sites carry their pinned release in fetch_pytorch.SITES.
UNLOCKED = {'esmf', 'pytorch', 'pytorch-forecasting'}
REQUIRED_KEYS = ('name', 'path', 'version', 'kind')


def library_of(entry):
    return entry.get('library', 'esmf')


class RegistryConsistencyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.catalog = json.loads((ROOT / 'corpus.catalog.json').read_text())['sources']
        cls.versions = {}
        for entry in cls.catalog:
            cls.versions.setdefault(library_of(entry), set()).add(entry['version'])
        cls.locks = {}
        for name in LOCK_FILES:
            for library, pin in json.loads((ROOT / name).read_text()).items():
                cls.locks[library] = (name, pin)

    def test_catalog_entries_have_the_required_keys(self):
        for entry in self.catalog:
            with self.subTest(entry=entry.get('name')):
                for key in REQUIRED_KEYS:
                    self.assertIn(key, entry)

    def test_catalog_and_registry_name_the_same_libraries(self):
        self.assertEqual(set(self.versions), set(LIBRARIES))

    def test_fetched_libraries_are_registered(self):
        self.assertLessEqual(FETCHED_LIBRARIES, set(LIBRARIES))

    def test_committed_catalog_paths_exist(self):
        # Fetched libraries are acquired at build time, so their path may be absent.
        for entry in self.catalog:
            if library_of(entry) in FETCHED_LIBRARIES:
                continue
            with self.subTest(entry=entry['name']):
                self.assertTrue((ROOT / entry['path']).exists(), entry['path'])

    def test_lock_entries_name_registered_libraries(self):
        for library, (lock_file, _) in self.locks.items():
            with self.subTest(library=library):
                self.assertIn(library, LIBRARIES, f'{lock_file} pins an unregistered library')

    def test_every_pinned_library_has_a_lock_entry_matching_the_catalog_version(self):
        for library in sorted(set(LIBRARIES) - UNLOCKED):
            with self.subTest(library=library):
                self.assertIn(library, self.locks, 'no lock file pins this library')
                lock_file, pin = self.locks[library]
                self.assertIn(pin['version'], self.versions[library],
                              f'{lock_file} disagrees with corpus.catalog.json')

    def test_search_tool_description_lists_every_library(self):
        for library in LIBRARIES:
            with self.subTest(library=library):
                self.assertIn(library, server.search_docs.__doc__)

    def test_readme_lists_every_library(self):
        readme = (ROOT / 'README.md').read_text()
        for library in LIBRARIES:
            with self.subTest(library=library):
                self.assertIn(f'`{library}`', readme)


if __name__ == '__main__':
    unittest.main()
