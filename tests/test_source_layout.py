import importlib
import tomllib
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class SourceLayoutTests(unittest.TestCase):
    def test_project_modules_are_installed_from_src(self):
        config = tomllib.loads((ROOT / 'pyproject.toml').read_text())
        setuptools = config['tool']['setuptools']

        self.assertEqual(setuptools.get('package-dir'), {'': 'src'})
        for module in setuptools['py-modules']:
            with self.subTest(module=module):
                self.assertTrue((ROOT / 'src' / f'{module}.py').is_file())
                self.assertFalse((ROOT / f'{module}.py').exists())
                imported = importlib.import_module(module)
                if hasattr(imported, 'ROOT'):
                    self.assertEqual(imported.ROOT, ROOT)


if __name__ == '__main__':
    unittest.main()