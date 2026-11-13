"""Keep local data-derived outputs out of the distributable notebook."""

import json
import unittest
from pathlib import Path


class NotebookHygieneTests(unittest.TestCase):
    def test_source_notebook_has_no_outputs_or_machine_paths(self):
        root = Path(__file__).resolve().parents[1]
        notebook = json.loads((root / "runner.ipynb").read_text())
        for index, cell in enumerate(notebook["cells"]):
            with self.subTest(cell=index):
                self.assertFalse(cell.get("outputs"))
                self.assertIsNone(cell.get("execution_count"))
                self.assertFalse(cell.get("attachments"))
                source = "".join(cell["source"])
                self.assertNotIn("/Users/", source)
                self.assertNotIn("/home/", source)
                self.assertNotIn("import marimo", source)
        self.assertNotIn("marimo", notebook["metadata"])


if __name__ == "__main__":
    unittest.main()
