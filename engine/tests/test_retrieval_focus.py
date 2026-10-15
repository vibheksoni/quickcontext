import unittest
from dataclasses import dataclass
from typing import Optional
from unittest import mock

import numpy as np

from engine.sdk import QuickContext
from engine.src.config import EngineConfig
from engine.src.embedder import embed_length_sorted
from engine.src.parsing import TextSearchMatch
from engine.src.searcher import _path_tail


@dataclass(frozen=True)
class _Sym:
    name: str
    kind: str
    source: str
    file_path: str = "C:/repo/src/process_cleanup.py"
    line_start: int = 0
    line_end: int = 5
    byte_start: int = 0
    byte_end: int = 10
    signature: Optional[str] = None
    docstring: Optional[str] = None
    parent: Optional[str] = None
    language: str = "python"


SYMBOLS = [
    _Sym("ProcessCleanup", "class", "class ProcessCleanup:\n    profile directory cleanup delete remove tracked browser", byte_start=0, byte_end=500),
    _Sym("_cleanup_profile_dir", "method", "def _cleanup_profile_dir(self, profile_dir):\n    shutil.rmtree(profile_dir)  # delete temp profile directory", docstring="Remove a browser temp profile directory when it is safe.", parent="ProcessCleanup", byte_start=10, byte_end=100),
    _Sym("track_browser_process", "method", "def track_browser_process(self, pid):\n    self.processes[pid] = time.time()", parent="ProcessCleanup", byte_start=100, byte_end=200),
    _Sym("os", "import", "import os", byte_start=200, byte_end=210),
]


class _OrderModel:
    def embed(self, texts, batch_size):
        return [np.array([float(len(text))]) for text in texts]


class RetrievalFocusTests(unittest.TestCase):
    def setUp(self) -> None:
        self.qc = QuickContext(EngineConfig(store=None, code_embedding=None, desc_embedding=None, llm=None))

    def tearDown(self) -> None:
        self.qc.close()

    def test_best_symbol_prefers_named_method_over_container(self) -> None:
        terms = {term[:6] for term in self.qc._split_symbol_text_tokens("Where are temporary browser profile directories deleted?")}
        best = self.qc._best_symbol_for_terms(terms, SYMBOLS)
        self.assertEqual(best.name, "_cleanup_profile_dir")

    def test_best_symbol_returns_none_without_overlap(self) -> None:
        terms = {term[:6] for term in self.qc._split_symbol_text_tokens("quantum chromodynamics lattice")}
        self.assertIsNone(self.qc._best_symbol_for_terms(terms, SYMBOLS))

    def test_text_primary_file_matches_are_focused_on_symbols(self) -> None:
        match = TextSearchMatch(
            file_path="C:/repo/src/process_cleanup.py",
            score=7.5,
            matched_terms=["profile", "directory", "delete"],
            snippet="import os",
            snippet_line_start=1,
            snippet_line_end=1,
            language="python",
        )
        rows = self.qc._text_matches_to_search_results([match])
        with mock.patch.object(self.qc, "_load_file_symbols", return_value=SYMBOLS):
            focused = self.qc._focus_file_matches_on_symbols("how is the temp profile directory deleted", rows)
        self.assertEqual(focused[0].symbol_name, "_cleanup_profile_dir")
        self.assertEqual(focused[0].symbol_kind, "method")
        self.assertEqual(focused[0].score, 7.5)
        self.assertIn("rmtree", focused[0].source)

    def test_embed_length_sorted_preserves_input_order(self) -> None:
        texts = ["ccc", "a", "bbbb", "bb"]
        vectors = embed_length_sorted(_OrderModel(), texts, batch_size=2)
        self.assertEqual([int(vector[0]) for vector in vectors], [3, 1, 4, 2])

    def test_path_tail_drops_absolute_prefix(self) -> None:
        self.assertEqual(
            _path_tail("\\\\?\\C:\\Users\\me\\Desktop\\repo\\benchmarks\\targets\\input_fidelity.py"),
            "benchmarks/targets/input_fidelity.py",
        )


if __name__ == "__main__":
    unittest.main()
