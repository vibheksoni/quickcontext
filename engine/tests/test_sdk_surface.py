import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from engine.sdk import QuickContext
from engine.src.config import EngineConfig, QdrantConfig
from engine.src.indexer import IndexStats
from engine.src.operation_status import GLOBAL_OPERATION_REGISTRY
from engine.src.parsing import TextSearchMatch
from engine.src.project import detect_project_name
from engine.src.sdk_models import ProjectCollectionInfo


class SDKSurfaceTests(unittest.TestCase):
    def test_qdrant_available_probes_connection_when_requested(self) -> None:
        qc = QuickContext(EngineConfig(qdrant=QdrantConfig()))

        with mock.patch.object(qc, "connect", return_value=qc) as connect:
            self.assertTrue(qc.qdrant_available(verify=True))

        connect.assert_called_once_with(verify=True)

    def test_status_reports_qdrant_health_without_prior_cached_client(self) -> None:
        qc = QuickContext(EngineConfig(qdrant=QdrantConfig()))
        fake_pipe = mock.Mock(_pipe_name="test-pipe")
        qc._parser_service = mock.Mock(connected=True, _client=fake_pipe)

        with mock.patch.object(qc, "connect", return_value=qc) as connect:
            status = qc.status()

        self.assertTrue(status["qdrant"]["configured"])
        self.assertTrue(status["qdrant"]["alive"])
        self.assertEqual(status["parser"]["pipe_name"], "test-pipe")
        connect.assert_called_once_with(verify=True)

    def test_project_info_returns_stable_discovery_payload(self) -> None:
        qc = QuickContext(EngineConfig(qdrant=QdrantConfig()))
        qc._parser_service = mock.Mock(connected=False)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "package.json").write_text("{}", encoding="utf-8")
            (root / "src").mkdir()

            with mock.patch.object(qc, "qdrant_available", return_value=False):
                info = qc.project_info(root, include_folders=True)

        self.assertEqual(info.project_name, root.name)
        self.assertFalse(info.qdrant_available)
        self.assertFalse(info.collection.indexed)
        self.assertEqual(info.folders[0].relative_path, ".")
        self.assertIn("src", {item.relative_path for item in info.folders})

    def test_detect_project_name_ignores_home_level_markers_for_external_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            (home / ".project").write_text("", encoding="utf-8")
            target = home / "Downloads" / "windsurf.com"
            target.mkdir(parents=True)

            with mock.patch("engine.src.project.Path.home", return_value=home):
                project_name = detect_project_name(target)

        self.assertEqual(project_name, "windsurf_com")

    def test_list_projects_returns_typed_collection_info(self) -> None:
        qc = QuickContext(EngineConfig(qdrant=QdrantConfig()))
        qc._conn = mock.Mock()
        qc._conn.client = object()

        fake_projects = [
            {
                "name": "demo",
                "real_collection": "demo_v2",
                "points_count": 42,
                "indexed_vectors_count": 84,
                "segments_count": 3,
                "status": "green",
                "vectors": {"code": {"size": 768, "distance": "cosine"}},
            }
        ]

        with mock.patch.object(qc, "qdrant_available", return_value=True):
            with mock.patch("engine.src.collection.CollectionManager.list_all_projects", return_value=fake_projects):
                projects = qc.list_projects()

        self.assertEqual(len(projects), 1)
        self.assertIsInstance(projects[0], ProjectCollectionInfo)
        self.assertEqual(projects[0].project_name, "demo")
        self.assertTrue(projects[0].indexed)
        self.assertEqual(projects[0].vectors["code"]["size"], 768)

    def test_text_search_match_normalizes_windows_verbatim_prefix(self) -> None:
        match = TextSearchMatch.from_dict(
            {
                "file_path": "\\\\?\\C:\\repo\\src\\main.py",
                "score": 1.0,
                "matched_terms": ["main"],
                "snippet": "def main(): pass",
                "snippet_line_start": 1,
                "snippet_line_end": 1,
                "language": "python",
            }
        )
        self.assertEqual(match.file_path, "C:\\repo\\src\\main.py")

    def test_start_index_directory_exposes_pollable_progress_snapshot(self) -> None:
        qc = QuickContext(EngineConfig(qdrant=QdrantConfig()))

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "package.json").write_text("{}", encoding="utf-8")

            def _fake_index(*args, **kwargs):
                reporter = kwargs.get("_progress_reporter")
                if reporter is not None:
                    reporter.start("extract", "Extracting symbols")
                    reporter.update(files_discovered=10, files_planned=4, files_remaining=4)
                    reporter.set_stage("upsert", "Upserting chunks")
                    reporter.update(
                        files_indexed=4,
                        files_remaining=0,
                        chunks_built=12,
                        chunks_kept=8,
                        description_chunks_completed=8,
                        embedding_chunks_completed=8,
                        points_upserted=8,
                    )
                return IndexStats(
                    total_chunks=8,
                    upserted_points=8,
                    failed_points=0,
                    total_tokens=0,
                    duration_seconds=0.01,
                )

            with mock.patch.object(QuickContext, "index_directory", side_effect=_fake_index):
                snapshot = qc.start_index_directory(root, fast=True)
                final = None
                for _ in range(50):
                    final = qc.get_operation_status(snapshot.operation_id)
                    if final is not None and final.status == "completed":
                        break
                    time.sleep(0.02)

            self.assertIsNotNone(final)
            assert final is not None
            self.assertEqual(final.status, "completed")
            self.assertEqual(final.current_stage, "completed")
            self.assertEqual(final.files_discovered, 10)
            self.assertEqual(final.files_indexed, 4)
            self.assertEqual(final.chunks_kept, 8)
            self.assertEqual(final.points_upserted, 8)
