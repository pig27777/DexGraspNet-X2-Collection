"""Tests for deterministic audited general-mesh selection."""

from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path

import trimesh

from scripts.select_x2_general_meshes import (
    DEFAULT_TARGET_COUNT,
    MeshSelectionError,
    audit_mesh,
    run,
)


class X2GeneralMeshSelectionTest(unittest.TestCase):
    @staticmethod
    def _box(
        root: Path,
        object_id: str,
        extents: tuple[float, float, float] = (0.12, 0.08, 0.06),
    ) -> Path:
        path = root / object_id / "coacd" / "decomposed.obj"
        path.parent.mkdir(parents=True, exist_ok=True)
        trimesh.creation.box(extents=extents).export(path)
        return path

    def test_default_target_is_official_graspnet_count(self) -> None:
        self.assertEqual(DEFAULT_TARGET_COUNT, 88)

    def test_scale_one_audits_official_metric_dimensions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            accepted = self._box(
                root, "sem-Official-metric", extents=(0.30, 0.04, 0.003)
            )
            audited = audit_mesh(accepted, object_scale=1.0)
            self.assertEqual(audited.object_scale, 1.0)
            self.assertAlmostEqual(max(audited.scaled_extents), 0.30)
            self.assertGreater(audited.scaled_volume, 1.0e-8)

            too_thin = self._box(
                root, "sem-Official-too-thin", extents=(0.10, 0.04, 0.0029)
            )
            too_large = self._box(
                root, "sem-Official-too-large", extents=(0.351, 0.04, 0.03)
            )
            with self.assertRaises(MeshSelectionError):
                audit_mesh(too_thin, object_scale=1.0)
            with self.assertRaises(MeshSelectionError):
                audit_mesh(too_large, object_scale=1.0)

    def test_selection_preserves_existing_and_is_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            destination = root / "destination"
            existing = self._box(destination, "sem-Camera-existing")
            for object_id in (
                "sem-Bottle-a",
                "sem-Mug-b",
                "core-03001627-c",
                "ddg-item-d",
            ):
                self._box(source, object_id)
            self.assertGreater(audit_mesh(existing).scaled_volume, 1.0e-8)
            args = argparse.Namespace(
                source_root=source,
                destination=destination,
                target_count=4,
                object_scale=1.0,
                replace_existing=False,
                manifest=None,
            )
            first = run(args)
            self.assertEqual(first["selected_count"], 4)
            self.assertEqual(first["preexisting_count"], 1)
            self.assertEqual(len(list(destination.glob("*/coacd/decomposed.obj"))), 4)
            second = run(args)
            self.assertEqual(second["selected_count"], 4)
            self.assertEqual(second["preexisting_count"], 4)
            self.assertEqual(
                [record["object_id"] for record in first["meshes"]],
                [record["object_id"] for record in second["meshes"]],
            )

    def test_replace_archives_old_objects_and_finishes_at_exact_target(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            destination = root / "destination"
            manifest = root / "manifest.json"

            # One selected ID conflicts with old content; the other old ID is no
            # longer selected. Both originals must survive under the archive.
            self._box(
                destination,
                "sem-Bottle-a",
                extents=(0.10, 0.07, 0.05),
            )
            self._box(destination, "sem-Old-z")
            source_selected_a = self._box(
                source,
                "sem-Bottle-a",
                extents=(0.14, 0.08, 0.06),
            )
            self._box(source, "sem-Bottle-b")
            self._box(source, "sem-Bottle-c")

            report = run(
                argparse.Namespace(
                    source_root=source,
                    destination=destination,
                    target_count=2,
                    object_scale=1.0,
                    replace_existing=True,
                    manifest=manifest,
                )
            )

            direct_paths = sorted(destination.glob("*/coacd/decomposed.obj"))
            self.assertEqual(len(direct_paths), 2)
            self.assertEqual(
                {path.parent.parent.name for path in direct_paths},
                {"sem-Bottle-a", "sem-Bottle-b"},
            )
            archive = destination / "_excluded_general_meshes"
            self.assertTrue((archive / "sem-Bottle-a" / "coacd" / "decomposed.obj").is_file())
            self.assertTrue((archive / "sem-Old-z" / "coacd" / "decomposed.obj").is_file())
            self.assertEqual(
                (destination / "sem-Bottle-a" / "coacd" / "decomposed.obj").read_bytes(),
                source_selected_a.read_bytes(),
            )
            self.assertEqual(report["selected_count"], 2)
            self.assertEqual(report["archived_ids"], ["sem-Bottle-a", "sem-Old-z"])
            self.assertEqual(report["archived_conflict_ids"], ["sem-Bottle-a"])
            self.assertEqual(report["archived_unselected_ids"], ["sem-Old-z"])
            self.assertTrue(all(mesh["object_scale"] == 1.0 for mesh in report["meshes"]))

            saved = json.loads(manifest.read_text(encoding="utf-8"))
            self.assertEqual(saved["archived_ids"], report["archived_ids"])
            self.assertEqual(saved["object_scale"], 1.0)
            self.assertEqual(saved["selected_count"], 2)


if __name__ == "__main__":
    unittest.main()
