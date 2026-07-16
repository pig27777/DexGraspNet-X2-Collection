from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path

from scripts.diagnose_x2_self_collision import build_report


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PRIMARY_REGRESSION = (
    PROJECT_ROOT
    / "data/x2_formal_grasps_6000/sphere_r020_seed1/front_single/raw"
    / "sphere_r020_front_000031.json"
)
PRIMARY_SHA256 = "d4b6bd97826490e7356b3c536997c8390fe3ee9ca12f7eb09eb20e38a160b489"


class X2SelfCollisionDiagnosticCliTests(unittest.TestCase):
    def test_primary_regression_report_is_read_only_and_pair_complete(self) -> None:
        source_before = PRIMARY_REGRESSION.read_bytes()
        report = build_report(PRIMARY_REGRESSION)

        self.assertEqual(PRIMARY_REGRESSION.read_bytes(), source_before)
        self.assertEqual(
            report["source_sha256"], hashlib.sha256(source_before).hexdigest()
        )
        self.assertEqual(report["source_sha256"], PRIMARY_SHA256)
        self.assertFalse(report["physx_self_collision_enabled"])
        self.assertEqual(report["sampling"]["surface_samples_per_link_per_set"], 64)
        self.assertEqual(len(report["pairs"]), 119)
        self.assertEqual(len(report["links"]), 17)

        summary = report["summary"]
        self.assertEqual(summary["worst_pair"], ["rh_ffmiddle", "rh_thdistal"])
        self.assertAlmostEqual(summary["maximum_penetration_m"], 0.0080814267, places=9)
        self.assertFalse(summary["feasible"])
        self.assertEqual(summary["threshold_m"], 0.0005)

        pair = next(
            value
            for value in report["pairs"]
            if value["links"] == ["rh_ffmiddle", "rh_thdistal"]
        )
        self.assertGreater(pair["capsule_signed_depth_m"], 0.0)
        self.assertGreater(pair["hull_total_penetration_m"], 0.0)
        self.assertGreater(pair["capsule_energy_contribution"], 0.0)
        self.assertGreater(pair["hull_pair_energy"], 0.0)
        self.assertGreater(pair["hull_weighted_energy_contribution"], 0.0)
        self.assertGreater(summary["capsule_weighted_energy"], 0.0)
        self.assertGreater(summary["hull_weighted_energy"], 0.0)
        distal = report["links"]["rh_thdistal"]
        self.assertEqual(distal["collision_vertex_count"], 64)
        self.assertGreater(distal["visual_vertex_count"], 64)
        self.assertGreater(distal["maximum_support_plane_shrink_m"], 0.0)
        json.dumps(report, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
