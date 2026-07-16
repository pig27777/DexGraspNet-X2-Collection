"""Regression audit for the X2 Newton mimic provenance contract."""

from __future__ import annotations

import math
import unittest

from pxr import Usd, UsdPhysics

from grasp_generation.utils.x2_config import load_x2_mesh_config


def _raw_api_schemas(prim: Usd.Prim) -> set[str]:
    names = {str(name) for name in prim.GetAppliedSchemas()}
    raw = prim.GetMetadata("apiSchemas")
    if raw is not None:
        names.update(str(name) for name in raw.GetAppliedItems())
    return names


def _effective_attribute(prim: Usd.Prim, name: str, default):
    attribute = prim.GetAttribute(name)
    return attribute.Get() if attribute and attribute.HasAuthoredValue() else default


class NewtonMimicProvenanceTest(unittest.TestCase):
    def test_asset_and_config_define_one_read_only_newton_follower_graph(self) -> None:
        config = load_x2_mesh_config()
        stage = Usd.Stage.Open(
            str(config.configured_path("robot.usd_path", must_exist=True))
        )
        self.assertTrue(stage)

        source_root = str(config.require("robot.source_prim_path")).rstrip("/")
        joint_prims = {
            str(prim.GetName()): prim
            for prim in stage.Traverse()
            if prim.GetPath().pathString.startswith(source_root + "/")
            and prim.IsA(UsdPhysics.RevoluteJoint)
        }
        self.assertEqual(
            set(joint_prims), set(config.require("robot.full_joint_names"))
        )

        schemas_by_joint = {
            name: _raw_api_schemas(prim) for name, prim in joint_prims.items()
        }
        for schemas in schemas_by_joint.values():
            self.assertFalse(
                any(
                    schema == "PhysxMimicJointAPI"
                    or schema.startswith("PhysxMimicJointAPI:")
                    for schema in schemas
                )
            )

        coupling = config.require("robot.passive_joint_coupling")
        actual_followers = {
            name
            for name, schemas in schemas_by_joint.items()
            if "NewtonMimicAPI" in schemas
        }
        self.assertEqual(actual_followers, set(coupling))

        joint_path_by_name = {
            name: prim.GetPath().pathString for name, prim in joint_prims.items()
        }
        for follower_name, follower_config in coupling.items():
            with self.subTest(follower=follower_name):
                prim = joint_prims[follower_name]
                relationship = prim.GetRelationship("newton:mimicJoint")
                self.assertTrue(relationship)
                self.assertTrue(relationship.HasAuthoredTargets())
                self.assertEqual(
                    [str(target) for target in relationship.GetTargets()],
                    [joint_path_by_name[str(follower_config["driver"])]],
                )

                enabled = _effective_attribute(prim, "newton:mimicEnabled", True)
                coef0 = float(
                    _effective_attribute(prim, "newton:mimicCoef0", 0.0)
                )
                coef1 = float(
                    _effective_attribute(prim, "newton:mimicCoef1", 1.0)
                )
                self.assertIs(enabled, True)
                self.assertTrue(math.isfinite(coef0))
                self.assertTrue(math.isfinite(coef1))
                self.assertAlmostEqual(
                    coef0, float(follower_config["offset"]), places=9
                )
                self.assertAlmostEqual(
                    coef1, float(follower_config["multiplier"]), places=9
                )


if __name__ == "__main__":
    unittest.main()
