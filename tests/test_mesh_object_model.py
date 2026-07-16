"""Focused regressions for differentiable mesh penetration queries."""

from __future__ import annotations

import unittest
from pathlib import Path
from unittest import mock

import torch

from grasp_generation.utils.mesh_object_model import (
    MeshObjectModel,
    _triangle_chunk_closest,
)


class MeshObjectPenetrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        root = Path(__file__).resolve().parents[1]
        cls.model = MeshObjectModel(
            root / "data/meshdata/x2_primitives/sphere/sphere_r020.obj",
            batch_size=2,
            scale=1.0,
            num_surface_samples=8,
            device="cpu",
            dtype=torch.float64,
            seed=7,
        )

    @staticmethod
    def _points() -> torch.Tensor:
        return torch.tensor(
            [
                [
                    [0.005, 0.002, 0.001],
                    [-0.007, 0.003, 0.002],
                    [0.018, 0.018, 0.001],
                    [0.030, 0.004, 0.001],
                ],
                [
                    [0.004, -0.006, 0.003],
                    [-0.009, -0.002, 0.004],
                    [-0.018, 0.018, -0.001],
                    [-0.031, -0.003, 0.002],
                ],
            ],
            dtype=torch.float64,
        )

    def test_closed_aabb_filter_matches_full_query_values_and_gradients(self) -> None:
        full_points = self._points().requires_grad_()
        full = torch.relu(self.model.query(full_points).signed_distance)
        full.sum().backward()

        filtered_points = self._points().requires_grad_()
        filtered = self.model.penetration_depth(filtered_points)
        filtered.sum().backward()

        torch.testing.assert_close(filtered, full, rtol=0.0, atol=0.0)
        torch.testing.assert_close(
            filtered_points.grad, full_points.grad, rtol=0.0, atol=0.0
        )
        self.assertEqual(tuple(filtered.shape), (2, 4))

    def test_empty_aabb_candidate_set_returns_differentiable_zeros(self) -> None:
        upper = self.model.bounds_upper.detach()
        points = torch.stack((upper + 1.0, upper + 2.0)).requires_grad_()
        with mock.patch.object(
            self.model,
            "query",
            side_effect=AssertionError("empty candidate set must skip full query"),
        ):
            penetration = self.model.penetration_depth(points)
        torch.testing.assert_close(
            penetration, torch.zeros(2, dtype=torch.float64)
        )
        penetration.sum().backward()
        torch.testing.assert_close(points.grad, torch.zeros_like(points))

    def test_penetration_summary_matches_exact_dense_reductions(self) -> None:
        points = self._points()
        expected = torch.relu(self.model.query(points).signed_distance)
        total, maximum = self.model.penetration_summary(points)
        torch.testing.assert_close(total, expected.sum(dim=-1))
        torch.testing.assert_close(maximum, expected.amax(dim=-1))

    def test_spatial_query_matches_all_triangle_brute_force_and_gradients(self) -> None:
        generator = torch.Generator().manual_seed(20260716)
        base = torch.randn(24, 3, generator=generator, dtype=torch.float64)
        accelerated_points = (0.028 * base).requires_grad_()
        accelerated = self.model.query(accelerated_points)
        accelerated.signed_distance.sum().backward()

        brute_points = (0.028 * base).requires_grad_()
        closest, squared, face = _triangle_chunk_closest(
            brute_points, self.model.triangles
        )
        normals = self.model.face_normals[face]
        distance = torch.sqrt(squared.clamp_min(1.0e-18))
        orientation = torch.sum((brute_points - closest) * normals, dim=-1)
        brute_signed = torch.where(orientation <= 0.0, distance, -distance)
        brute_signed.sum().backward()

        torch.testing.assert_close(
            accelerated.signed_distance, brute_signed, rtol=1.0e-12, atol=1.0e-12
        )
        torch.testing.assert_close(
            accelerated.closest_points, closest, rtol=1.0e-12, atol=1.0e-12
        )
        torch.testing.assert_close(
            accelerated.outward_normals, normals, rtol=0.0, atol=0.0
        )
        torch.testing.assert_close(
            accelerated_points.grad,
            brute_points.grad,
            rtol=1.0e-11,
            atol=1.0e-12,
        )
        self.assertEqual(tuple(self.model.audit_surface_points.shape), (8192, 3))


if __name__ == "__main__":
    unittest.main()
