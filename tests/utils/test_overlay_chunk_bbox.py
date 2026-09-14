"""Regression tests for the bounding-box overlay's chunk/import-bbox overlap
check in cellmap_flow.dashboard.routes.finetune.overlay."""

import unittest

import numpy as np

from cellmap_flow.dashboard.routes.finetune.overlay import _chunk_outside_all_bboxes


class ChunkOutsideAllBboxesTests(unittest.TestCase):
    def test_no_bboxes_is_outside(self):
        self.assertTrue(
            _chunk_outside_all_bboxes(
                np.array([0, 0, 0]),
                np.array([128, 128, 128]),
                np.zeros((0, 3), dtype=np.int64),
                np.zeros((0, 3), dtype=np.int64),
            )
        )

    def test_fully_contained_chunk_is_covered(self):
        offsets = np.array([[0, 0, 0]])
        ends = np.array([[256, 256, 256]])
        self.assertFalse(
            _chunk_outside_all_bboxes(
                np.array([128, 128, 128]), np.array([256, 256, 256]), offsets, ends
            )
        )

    def test_boundary_straddling_chunk_is_covered_not_painted(self):
        """A crop's global offset is essentially never chunk-aligned, so
        chunks straddling the import bbox's edge only partially overlap it.
        These must still count as covered, not as separate painted-only
        chunks (which used to fence the real import box with duplicate
        boxes -- see the jrc_axolotl-heart-1 mito005 crop bug report)."""
        offsets = np.array([[14283, 5655, 3352]])
        ends = offsets + np.array([[440, 857, 1000]])
        chunk_lo = np.array([14208, 5632, 3328])  # 128-aligned chunk grid
        chunk_hi = chunk_lo + 128
        self.assertFalse(
            _chunk_outside_all_bboxes(chunk_lo, chunk_hi, offsets, ends)
        )

    def test_disjoint_chunk_is_painted_only(self):
        offsets = np.array([[0, 0, 0]])
        ends = np.array([[128, 128, 128]])
        self.assertTrue(
            _chunk_outside_all_bboxes(
                np.array([256, 256, 256]), np.array([384, 384, 384]), offsets, ends
            )
        )


if __name__ == "__main__":
    unittest.main()
