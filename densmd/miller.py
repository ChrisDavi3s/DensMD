"""Miller-plane geometry: normals, voxel masks, point filtering.

Pure numpy, no Qt or VTK, so it is trivially testable.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class MillerParams:
    """Immutable snapshot of the Miller controls."""

    use: bool
    h: int
    k: int
    l: int
    thickness: float
    offset: float

    @property
    def normal(self) -> Optional[np.ndarray]:
        """Unit normal of the (hkl) plane, or None if indices are all zero."""
        if not self.use:
            return None
        v = np.array([self.h, self.k, self.l], dtype=float)
        n = np.linalg.norm(v)
        return v / n if n > 0 else None

    def key(self) -> Tuple:
        """Hashable identity for caching."""
        return (self.use, self.h, self.k, self.l, self.thickness, self.offset)


def voxel_centers(roi: Dict[str, int], origin: np.ndarray,
                  spacing: np.ndarray) -> np.ndarray:
    """Physical coordinates of ROI voxel centres, shape (nx, ny, nz, 3)."""
    roi_origin = origin + np.array([roi["xmin"], roi["ymin"], roi["zmin"]]) * spacing
    dims = np.array([
        roi["xmax"] - roi["xmin"] + 1,
        roi["ymax"] - roi["ymin"] + 1,
        roi["zmax"] - roi["zmin"] + 1,
    ])
    axes = [roi_origin[i] + (np.arange(dims[i]) + 0.5) * spacing[i] for i in range(3)]
    grid = np.meshgrid(*axes, indexing="ij")
    return np.stack(grid, axis=-1)


def voxel_mask(centers: np.ndarray, cell_center: np.ndarray,
               params: MillerParams) -> Optional[np.ndarray]:
    """Boolean mask of voxels within the slab, or None when Miller is off."""
    n = params.normal
    if n is None:
        return None
    dist = np.abs(np.sum((centers - cell_center) * n, axis=-1) - params.offset)
    return dist < (params.thickness / 2.0)


def filter_points(points: np.ndarray, cell_center: np.ndarray,
                  params: MillerParams) -> np.ndarray:
    """Keep only points lying within the Miller slab."""
    n = params.normal
    if n is None:
        return points
    dist = np.abs(np.dot(points - cell_center, n) - params.offset)
    return points[dist < (params.thickness / 2.0)]
