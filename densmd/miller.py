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


def voxel_axes(roi: Dict[str, int], origin: np.ndarray,
               spacing: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Physical coordinates of ROI voxel centres per axis."""
    roi_origin = origin + np.array([roi["xmin"], roi["ymin"], roi["zmin"]]) * spacing
    dims = [
        roi["xmax"] - roi["xmin"] + 1,
        roi["ymax"] - roi["ymin"] + 1,
        roi["zmax"] - roi["zmin"] + 1,
    ]
    return tuple(roi_origin[i] + (np.arange(dims[i]) + 0.5) * spacing[i] for i in range(3))


def voxel_mask(axes: Tuple[np.ndarray, np.ndarray, np.ndarray], cell_center: np.ndarray,
               params: MillerParams) -> Optional[np.ndarray]:
    """Boolean mask of voxels within the slab, or None when Miller is off."""
    n = params.normal
    if n is None:
        return None
    dx = (axes[0] - cell_center[0]) * n[0]
    dy = (axes[1] - cell_center[1]) * n[1]
    dz = (axes[2] - cell_center[2]) * n[2]
    dist = np.abs(dx[:, None, None] + dy[None, :, None] + dz[None, None, :] - params.offset)
    return dist < (params.thickness / 2.0)


def filter_points(points: np.ndarray, cell_center: np.ndarray,
                  params: MillerParams) -> np.ndarray:
    """Keep only points lying within the Miller slab."""
    n = params.normal
    if n is None:
        return points
    dist = np.abs(np.dot(points - cell_center, n) - params.offset)
    return points[dist < (params.thickness / 2.0)]
