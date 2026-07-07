"""Heuristic trajectory unwrapping for averaged-atom positions.

Averaging *wrapped* coordinates of a mobile ion is meaningless: an ion that
hops across a periodic boundary averages to somewhere in the middle of the box.
We unwrap in fractional space using the minimum-image convention between
consecutive frames, which also makes the routine correct for NPT trajectories
(each frame carries its own cell).

Everything here is pure numpy and independent of Qt/VTK.
"""
from __future__ import annotations

import numpy as np


def to_fractional(positions: np.ndarray, cells: np.ndarray) -> np.ndarray:
    """Cartesian -> fractional per frame. Shapes (F, N, 3) and (F, 3, 3)."""
    inv = np.linalg.inv(cells)                       # (F, 3, 3)
    return np.einsum("fnj,fjk->fnk", positions, inv)


def unwrap_fractional(fracs: np.ndarray) -> np.ndarray:
    """Minimum-image unwrap of fractional coords along the frame axis."""
    if fracs.shape[0] < 2:
        return fracs
    delta = np.diff(fracs, axis=0)
    delta -= np.round(delta)                         # shortest step each frame
    return np.concatenate(
        [fracs[:1], fracs[:1] + np.cumsum(delta, axis=0)], axis=0)


def averaged_positions(positions: np.ndarray, cells: np.ndarray,
                       stride: int = 1, unwrap: bool = True) -> np.ndarray:
    """Per-atom average position in Cartesian coordinates.

    Args:
        positions: (F, N, 3) trajectory for one species.
        cells: (F, 3, 3) per-frame cells (constant for NVT, varying for NPT).
        stride: subsample every ``stride`` frames for speed.
        unwrap: apply minimum-image unwrapping before averaging.

    Returns:
        (N, 3) averaged positions, wrapped back into the mean cell.
    """
    stride = max(int(stride), 1)
    pos = positions[::stride]
    cel = cells[::stride]
    if pos.shape[0] == 0:
        return np.zeros((positions.shape[1], 3))

    if not unwrap:
        return pos.mean(axis=0)

    fracs = to_fractional(pos, cel)
    fracs = unwrap_fractional(fracs)
    mean_frac = fracs.mean(axis=0)                    # (N, 3)
    mean_frac -= np.floor(mean_frac)                  # wrap into [0, 1)
    ref_cell = cel.mean(axis=0)                       # mean cell handles NPT
    return mean_frac @ ref_cell
