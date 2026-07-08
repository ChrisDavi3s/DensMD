"""Representative atom positions under periodic boundary conditions.

The problem: where do you *draw* an atom given its whole trajectory? Naive
averaging of wrapped coordinates puts boundary hoppers in the middle of the
box. A periodic-aware (circular) mean fixes that, but no mean can represent a
genuinely two-site atom -- an ion flickering between two sites averages to
somewhere between them, a spot it never occupies.

So the default statistic is the **mode**: the place the atom is most often
found. For each atom we score every (subsampled) frame position by the local
density of the atom's own trajectory around it -- a Gaussian kernel over
minimum-image distances, so periodic images are irrelevant -- pick the
densest sample, and return the circular mean of just that dominant cluster.
A vibrating atom gives its site centre; a 70/30 two-site hopper gives the
70% site; nothing ever lands in an unvisited gap.

Methods (Settings > Averaged positions):
  * "mode"  -- most-visited site (default, recommended)
  * "mean"  -- circular mean on the torus (image-invariant)
  * "naive" -- plain Cartesian mean of the stored coordinates

Everything here is pure numpy and independent of Qt/VTK. Per-frame cells keep
all methods correct for NPT trajectories.
"""
from __future__ import annotations

import numpy as np


def to_fractional(positions: np.ndarray, cells: np.ndarray) -> np.ndarray:
    """Cartesian -> fractional per frame. Shapes (F, N, 3) and (F, 3, 3)."""
    inv = np.linalg.inv(cells)                       # (F, 3, 3)
    return np.einsum("fnj,fjk->fnk", positions, inv)


def circular_mean_fractional(fracs: np.ndarray) -> np.ndarray:
    """Per-atom circular mean of fractional coords along the frame axis.

    (F, N, 3) -> (N, 3), each component wrapped into [0, 1). Integer image
    offsets vanish in sin/cos, so wrapped and unwrapped input agree.
    """
    ang = fracs * (2.0 * np.pi)
    mean = np.arctan2(np.sin(ang).mean(axis=0),
                      np.cos(ang).mean(axis=0)) / (2.0 * np.pi)
    return mean - np.floor(mean)


def mode_fractional(fracs: np.ndarray, cell_lengths: np.ndarray,
                    max_samples: int = 256, radius: float = 0.75,
                    chunk: int = 32) -> np.ndarray:
    """Per-atom modal position in fractional coords, (F, N, 3) -> (N, 3).

    For each atom: kernel-density score of every sampled frame position over
    minimum-image distances, argmax picks the densest sample, and the result
    is the min-image mean of that sample's own cluster. ``radius`` (Angstrom)
    sets the kernel width -- it should swallow a site's vibration cloud
    (~0.2-0.5 A) but keep distinct sites (>~2 A apart) separate.
    """
    F, N, _ = fracs.shape
    step = -(-F // max_samples)                      # ceil: M never exceeds cap
    s = fracs[::step]                                # (M, N, 3)
    M = s.shape[0]
    r_frac = radius / np.maximum(cell_lengths, 1e-9)  # per-axis kernel width
    out = np.empty((N, 3))
    for c0 in range(0, N, chunk):
        a = s[:, c0:c0 + chunk, :]                   # (M, C, 3)
        C = a.shape[1]
        d = a[:, None, :, :] - a[None, :, :, :]      # (M, M, C, 3)
        d -= np.round(d)                             # minimum image
        w = np.exp(-0.5 * np.sum((d / r_frac) ** 2, axis=-1))  # (M, M, C)
        k = w.sum(axis=1).argmax(axis=0)             # densest sample per atom
        cols = np.arange(C)
        anchor = a[k, cols, :]                       # (C, 3)
        near = w[k, :, cols] > 0.3                   # (C, M): its cluster
        rel = a.transpose(1, 0, 2) - anchor[:, None, :]
        rel -= np.round(rel)
        counts = np.maximum(near.sum(axis=1, keepdims=True), 1)
        mode = anchor + (rel * near[..., None]).sum(axis=1) / counts
        out[c0:c0 + C] = mode - np.floor(mode)
    return out


def representative_positions(positions: np.ndarray, cells: np.ndarray,
                             stride: int = 1,
                             method: str = "mode") -> np.ndarray:
    """Per-atom representative position in Cartesian coordinates.

    Args:
        positions: (F, N, 3) trajectory for one species.
        cells: (F, 3, 3) per-frame cells (constant for NVT, varying for NPT).
        stride: subsample every ``stride`` frames for speed.
        method: "mode" (most-visited site), "mean" (circular mean), or
            "naive" (plain Cartesian mean).

    Returns:
        (N, 3) positions inside the mean cell.
    """
    stride = max(int(stride), 1)
    pos = positions[::stride]
    cel = cells[::stride]
    if pos.shape[0] == 0:
        return np.zeros((positions.shape[1], 3))

    if method == "naive":
        return pos.mean(axis=0)

    fracs = to_fractional(pos, cel)
    ref_cell = cel.mean(axis=0)                       # mean cell handles NPT
    if method == "mean":
        return circular_mean_fractional(fracs) @ ref_cell
    lengths = np.linalg.norm(ref_cell, axis=1)
    return mode_fractional(fracs - np.floor(fracs), lengths) @ ref_cell
