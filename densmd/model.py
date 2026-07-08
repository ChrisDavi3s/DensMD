"""Data model and compute core: trajectory in, scalar fields out.

``load()`` reads a trajectory once and bins each species into a per-atom
histogram plus an averaged-position array. From there, every UI geometry
change (ROI bounds, Miller plane, smoothing sigma) runs through ``region()``,
``smoothed()`` and ``volume_data()`` / ``sample_on_plane()`` to produce the
scalar fields ``render.py`` draws.

Only *geometry* triggers work here. Appearance (colour, opacity, gamma,
colormap) never touches this module : render.py recomputes that downstream,
so dragging an opacity slider costs nothing on this side.

Caches, cheaply keyed:
  * smoothed histograms          -> key (atype, sigma)   [raw data is static]
  * region (bounds/mask/focus)   -> key (roi, miller)
No hashing of multi-million-element arrays anywhere.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates

from .config import Settings
from .io import LoadSpec
from .miller import MillerParams, voxel_axes, voxel_mask
from .unwrap import representative_positions


# ---------------------------------------------------------------------------
# Result value objects
# ---------------------------------------------------------------------------
@dataclass
class Region:
    """Geometry state shared by every species for one update."""

    roi_indices: Dict[str, int]
    phys_min: np.ndarray
    phys_max: np.ndarray
    miller: MillerParams
    mask: Optional[np.ndarray]           # ROI-shaped bool, or None
    focal_point: np.ndarray


@dataclass
class VolumeData:
    """Cached, geometry-dependent inputs for a histogram volume's RGBA.

    Reproduces the original app's approach: the volume is rendered from a
    per-voxel RGBA array (``add_volume(scalars="rgba")``) with *no* VTK transfer
    function -- so VTK never allocates a range-sized transfer texture (the cause
    of the red-cube mis-mapping). Computed once per *geometry* change; appearance
    tweaks only re-run the cheap RGBA remap (in render) over these arrays.
    """

    data: np.ndarray                     # ROI density (original scale)
    quantile: np.ndarray                 # quantile-transformed density, 0..255
    mask: Optional[np.ndarray]           # ROI-shaped bool, or None
    origin: np.ndarray
    spacing: np.ndarray


def quantile_transform(data: np.ndarray, multiplier: float = 255.0) -> np.ndarray:
    """Uniformise ``data`` onto [0, multiplier] by rank (empirical CDF).

    Rank-based numpy replacement for sklearn's ``quantile_transform`` (the
    routine the original app used). Robust to the many tied empty voxels; the
    minimum maps to 0 and stays invisible. The reference is subsampled so the
    sort stays cheap even at high grid resolution.
    """
    flat = np.ascontiguousarray(data.ravel(), dtype=np.float32)
    if flat.size == 0:
        return np.zeros_like(data, dtype=np.float32)
    step = max(1, flat.size // 100_000)
    order = np.sort(flat[::step])
    ranks = np.searchsorted(order, flat, side="left")
    scale = np.float32(multiplier / max(order.size - 1, 1))
    return (ranks * scale).astype(np.float32).reshape(data.shape)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class DensityModel:
    """Owns loaded data and all heavy computation."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.atom_data: Dict[str, Dict] = {}
        self.species: List[str] = []
        self.origin = np.zeros(3)
        self.spacing = np.ones(3)
        self.cell_center = np.zeros(3)
        self.dims = np.zeros(3, dtype=int)
        self._smooth_cache: Dict[Tuple[str, int], np.ndarray] = {}
        self._voldata_cache: Dict[Tuple, VolumeData] = {}
        self._region_cache: Optional[Tuple[Tuple, Region]] = None
        self._gmin = np.zeros(3)
        self._gmax = np.ones(3)

    # -- scanning (for the Open dialog) ----------------------------------
    @staticmethod
    def scan_species(spec: LoadSpec) -> List[str]:
        """Read only the first frame and return its (mapped) unique species."""
        first = spec.load_first_frame()
        return sorted(set(spec.mapped_symbols(first)))

    # -- loading ---------------------------------------------------------
    def load(self, spec: LoadSpec) -> None:
        """Load a trajectory and precompute per-species data."""
        frames = spec.load_frames()
        if not frames:
            raise ValueError("No frames loaded. Check file path and slice.")

        symbols = spec.mapped_symbols(frames[0])
        self.species = sorted(set(symbols))
        idx_map = {s: np.array([i for i, sym in enumerate(symbols) if sym == s])
                   for s in self.species}

        positions = np.stack([f.get_positions() for f in frames], axis=0)
        cells = np.stack([np.array(f.get_cell()) for f in frames], axis=0)

        # For NPT (cell varies frame to frame), remap every frame into the mean
        # cell via fractional coordinates so the density grid is consistent.
        # For NVT this is a no-op, so behaviour is unchanged.
        ref_cell = cells[0]
        hist_positions = positions
        if not np.allclose(cells, cells[0]):
            ref_cell = cells.mean(axis=0)
            fracs = np.einsum("fnj,fjk->fnk", positions, np.linalg.inv(cells))
            fracs -= np.floor(fracs)                  # wrap into [0, 1)
            hist_positions = fracs @ ref_cell

        corners = np.array([np.dot([i, j, k], ref_cell)
                            for i in (0, 1) for j in (0, 1) for k in (0, 1)])
        self._gmin, self._gmax = corners.min(axis=0), corners.max(axis=0)

        self.atom_data = self._process_species(positions, hist_positions, cells,
                                               idx_map, self._gmin, self._gmax)
        self._apply_grid(self.settings.grid_resolution)

        self._region_cache = None
        self._voldata_cache.clear()
        del positions, hist_positions, cells, frames

    def rebuild_grid(self, resolution: int) -> None:
        """Rebuild histograms at a new grid resolution from stored positions."""
        if not self.species:
            return
        self.settings.grid_resolution = int(resolution)
        self._apply_grid(int(resolution))
        self._region_cache = None
        self._voldata_cache.clear()

    def _process_species(self, positions, hist_positions, cells, idx_map,
                         roi_min, roi_max) -> Dict[str, Dict]:
        stride = max(int(self.settings.average_subsample), 1)
        out: Dict[str, Dict] = {}
        for atype, indices in idx_map.items():
            # Histogram uses (NPT-remapped) positions; averaging uses the raw
            # trajectory + per-frame cells so periodic averaging stays correct.
            # All three position statistics are precomputed here because the
            # raw trajectory is discarded after load; storing N x 3 per method
            # is trivial and lets the UI switch methods instantly.
            allpos = hist_positions[:, indices, :].reshape(-1, 3)
            in_roi = np.all((allpos >= roi_min) & (allpos <= roi_max), axis=1)
            reps = {}
            for method in ("mode", "mean", "naive"):
                rep = representative_positions(positions[:, indices, :], cells,
                                               stride=stride, method=method)
                keep = np.all((rep >= roi_min) & (rep <= roi_max), axis=1)
                reps[method] = rep[keep]
            out[atype] = {
                "global_positions": allpos[in_roi].astype(np.float32),
                "individual_averages": reps,
            }
        return out

    def _apply_grid(self, res: int) -> None:
        """Set grid parameters and (re)compute per-species histograms."""
        gmin, gmax = self._gmin, self._gmax
        span = np.maximum(gmax - gmin, 1e-6)
        self.origin = gmin
        self.dims = np.array([res, res, res])
        self.spacing = np.where(span / (res - 1) == 0, 1e-6, span / (res - 1))
        self.cell_center = gmin + span / 2.0
        self._smooth_cache.clear()

        for atype in self.species:
            pos = self.atom_data[atype]["global_positions"]
            if pos.size == 0:
                hist = np.zeros((res, res, res), dtype=np.float32)
            else:
                clipped = np.clip(pos, gmin, gmax - 1e-9)
                idx = ((clipped - gmin) / self.spacing).astype(np.int32)
                flat_idx = idx[:, 0] * (res**2) + idx[:, 1] * res + idx[:, 2]
                hist = np.bincount(flat_idx, minlength=res**3).reshape(res, res, res).astype(np.float32)
            self.atom_data[atype]["raw_hist"] = hist

    # -- smoothing cache -------------------------------------------------
    def smoothed(self, atype: str, sigma: int) -> np.ndarray:
        """Gaussian-smoothed full histogram, cached by (atype, sigma)."""
        raw = self.atom_data[atype]["raw_hist"]
        if sigma <= 0:
            return raw
        key = (atype, sigma)
        cached = self._smooth_cache.get(key)
        if cached is None:
            cached = gaussian_filter(raw, sigma=sigma).astype(np.float32)
            self._smooth_cache[key] = cached
        return cached

    # -- region ----------------------------------------------------------
    def region(self, roi_indices: Dict[str, int], miller: MillerParams) -> Region:
        """Physical bounds, Miller mask and focal point for the current ROI."""
        key = (tuple(sorted(roi_indices.items())), miller.key())
        if self._region_cache and self._region_cache[0] == key:
            return self._region_cache[1]

        phys_min = self.origin + np.array(
            [roi_indices["xmin"], roi_indices["ymin"], roi_indices["zmin"]]) * self.spacing
        phys_max = self.origin + np.array(
            [roi_indices["xmax"], roi_indices["ymax"], roi_indices["zmax"]]) * self.spacing

        mask = None
        focal = 0.5 * (phys_min + phys_max)
        if miller.normal is not None:
            axes = voxel_axes(roi_indices, self.origin, self.spacing)
            mask = voxel_mask(axes, self.cell_center, miller)
            sm = mask[::5, ::5, ::5]
            if np.any(sm):
                ix, iy, iz = np.where(sm)
                focal = np.array([
                    axes[0][::5][ix].mean(),
                    axes[1][::5][iy].mean(),
                    axes[2][::5][iz].mean()
                ])

        region = Region(roi_indices, phys_min, phys_max, miller, mask, focal)
        self._region_cache = (key, region)
        return region

    # -- volume data for the histogram mode ------------------------------
    def volume_data(self, atype: str, sigma: int, region: Region,
                    smooth_before: bool) -> Optional[VolumeData]:
        """ROI density + quantile transform for the histogram mode.

        The colour/opacity RGBA is *not* computed here -- render does that from
        these arrays so appearance tweaks stay cheap.
        """
        # _rebuild recomputes every visible species on ANY geometry signal;
        # cache so mode/shell toggles don't re-slice + re-quantile 27M voxels.
        roi = region.roi_indices
        key = (atype, sigma, tuple(sorted(roi.items())),
               region.miller.key(), smooth_before)
        if key in self._voldata_cache:
            return self._voldata_cache[key]
        roi_slice = (
            slice(roi["xmin"], roi["xmax"] + 1),
            slice(roi["ymin"], roi["ymax"] + 1),
            slice(roi["zmin"], roi["zmax"] + 1),
        )
        if smooth_before:
            data = self.smoothed(atype, sigma)[roi_slice]
        else:
            data = self.atom_data[atype]["raw_hist"][roi_slice]
            if sigma > 0:
                data = gaussian_filter(data, sigma=sigma)
        data = np.ascontiguousarray(data, dtype=np.float32)
        if data.size == 0:
            return None

        quantile = quantile_transform(data, multiplier=255.0)
        origin = self.origin + np.array(
            [roi["xmin"], roi["ymin"], roi["zmin"]]) * self.spacing
        vol = VolumeData(data, quantile, region.mask, origin, self.spacing)
        if len(self._voldata_cache) > 8:      # bound memory
            self._voldata_cache.clear()
        self._voldata_cache[key] = vol
        return vol

    # -- sampling for the Miller-plane mode ------------------------------
    def sample_on_plane(self, atype: str, sigma: int, points: np.ndarray,
                        smooth_before: bool) -> np.ndarray:
        """Interpolate the (optionally smoothed) histogram onto plane points."""
        if smooth_before and sigma > 0:
            data = self.smoothed(atype, sigma)
        else:
            data = self.atom_data[atype]["raw_hist"]
        inv = np.array([1.0 / s if s else 0.0 for s in self.spacing])
        idx = (points - self.origin) * inv
        order = self.settings.miller_sample_order
        return map_coordinates(data, idx.T, order=order, mode="constant", cval=0.0)
