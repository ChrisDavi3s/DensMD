"""Application settings:

Grid resolution, render quality, default ranges, timing. Can be edited live from the
Settings dialog and round-tripped to ``~/.densmd.json`` between runs.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import List

SETTINGS_PATH = Path.home() / ".densmd.json"

# Colormaps offered in the per-atom controls.
# See https://matplotlib.org/stable/users/explain/colors/colormaps.html
DEFAULT_COLORMAPS: List[str] = [
    "coolwarm", "viridis", "jet", "plasma", "inferno",
    "magma", "cividis", "Reds", "Blues", "Greens", "Purples",
]

# VTK render-window stereo modes exposed by the Stereo control.
STEREO_MODES: List[str] = [
    "Anaglyph", "Interlaced", "CrystalEyes", "SplitViewport", "Checkerboard",
]


@dataclass
class Settings:
    """Runtime configuration. Immutable defaults; mutate an instance freely."""

    # --- Grid / compute -------------------------------------------------
    grid_resolution: int = 300        # histogram bins per axis (N^3 voxels)
    gaussian_sigma: int = 2          # default smoothing sigma (voxels)
    quantile_bins: int = 256          # nodes used to build transfer functions
    miller_plane_res_factor: int = 4  # plane mesh res = grid_resolution * factor
    miller_sample_order: int = 3      # spline order for map_coordinates on plane

    # --- Averaged positions --------------------------------------------
    average_method: str = "mode"      # 'mode' (most-visited site), 'mean'
                                      # (circular mean), 'naive' (raw mean)
    average_subsample: int = 1        # use every Nth frame when averaging

    # --- Volume render quality -----------------------------------------
    volume_mapper: str = "smart"      # 'smart' (GPU, fast; soft oblique edges)
                                      # or 'fixed_point' (CPU, exact; slow)
    interpolation: str = "nearest"    # 'nearest' (sharp, default) or 'linear' (smooth)
    volume_sample_factor: float = 0.5  # ray step as fraction of voxel spacing
                                       # (smaller = crisper + slower)
    projection: str = "perspective"   # 'perspective' or 'orthographic'
    background_color: str = "#ffffff"  # render-window background
    depth_peeling: bool = True        # per-fragment volume compositing
    interactive_lod: bool = True      # coarse sampling + no peeling while the
                                      # camera is being dragged

    # --- Timing ---------------------------------------------------------
    update_delay_ms: int = 150        # debounce for geometry recomputes
    rotation_fps: int = 20
    rotation_azimuth: float = 0.5     # degrees per rotation frame

    # --- Window ---------------------------------------------------------
    window_width: int = 1400
    window_height: int = 900
    main_panel_ratio: int = 3         # plotter : control-panel width ratio

    # --- Visual ranges / defaults --------------------------------------
    sphere_size_range: tuple = (1, 50)
    sphere_size_default: int = 5
    density_range: tuple = (0, 255)
    density_lower_default: int = 77
    density_upper_default: int = 178
    opacity_default: int = 100
    gamma_range: tuple = (0.0, 2.0)
    gamma_default: float = 0.0

    # --- Isosurface mode -------------------------------------------------
    iso_shells_default: int = 4       # nested contour shells
    iso_tolerance_default: float = 0.05  # window fraction trimmed off each end
    iso_smooth_default: int = 20      # Taubin mesh-smoothing iterations

    # --- Miller ranges --------------------------------------------------
    miller_index_range: tuple = (-10, 10)
    miller_index_default: int = 1
    miller_thickness_range: tuple = (0.1, 20.0)
    miller_thickness_default: float = 2.0
    miller_offset_range: tuple = (-10.0, 10.0)
    miller_offset_default: float = 0.0

    colormaps: List[str] = field(default_factory=lambda: list(DEFAULT_COLORMAPS))

    # --- persistence ----------------------------------------------------
    def save(self, path: Path = SETTINGS_PATH) -> None:
        path.write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, path: Path = SETTINGS_PATH) -> "Settings":
        if not path.exists():
            return cls()
        try:
            data = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            return cls()
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        # tuples arrive from JSON as lists; coerce the ones we treat as tuples
        tuple_fields = {
            "sphere_size_range", "density_range", "gamma_range",
            "miller_index_range", "miller_thickness_range", "miller_offset_range",
        }
        clean = {}
        for k, v in data.items():
            if k not in known:
                continue
            clean[k] = tuple(v) if k in tuple_fields and isinstance(v, list) else v
        return cls(**clean)
