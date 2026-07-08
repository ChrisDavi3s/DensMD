"""Rendering layer: turns model output into on-screen PyVista/VTK actors.

Pipeline, per atom species, per update:

  1. ``model.py`` hands over geometry-dependent data: a ``VolumeData`` (ROI
     density + quantile array) for the Histogram and Isosurface modes,
     sampled scalars on a plane mesh for Miller Plane Slice, or raw points
     for Averaged Positions.
  2. This module turns that data plus an ``Appearance`` snapshot (cmap,
     density window, opacity, gamma, colour) into something VTK can draw:
     an RGBA volume texture (``histogram_rgba``), nested contour shells
     (``show_isosurface``), a scalar lookup table (``build_lut``), or
     coloured sphere glyphs.
  3. ``RenderView`` builds/replaces the actor and hands it to the plotter.

Two-tier updates keep dragging a slider cheap:

  set_* / show_*      -> geometry changed: rebuild the actor from scratch.
  update_appearance   -> only colour/opacity changed: re-run step 2 over the
                         cached arrays from step 1 and retint the existing
                         actor -- no histogram/smoothing/model work, no
                         camera move.

Histogram volumes are painted as per-voxel RGBA with no VTK transfer function,
which avoids VTK's transfer-texture size limits and the colour mis-mapping
they cause at high grid resolution.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt
import vtk

from .config import Settings
from .miller import MillerParams


@dataclass
class Appearance:
    """Snapshot of one species' appearance controls."""

    cmap: str = "coolwarm"
    density_lower: int = 77
    density_upper: int = 178
    opacity: int = 100          # percent
    gamma: float = 0.0
    normalize_in_range: bool = False
    color: str = "#ff0000"
    sphere_size: int = 5
    # Isosurface mode
    iso_shells: int = 4         # number of nested contour shells
    iso_tolerance: float = 0.05  # fraction of the window trimmed off each end
    iso_smooth: int = 20        # Taubin mesh-smoothing iterations


# ---------------------------------------------------------------------------
# Transfer-function helpers (pure, testable)
# ---------------------------------------------------------------------------
def _density_at(quantiles: np.ndarray, slider255: float) -> float:
    """Density value at the quantile a 0-255 slider selects."""
    p = np.linspace(0.0, 1.0, quantiles.size)
    return float(np.interp(slider255 / 255.0, p, quantiles))


def _alpha_curve(q255: np.ndarray, app: Appearance) -> np.ndarray:
    """Opacity for values already expressed on the 0-255 quantile scale."""
    lo, hi, gamma = app.density_lower, app.density_upper, app.gamma
    width = hi - lo
    base = np.zeros_like(q255, dtype=float)
    inwin = (q255 >= lo) & (q255 <= hi)
    if width > 1e-9:
        base[inwin] = (q255[inwin] - lo) / width
    elif inwin.any():
        base[inwin] = 1.0
    base[q255 > hi] = 1.0
    base = np.clip(base, 0.0, 1.0)
    if gamma == 0:
        alpha = np.where(base > 1e-6, 1.0, 0.0)
    else:
        alpha = base ** gamma
    return np.clip(alpha * (app.opacity / 100.0), 0.0, 1.0)


def _color_inputs(ds: np.ndarray, quantiles: np.ndarray, vmin: float,
                  vmax: float, app: Appearance) -> np.ndarray:
    """Colormap coordinates (0-1) for densities ``ds``."""
    if app.normalize_in_range:
        d_lo = _density_at(quantiles, app.density_lower)
        d_hi = max(_density_at(quantiles, app.density_upper), d_lo)
        rng = max(d_hi - d_lo, 1e-9)
        return np.clip((ds - d_lo) / rng, 0.0, 1.0)
    return np.clip((ds - vmin) / max(vmax - vmin, 1e-9), 0.0, 1.0)


def iso_levels(lower: float, upper: float, shells: int,
               tolerance: float) -> np.ndarray:
    """Quantile levels (0-255 scale) for the isosurface shells.

    Shells are spread evenly across the density lower..upper window;
    ``tolerance`` trims that fraction of the window off each end so the
    outermost/innermost shells sit clear of the fuzzy extremes.
    """
    lo, hi = float(min(lower, upper)), float(max(lower, upper))
    pad = (hi - lo) * float(tolerance)
    levels = np.linspace(lo + pad, hi - pad, max(int(shells), 1))
    return np.unique(np.clip(levels, 0.5, 254.5))


def iso_shell_style(vol, levels: np.ndarray, app: Appearance):
    """(rgb, alpha) per shell, matching ``histogram_rgba`` semantics.

    Colour follows the same convention as the volume: true-density range by
    default, or spread across the window when normalize_in_range. Opacity
    ramps outer->inner across the window, shaped by gamma (gamma 0 = all
    shells fully opaque), scaled by the opacity slider.
    """
    lo, hi = float(app.density_lower), float(app.density_upper)
    width = max(hi - lo, 1e-9)
    data = vol.data
    valid = data[vol.mask] if vol.mask is not None else data
    valid = valid[np.isfinite(valid)]
    if valid.size == 0:
        valid = np.array([0.0])
    if app.normalize_in_range:
        t = (levels - lo) / width
    else:
        d_at = np.quantile(valid, np.clip(levels / 255.0, 0.0, 1.0))
        dmin, dmax = float(valid.min()), float(valid.max())
        t = (d_at - dmin) / max(dmax - dmin, 1e-9)
    rgb = plt.get_cmap(app.cmap)(np.clip(t, 0.0, 1.0))[:, :3]
    base = np.clip((levels - lo) / width, 0.0, 1.0)
    alpha = np.ones_like(base) if app.gamma == 0 else base ** app.gamma
    alpha = np.clip(alpha * (app.opacity / 100.0), 0.0, 1.0)
    return rgb, alpha


def iso_lut(vol, levels: np.ndarray, app: Appearance):
    """Discrete per-shell lookup table + scalar range for the merged mesh."""
    rgb, alpha = iso_shell_style(vol, levels, app)
    n = levels.size
    half = (levels[-1] - levels[0]) / max(n - 1, 1) / 2.0 if n > 1 else 0.5
    lut = vtk.vtkLookupTable()
    lut.SetNumberOfTableValues(n)
    lut.SetTableRange(levels[0] - half, levels[-1] + half)
    for i in range(n):
        lut.SetTableValue(i, float(rgb[i, 0]), float(rgb[i, 1]),
                          float(rgb[i, 2]), float(alpha[i]))
    lut.Build()
    return lut, levels[0] - half, levels[-1] + half


# Per-voxel RGBA, exactly as the original app did it -- NO VTK transfer
# function, so VTK never allocates a range-sized transfer texture (the cause of
# the red-cube mis-mapping). Colour comes from the true-density range; opacity
# from the quantile window; gamma shapes the ramp. Cheap enough to re-run on
# appearance changes over the cached VolumeData arrays.
def histogram_rgba(vol, app: Appearance) -> np.ndarray:
    """Return an (X, Y, Z, 4) uint8 RGBA array for a VolumeData ``vol``."""
    data, quant, mask = vol.data, vol.quantile, vol.mask
    lower, upper = float(app.density_lower), float(app.density_upper)
    max_opacity, gamma = app.opacity / 100.0, app.gamma
    cmap = plt.get_cmap(app.cmap)

    # opacity from the quantile window
    opacity = np.zeros_like(quant, dtype=float)
    width = upper - lower
    inwin = (quant >= lower) & (quant <= upper)
    if width > 1e-9:
        opacity[inwin] = (quant[inwin] - lower) / width
    elif inwin.any():
        opacity[inwin] = 1.0
    opacity[quant > upper] = 1.0

    # colour from the true-density range (respecting the Miller mask)
    ranging = data.astype(float)
    if mask is not None:
        ranging = ranging.copy()
        ranging[~mask] = np.nan
    valid = ranging[np.isfinite(ranging)]
    if valid.size == 0:
        valid = np.array([0.0])
    dmin, dmax = float(valid.min()), float(valid.max())
    drange = dmax - dmin
    if not app.normalize_in_range:
        cin = (data - dmin) / drange if drange > 1e-9 else np.zeros_like(data)
    else:
        lo_d = np.quantile(valid, lower / 255.0)
        hi_d = max(np.quantile(valid, upper / 255.0), lo_d)
        sel = hi_d - lo_d
        cin = ((data - lo_d) / sel if sel > 1e-9
               else np.where(np.isclose(data, lo_d), 1.0, 0.0))
    cin = np.clip(cin, 0.0, 1.0)
    rgb = (cmap(cin)[..., :3] * 255).astype(np.uint8)

    # gamma + max opacity
    opacity = np.clip(opacity, 0.0, 1.0)
    if gamma == 0:
        alpha = np.where(opacity > 1e-6, 1.0, 0.0)
    else:
        alpha = opacity ** gamma
    alpha = np.clip(alpha * max_opacity * 255, 0, 255).astype(np.uint8)
    if mask is not None:
        alpha[~mask] = 0

    rgba = np.empty((*data.shape, 4), dtype=np.uint8)
    rgba[..., :3] = rgb
    rgba[..., 3] = alpha
    return rgba


def build_lut(values: np.ndarray, app: Appearance, size: int = 256) -> vtk.vtkLookupTable:
    """Lookup table (colour + alpha) for a scalar-coloured mesh (plane mode)."""
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        finite = np.array([0.0])
    vmin, vmax = float(finite.min()), float(finite.max())
    quantiles = np.quantile(finite, np.linspace(0.0, 1.0, 256))
    ps = np.linspace(0.0, 1.0, quantiles.size)

    lut = vtk.vtkLookupTable()
    lut.SetNumberOfTableValues(size)
    lut.SetTableRange(vmin, vmax)
    ds = np.linspace(vmin, vmax, size)
    rgb = plt.get_cmap(app.cmap)(_color_inputs(ds, quantiles, vmin, vmax, app))[:, :3]
    q255 = np.interp(ds, quantiles, ps) * 255.0     # each density -> its quantile
    alpha = _alpha_curve(q255, app)
    for i in range(size):
        lut.SetTableValue(i, float(rgb[i, 0]), float(rgb[i, 1]),
                          float(rgb[i, 2]), float(alpha[i]))
    lut.Build()
    return lut, vmin, vmax


# ---------------------------------------------------------------------------
# Interaction styles
# ---------------------------------------------------------------------------
class _PanStyle(vtk.vtkInteractorStyleTrackballCamera):
    """Trackball style with left-drag remapped to pan (grab & move).

    vtkInteractorStyle skips its built-in handler for any event that has an
    observer, so these observers *replace* the default rotate behaviour.
    Middle-drag still pans, right-drag still zooms.
    """

    def __init__(self):
        super().__init__()
        self.AddObserver("LeftButtonPressEvent", self._press)
        self.AddObserver("LeftButtonReleaseEvent", self._release)

    def _press(self, *_):
        iren = self.GetInteractor()
        x, y = iren.GetEventPosition()
        self.FindPokedRenderer(x, y)
        self.StartPan()

    def _release(self, *_):
        self.EndPan()


# ---------------------------------------------------------------------------
# Render view
# ---------------------------------------------------------------------------
class RenderView:
    """Wraps the plotter and owns per-species actors + their retint state."""

    def __init__(self, plotter, settings: Settings):
        self.plotter = plotter
        self.settings = settings
        self._entries: Dict[str, Dict[str, Any]] = {}
        self._had_actors = False
        self.interpolation = settings.interpolation
        self.sample_factor = settings.volume_sample_factor
        self.mapper_mode = settings.volume_mapper
        self._style = None  # keep the active interactor style alive
        try:
            iren = plotter.render_window.GetInteractor()
            style = iren.GetInteractorStyle() if iren is not None else None
            if style is not None:
                self._attach_lod(style)
        except Exception:
            pass

    # -- clearing --------------------------------------------------------
    def clear_all(self) -> None:
        for atype in list(self._entries):
            self.clear(atype)

    def clear(self, atype: str) -> None:
        entry = self._entries.pop(atype, None)
        if entry and entry.get("actor") is not None:
            self.plotter.remove_actor(entry["actor"])

    # -- geometry: build actors -----------------------------------------
    def show_volume(self, atype: str, vol_data, app: Appearance) -> None:
        self.clear(atype)
        rgba = histogram_rgba(vol_data, app)
        image = pv.ImageData()
        image.dimensions = np.array(vol_data.data.shape)
        image.origin = vol_data.origin
        image.spacing = vol_data.spacing
        image.point_data["rgba"] = rgba.reshape(-1, 4, order="F")
        # 'smart' (GPU) is fast but leaves a soft semi-transparent band at
        # oblique edges/corners even for binary alpha -- baked into VTK's GPU
        # ray integration, no mapper knob fixes it. 'fixed_point' (CPU) is
        # pixel-exact but slow on large grids, hence the render setting.
        actor = self.plotter.add_volume(
            image, scalars="rgba", mapper=self.mapper_mode, blending="composite",
            shade=False, show_scalar_bar=False,
        )
        self._tune_volume(actor, vol_data.spacing)
        self._entries[atype] = {"kind": "volume", "actor": actor,
                                "image": image, "vol": vol_data, "app": app}
        self._had_actors = True

    def _tune_volume(self, actor, spacing) -> None:
        """Set crisp, non-adaptive volume sampling + interpolation.

        The default smart mapper *auto-adjusts* the ray step (coarse while
        interacting), which leaves the volume permanently soft. Pinning a fine
        step gives a full-quality, stable render independent of grid resolution.
        """
        if self.interpolation == "nearest":
            actor.GetProperty().SetInterpolationTypeToNearest()
        else:
            actor.GetProperty().SetInterpolationTypeToLinear()
        try:
            mapper = actor.mapper
            mapper.SetAutoAdjustSampleDistances(False)
            mapper.SetSampleDistance(float(np.min(spacing)) * self.sample_factor)
        except Exception:
            pass

    def set_interpolation(self, mode: str) -> None:
        self.interpolation = mode
        for e in self._entries.values():
            if e["kind"] == "volume" and e.get("actor") is not None:
                self._tune_volume(e["actor"], e["vol"].spacing)
        self.plotter.render()

    def set_sample_factor(self, factor: float) -> None:
        self.sample_factor = max(0.05, float(factor))
        for e in self._entries.values():
            if e["kind"] == "volume" and e.get("actor") is not None:
                self._tune_volume(e["actor"], e["vol"].spacing)
        self.plotter.render()

    def set_mapper(self, mode: str) -> None:
        """'smart' (GPU, fast) or 'fixed_point' (CPU, exact edges). Rebuilds."""
        self.mapper_mode = mode
        for atype in list(self._entries):
            e = self._entries[atype]
            if e["kind"] == "volume" and e.get("actor") is not None:
                self.show_volume(atype, e["vol"], e["app"])
        self.plotter.render()

    # -- interactive LOD ---------------------------------------------------
    # While the camera is being dragged: volumes drop to coarse auto-adjusted
    # ray sampling, and depth peeling is suspended (translucent gamma > 0
    # scenes pay for multi-pass peeling *every frame*, which is what makes
    # dragging crawl). Both snap back to full quality on release.
    def _attach_lod(self, style) -> None:
        style.AddObserver("StartInteractionEvent", self._lod_coarse)
        style.AddObserver("EndInteractionEvent", self._lod_fine)

    def _volume_mappers(self):
        for e in self._entries.values():
            if e["kind"] == "volume" and e.get("actor") is not None:
                yield e

    def _lod_coarse(self, *_) -> None:
        if not self.settings.interactive_lod:
            return
        self.plotter.renderer.SetUseDepthPeeling(False)
        self.plotter.renderer.SetUseDepthPeelingForVolumes(False)
        for e in self._volume_mappers():
            try:
                e["actor"].mapper.SetAutoAdjustSampleDistances(True)
            except Exception:
                pass

    def _lod_fine(self, *_) -> None:
        peel = bool(self.settings.depth_peeling)
        self.plotter.renderer.SetUseDepthPeeling(peel)
        self.plotter.renderer.SetUseDepthPeelingForVolumes(peel)
        for e in self._volume_mappers():
            self._tune_volume(e["actor"], e["vol"].spacing)
        self.plotter.render()

    def show_isosurface(self, atype: str, vol_data, app: Appearance) -> None:
        """Nested contour shells extracted from the quantile field.

        One marching-cubes pass over all shell levels, merged into a single
        mesh whose vertex scalar is its shell level; a discrete LUT colours
        each shell. Rasterized, so it stays fast and razor-sharp under
        rotation -- no ray marching.
        """
        self.clear(atype)
        levels = iso_levels(app.density_lower, app.density_upper,
                            app.iso_shells, app.iso_tolerance)
        quant = vol_data.quantile
        if vol_data.mask is not None:
            # close surfaces at the Miller-slab boundary
            quant = np.where(vol_data.mask, quant, -1.0).astype(np.float32)
        image = pv.ImageData()
        image.dimensions = np.array(quant.shape)
        image.origin = vol_data.origin
        image.spacing = vol_data.spacing
        image.point_data["q"] = quant.ravel(order="F")
        # flying edges: multi-threaded marching cubes for image data
        mesh = image.contour(isosurfaces=list(levels), scalars="q",
                             method="flying_edges")
        if mesh.n_points == 0:
            self._entries[atype] = {"kind": "iso", "actor": None,
                                    "vol": vol_data, "app": app,
                                    "levels": levels}
            return
        if app.iso_smooth > 0:
            mesh = mesh.smooth_taubin(n_iter=int(app.iso_smooth), pass_band=0.1)
        # Cheap smooth shading: flying edges already emits consistently wound
        # triangles, so skip the (slow) consistency/splitting passes pyvista's
        # smooth_shading would run and just compute plain point normals.
        mesh = mesh.compute_normals(point_normals=True, cell_normals=False,
                                    split_vertices=False,
                                    consistent_normals=False,
                                    auto_orient_normals=False)
        lut, rmin, rmax = iso_lut(vol_data, levels, app)
        actor = self.plotter.add_mesh(
            mesh, scalars="q", show_scalar_bar=False,
        )
        actor.prop.SetInterpolationToGouraud()
        actor.mapper.SetLookupTable(lut)
        actor.mapper.SetScalarRange(rmin, rmax)
        self._entries[atype] = {"kind": "iso", "actor": actor,
                                "vol": vol_data, "app": app, "levels": levels}
        self._had_actors = True

    def show_plane(self, atype: str, mesh: pv.PolyData, values: np.ndarray,
                   app: Appearance) -> None:
        self.clear(atype)
        mesh.point_data["density"] = values
        lut, vmin, vmax = build_lut(values, app)
        actor = self.plotter.add_mesh(
            mesh, scalars="density", show_scalar_bar=False, lighting=False,
        )
        actor.mapper.SetLookupTable(lut)
        actor.mapper.SetScalarRange(vmin, vmax)
        self._entries[atype] = {"kind": "plane", "actor": actor, "mesh": mesh,
                                "values": values}
        self._had_actors = True

    def show_glyphs(self, atype: str, points: np.ndarray, app: Appearance) -> None:
        self.clear(atype)
        if points.size == 0:
            self._entries[atype] = {"kind": "glyph", "actor": None}
            return
        radius = app.sphere_size * 0.1
        glyphs = pv.PolyData(points).glyph(geom=pv.Sphere(radius=radius))
        actor = self.plotter.add_mesh(glyphs, color=app.color, name=f"{atype}_avg")
        self._entries[atype] = {"kind": "glyph", "actor": actor, "points": points}
        self._had_actors = True

    # -- appearance: retint only ----------------------------------------
    def update_appearance(self, atype: str, app: Appearance) -> bool:
        """Retint an existing actor. Returns False if geometry rebuild needed."""
        entry = self._entries.get(atype)
        if not entry or entry.get("actor") is None:
            return False
        kind = entry["kind"]
        if kind == "volume":
            # Rebuild the volume actor from the cached arrays (no model recompute).
            self.show_volume(atype, entry["vol"], app)
            return True
        elif kind == "iso":
            # Same shell geometry (levels + smoothing unchanged) -> swap the
            # LUT only: opacity/gamma/cmap/norm drags cost nothing. Only a
            # changed window/shell/smoothing re-contours (from cache).
            levels = iso_levels(app.density_lower, app.density_upper,
                                app.iso_shells, app.iso_tolerance)
            same_mesh = (np.array_equal(levels, entry.get("levels"))
                         and app.iso_smooth == entry["app"].iso_smooth)
            if not same_mesh:
                self.show_isosurface(atype, entry["vol"], app)
                return True
            lut, rmin, rmax = iso_lut(entry["vol"], levels, app)
            entry["actor"].mapper.SetLookupTable(lut)
            entry["actor"].mapper.SetScalarRange(rmin, rmax)
            entry["app"] = app
        elif kind == "plane":
            lut, vmin, vmax = build_lut(entry["values"], app)
            entry["actor"].mapper.SetLookupTable(lut)
            entry["actor"].mapper.SetScalarRange(vmin, vmax)
        elif kind == "glyph":
            rgb = tuple(int(app.color[i:i + 2], 16) / 255.0 for i in (1, 3, 5))
            entry["actor"].prop.SetColor(*rgb)
        else:
            return False
        self.plotter.render()
        return True

    # -- camera ----------------------------------------------------------
    def set_focus(self, point: np.ndarray, reset: bool) -> None:
        if reset:
            self.plotter.reset_camera()
        self.plotter.camera.SetFocalPoint(*point)
        self.plotter.render()

    def reset_view(self) -> None:
        self.plotter.camera.up = (0, 0, 1)
        self.plotter.camera.position = (1, 1, 1)
        self.plotter.camera.focal_point = (0, 0, 0)
        self.plotter.reset_camera()
        self.plotter.render()

    def align_axis(self, axis: str) -> None:
        pos = {"x": (1, 0, 0), "y": (0, 1, 0), "z": (0, 0, 1)}[axis]
        self.plotter.camera.up = (0, 0, 1)
        self.plotter.camera.position = pos
        self.plotter.camera.focal_point = (0, 0, 0)
        self.plotter.reset_camera()
        self.plotter.render()

    def align_miller(self, miller: MillerParams, cell_center: np.ndarray) -> None:
        n = miller.normal if miller.use else np.array([miller.h, miller.k, miller.l], float)
        norm = np.linalg.norm(n)
        if norm == 0:
            return
        n = n / norm
        self.plotter.camera.up = (0, 0, 1)
        dist = np.linalg.norm(np.array(self.plotter.camera.position) - cell_center)
        self.plotter.camera.position = tuple(cell_center + n * dist)
        self.plotter.camera.focal_point = tuple(cell_center)
        self.plotter.reset_camera()
        self.plotter.render()

    def rotate(self, azimuth: float) -> None:
        self.plotter.camera.Azimuth(azimuth)
        self.plotter.render()

    # -- render / interaction settings ------------------------------------
    def set_projection(self, orthographic: bool) -> None:
        self.plotter.camera.parallel_projection = bool(orthographic)
        self.plotter.render()

    def set_background(self, color: str) -> None:
        self.plotter.set_background(color)
        self.plotter.render()

    def set_depth_peeling(self, on: bool) -> None:
        # both flags: volumes AND translucent meshes (isosurface shells)
        self.plotter.renderer.SetUseDepthPeeling(bool(on))
        self.plotter.renderer.SetUseDepthPeelingForVolumes(bool(on))
        self.plotter.render()

    def set_interaction(self, mode: str) -> None:
        """'rotate' (trackball, default) or 'pan' (left-drag grabs & moves)."""
        iren = self.plotter.render_window.GetInteractor()
        if iren is None:
            return
        style = _PanStyle() if mode == "pan" else vtk.vtkInteractorStyleTrackballCamera()
        style.SetDefaultRenderer(self.plotter.renderer)
        iren.SetInteractorStyle(style)
        self._attach_lod(style)
        self._style = style

    # -- stereo ----------------------------------------------------------
    def set_stereo(self, on: bool, mode: str) -> None:
        rw = self.plotter.render_window
        if rw is None:
            return
        setter = {
            "Anaglyph": rw.SetStereoTypeToAnaglyph,
            "Interlaced": rw.SetStereoTypeToInterlaced,
            "CrystalEyes": rw.SetStereoTypeToCrystalEyes,
            "SplitViewport": rw.SetStereoTypeToSplitViewportHorizontal,
            "Checkerboard": rw.SetStereoTypeToCheckerboard,
        }.get(mode)
        if setter:
            setter()
        rw.SetStereoRender(bool(on))
        rw.Render()

    # -- orientation axes ------------------------------------------------
    def show_axes(self, on: bool) -> None:
        """Toggle the orientation-axes marker (bottom-right corner)."""
        try:
            if on:
                self.plotter.add_axes(viewport=(0.8, 0.0, 1.0, 0.2))
            else:
                self.plotter.hide_axes()
        except Exception:
            pass
        self.plotter.render()

    @property
    def first_build(self) -> bool:
        return not self._had_actors
