"""Main application window: wires controls to model + render view.

Layout is a VTK plotter on the left and a scrollable control panel on the
right (toolbar for camera/view, per-atom ``AtomPanel``s, then global
ROI/Miller/render controls). Every control funnels into one of two signal
paths -- this routing is the whole point of the file:

  * geometry controls   -> debounced ``_schedule_geometry`` -> ``_rebuild``,
                          which re-reads the model and rebuilds actors.
  * appearance controls -> ``_retint``, which only re-runs the cheap
                          colour/opacity remap on the existing actor -- no
                          model recompute, no camera move.

``_load`` (File > Open) tears down the old per-atom panels, builds fresh ones
from the scanned species list, and resets the ROI sliders to the new grid
before the first ``_rebuild``.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pyvista as pv
from PyQt5 import QtCore, QtWidgets
from pyvistaqt import QtInteractor

from ..config import Settings, STEREO_MODES
from ..io import LoadSpec
from ..miller import MillerParams
from ..model import DensityModel
from ..render import RenderView
from .dialogs import OpenDialog, SettingsDialog
from .panels import AtomPanel
from .widgets import (ComboBox, CollapsibleBox, DoubleSpinBox, SpinBox,
                      labelled_slider, labelled_spinbox)

_ROI_LABELS = {
    "xmin": "X min", "xmax": "X max", "ymin": "Y min",
    "ymax": "Y max", "zmin": "Z min", "zmax": "Z max",
}


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, settings: Settings, spec: Optional[LoadSpec] = None):
        super().__init__()
        self.settings = settings
        self.setWindowTitle("DensMD")
        self.model = DensityModel(settings)
        self.panels: Dict[str, AtomPanel] = {}

        self._build_layout()
        self._build_toolbar()
        self._build_global_controls()
        self._build_menu()
        self._build_timers()

        self.render = RenderView(self.plotter, settings)
        self.render.show_axes(self.axes_act.isChecked())

        if spec is not None:
            self._load(spec)

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------
    def _build_layout(self) -> None:
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root = QtWidgets.QHBoxLayout(central)

        self.plotter = QtInteractor()
        self.plotter.renderer.SetUseDepthPeeling(self.settings.depth_peeling)
        self.plotter.renderer.SetUseDepthPeelingForVolumes(self.settings.depth_peeling)
        self.plotter.renderer.SetAmbient(1.0, 1.0, 1.0)
        self.plotter.camera.parallel_projection = (
            self.settings.projection == "orthographic")
        self.plotter.set_background(self.settings.background_color)
        root.addWidget(self.plotter, stretch=self.settings.main_panel_ratio)

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        root.addWidget(scroll, stretch=1)
        self.control_widget = QtWidgets.QWidget()
        self.control_layout = QtWidgets.QVBoxLayout(self.control_widget)
        scroll.setWidget(self.control_widget)

    def _build_menu(self) -> None:
        bar = self.menuBar()
        file_menu = bar.addMenu("&File")
        file_menu.addAction("&Open Trajectory…", self._open_dialog)
        file_menu.addAction("&Settings…", self._settings_dialog)
        file_menu.addSeparator()
        file_menu.addAction("&Quit", self.close)

    def _build_toolbar(self) -> None:
        """All viewer/camera controls live here; the side panel is data-only."""
        tb = self.addToolBar("Main")
        tb.setMovable(False)
        tb.addAction("Open…", self._open_dialog)
        tb.addAction("Settings…", self._settings_dialog)
        tb.addSeparator()

        # mouse interaction: rotate (default) vs pan/grab
        mode_group = QtWidgets.QActionGroup(self)
        mode_group.setExclusive(True)
        self.act_rotate = QtWidgets.QAction("Rotate", self)
        self.act_rotate.setCheckable(True)
        self.act_rotate.setChecked(True)
        self.act_rotate.setToolTip("Left-drag rotates the camera")
        self.act_pan = QtWidgets.QAction("Pan", self)
        self.act_pan.setCheckable(True)
        self.act_pan.setToolTip("Left-drag grabs and moves the view")
        for act, mode in ((self.act_rotate, "rotate"), (self.act_pan, "pan")):
            mode_group.addAction(act)
            tb.addAction(act)
            act.triggered.connect(lambda _c=False, m=mode: self.render.set_interaction(m))
        tb.addSeparator()

        for ax in "xyz":
            tb.addAction(f"View {ax.upper()}",
                         lambda _c=False, a=ax: self.render.align_axis(a))
        tb.addAction("Reset", lambda: self.render.reset_view())
        tb.addSeparator()

        # spin animation
        self.spin_act = QtWidgets.QAction("Spin", self)
        self.spin_act.setCheckable(True)
        self.spin_act.setToolTip("Auto-rotate the camera")
        self.spin_act.toggled.connect(self._toggle_rotation)
        tb.addAction(self.spin_act)
        self.rot_speed = DoubleSpinBox()
        self.rot_speed.setRange(0.1, 10.0)
        self.rot_speed.setSingleStep(0.1)
        self.rot_speed.setValue(self.settings.rotation_azimuth)
        self.rot_speed.setToolTip("Spin speed (°/frame)")
        self.rot_speed.setFixedWidth(70)
        tb.addWidget(self.rot_speed)
        tb.addSeparator()

        self.axes_act = QtWidgets.QAction("Axes", self)
        self.axes_act.setCheckable(True)
        self.axes_act.setChecked(True)
        self.axes_act.setToolTip("Show orientation axes")
        self.axes_act.toggled.connect(lambda on: self.render.show_axes(on))
        tb.addAction(self.axes_act)

    # ------------------------------------------------------------------
    # Global (geometry + view) controls
    # ------------------------------------------------------------------
    def _build_global_controls(self) -> None:
        res = self.settings.grid_resolution - 1
        geo = self._schedule_geometry

        # Atoms first: the most-used controls sit at the top of the panel.
        atoms_header = QtWidgets.QLabel("Atoms")
        atoms_header.setStyleSheet("font-weight: bold; margin-top: 6px;")
        self.control_layout.addWidget(atoms_header)
        self.atom_container = QtWidgets.QVBoxLayout()
        self.control_layout.addLayout(self.atom_container)

        region = CollapsibleBox("Region & Slicing")
        self._build_grid_control(region.content_layout)
        self.roi: Dict[str, object] = {}
        for name, init in (("xmin", 0), ("xmax", res), ("ymin", 0),
                           ("ymax", res), ("zmin", 0), ("zmax", res)):
            self.roi[name] = labelled_slider(
                _ROI_LABELS[name], 0, res, init, region.content_layout, None)
        self._link_roi("xmin", "xmax")
        self._link_roi("ymin", "ymax")
        self._link_roi("zmin", "zmax")
        for name, c in self.roi.items():
            c.widget.valueChanged.connect(geo)
            c.widget.valueChanged.connect(lambda _v, n=name: self._update_roi_label(n))
        self._build_miller_controls(region.content_layout)

        # smoothing order lives with the ROI it interacts with
        srow = QtWidgets.QWidget()
        sh = QtWidgets.QHBoxLayout(srow)
        sh.setContentsMargins(0, 0, 0, 0)
        sh.addWidget(QtWidgets.QLabel("Smooth:"))
        self.smooth_before = QtWidgets.QRadioButton("before slice")
        self.smooth_after = QtWidgets.QRadioButton("after slice")
        self.smooth_before.setChecked(True)
        self.smooth_before.setToolTip("Apply Gaussian smoothing before ROI/Miller slicing")
        self.smooth_after.setToolTip("Apply Gaussian smoothing after ROI/Miller slicing")
        self.smooth_before.toggled.connect(geo)
        sh.addWidget(self.smooth_before)
        sh.addWidget(self.smooth_after)
        sh.addStretch()
        region.add(srow)
        self.control_layout.addWidget(region)

        self.control_layout.addWidget(self._build_render_box())
        self.control_layout.addStretch()

    def _build_grid_control(self, layout) -> None:
        row = QtWidgets.QWidget()
        h = QtWidgets.QHBoxLayout(row)
        h.setContentsMargins(0, 0, 0, 0)
        h.addWidget(QtWidgets.QLabel("Histogram grid (bins/axis):"))
        h.addStretch()
        self.grid_spin = SpinBox()
        self.grid_spin.setRange(32, 512)
        self.grid_spin.setSingleStep(10)
        self.grid_spin.setValue(self.settings.grid_resolution)
        self.grid_spin.setToolTip(
            "Number of histogram bins along each axis (N³ voxels). "
            "Higher = finer detail but slower. Rebuilds histograms.")
        self.grid_spin.editingFinished.connect(self._change_grid)
        h.addWidget(self.grid_spin)
        layout.addWidget(row)

    def _link_roi(self, lo: str, hi: str) -> None:
        lo_s, hi_s = self.roi[lo].widget, self.roi[hi].widget
        lo_s.valueChanged.connect(lambda v: v > hi_s.value() and hi_s.setValue(v))
        hi_s.valueChanged.connect(lambda v: v < lo_s.value() and lo_s.setValue(v))

    def _build_miller_controls(self, layout) -> None:
        s = self.settings
        geo = self._schedule_geometry
        self.miller_on = QtWidgets.QCheckBox("Enable Miller Slicing")
        self.miller_on.toggled.connect(self._toggle_miller)
        layout.addWidget(self.miller_on)

        self.miller_body = QtWidgets.QWidget()
        body = QtWidgets.QVBoxLayout(self.miller_body)
        body.setContentsMargins(0, 0, 0, 0)
        ir = s.miller_index_range
        self.mh = labelled_spinbox("Miller h", ir[0], ir[1], s.miller_index_default, body, geo)
        self.mk = labelled_spinbox("Miller k", ir[0], ir[1], s.miller_index_default, body, geo)
        self.ml = labelled_spinbox("Miller l", ir[0], ir[1], s.miller_index_default, body, geo)
        tr = s.miller_thickness_range
        self.mthick = labelled_spinbox("Thickness (Å)", tr[0], tr[1],
                                       s.miller_thickness_default, body, geo,
                                       double=True, step=0.1)
        orr = s.miller_offset_range
        self.moff = labelled_spinbox("Offset (Å)", orr[0], orr[1],
                                     s.miller_offset_default, body, geo,
                                     double=True, step=0.1)
        view_btn = QtWidgets.QPushButton("View Miller Plane")
        view_btn.clicked.connect(self._view_miller)
        body.addWidget(view_btn)
        self.miller_body.setVisible(False)
        layout.addWidget(self.miller_body)

    def _build_render_box(self) -> CollapsibleBox:
        box = CollapsibleBox("Render", expanded=False)
        # interpolation: smooth (linear) vs sharp (nearest)
        row = QtWidgets.QWidget()
        h = QtWidgets.QHBoxLayout(row)
        h.setContentsMargins(0, 0, 0, 0)
        h.addWidget(QtWidgets.QLabel("Edges:"))
        self.interp_smooth = QtWidgets.QRadioButton("Smooth")
        self.interp_sharp = QtWidgets.QRadioButton("Sharp")
        self.interp_smooth.setChecked(self.settings.interpolation != "nearest")
        self.interp_sharp.setChecked(self.settings.interpolation == "nearest")
        self.interp_sharp.toggled.connect(self._set_interpolation)
        h.addWidget(self.interp_smooth)
        h.addWidget(self.interp_sharp)
        h.addStretch()
        box.add(row)

        # volume mapper: GPU (fast) vs CPU (exact silhouettes)
        mrow = QtWidgets.QWidget()
        mh = QtWidgets.QHBoxLayout(mrow)
        mh.setContentsMargins(0, 0, 0, 0)
        mh.addWidget(QtWidgets.QLabel("Volumes:"))
        self.map_fast = QtWidgets.QRadioButton("Fast (GPU)")
        self.map_exact = QtWidgets.QRadioButton("Exact (CPU)")
        exact = self.settings.volume_mapper == "fixed_point"
        self.map_fast.setChecked(not exact)
        self.map_exact.setChecked(exact)
        self.map_fast.setToolTip("GPU ray casting. Fast, but oblique edges and "
                                 "corners keep a slightly soft, semi-transparent rim.")
        self.map_exact.setToolTip("CPU ray casting. Pixel-exact fully opaque "
                                  "edges; slow on large grids. Good for screenshots.")
        self.map_exact.toggled.connect(self._set_mapper)
        mh.addWidget(self.map_fast)
        mh.addWidget(self.map_exact)
        mh.addStretch()
        box.add(mrow)

        # sampling quality: smaller step = crisper (and slower)
        srow = QtWidgets.QWidget()
        sh = QtWidgets.QHBoxLayout(srow)
        sh.setContentsMargins(0, 0, 0, 0)
        sh.addWidget(QtWidgets.QLabel("Ray step (smaller = sharper):"))
        sh.addStretch()
        self.sample_spin = DoubleSpinBox()
        self.sample_spin.setRange(0.1, 2.0)
        self.sample_spin.setSingleStep(0.1)
        self.sample_spin.setValue(self.settings.volume_sample_factor)
        self.sample_spin.setToolTip("Ray step as a fraction of voxel size. "
                                    "0.5 = 2× oversampled (crisp).")
        self.sample_spin.valueChanged.connect(
            lambda v: self.render.set_sample_factor(v))
        sh.addWidget(self.sample_spin)
        box.add(srow)

        # projection
        prow = QtWidgets.QWidget()
        ph = QtWidgets.QHBoxLayout(prow)
        ph.setContentsMargins(0, 0, 0, 0)
        ph.addWidget(QtWidgets.QLabel("Projection:"))
        self.proj_persp = QtWidgets.QRadioButton("Perspective")
        self.proj_ortho = QtWidgets.QRadioButton("Orthographic")
        ortho = self.settings.projection == "orthographic"
        self.proj_persp.setChecked(not ortho)
        self.proj_ortho.setChecked(ortho)
        self.proj_ortho.toggled.connect(self._set_projection)
        ph.addWidget(self.proj_persp)
        ph.addWidget(self.proj_ortho)
        ph.addStretch()
        box.add(prow)

        # background colour
        brow = QtWidgets.QWidget()
        bh = QtWidgets.QHBoxLayout(brow)
        bh.setContentsMargins(0, 0, 0, 0)
        bh.addWidget(QtWidgets.QLabel("Background:"))
        bh.addStretch()
        self.bg_btn = QtWidgets.QPushButton()
        self.bg_btn.setFixedSize(40, 20)
        self._set_bg_chip(self.settings.background_color)
        self.bg_btn.clicked.connect(self._pick_background)
        bh.addWidget(self.bg_btn)
        box.add(brow)

        # depth peeling
        self.depth_peel = QtWidgets.QCheckBox("Depth peeling")
        self.depth_peel.setChecked(self.settings.depth_peeling)
        self.depth_peel.setToolTip("Per-fragment compositing for overlapping "
                                   "translucent volumes and isosurface shells. "
                                   "Off = faster.")
        self.depth_peel.toggled.connect(self._set_depth_peeling)
        box.add(self.depth_peel)

        # interactive LOD: drop quality only while the camera is moving
        self.lod_check = QtWidgets.QCheckBox("Fast interaction (coarse while moving)")
        self.lod_check.setChecked(self.settings.interactive_lod)
        self.lod_check.setToolTip("While dragging the camera, volumes use coarse "
                                  "ray sampling and depth peeling is paused; full "
                                  "quality snaps back on release.")
        self.lod_check.toggled.connect(self._set_lod)
        box.add(self.lod_check)

        # stereo (a render property, so it lives here)
        strow = QtWidgets.QWidget()
        sth = QtWidgets.QHBoxLayout(strow)
        sth.setContentsMargins(0, 0, 0, 0)
        self.stereo_on = QtWidgets.QCheckBox("Stereo 3D")
        self.stereo_mode = ComboBox()
        self.stereo_mode.addItems(STEREO_MODES)
        self.stereo_on.toggled.connect(self._apply_stereo)
        self.stereo_mode.currentIndexChanged.connect(self._apply_stereo)
        sth.addWidget(self.stereo_on)
        sth.addWidget(self.stereo_mode)
        sth.addStretch()
        box.add(strow)
        return box

    def _set_interpolation(self, sharp: bool) -> None:
        self.settings.interpolation = "nearest" if sharp else "linear"
        self.render.set_interpolation(self.settings.interpolation)

    def _set_mapper(self, exact: bool) -> None:
        self.settings.volume_mapper = "fixed_point" if exact else "smart"
        self.render.set_mapper(self.settings.volume_mapper)

    def _set_projection(self, ortho: bool) -> None:
        self.settings.projection = "orthographic" if ortho else "perspective"
        self.render.set_projection(ortho)

    def _set_bg_chip(self, color: str) -> None:
        self.bg_btn.setStyleSheet(
            f"background-color: {color}; border: 1px solid #888;")
        self.bg_btn.setProperty("color", color)

    def _pick_background(self) -> None:
        from PyQt5 import QtGui
        cur = QtGui.QColor(self.bg_btn.property("color"))
        col = QtWidgets.QColorDialog.getColor(initial=cur, parent=self)
        if col.isValid():
            self._set_bg_chip(col.name())
            self.settings.background_color = col.name()
            self.render.set_background(col.name())

    def _set_depth_peeling(self, on: bool) -> None:
        self.settings.depth_peeling = on
        self.render.set_depth_peeling(on)

    def _set_lod(self, on: bool) -> None:
        self.settings.interactive_lod = on

    def _build_timers(self) -> None:
        self.geo_timer = QtCore.QTimer(self)
        self.geo_timer.setSingleShot(True)
        self.geo_timer.setInterval(self.settings.update_delay_ms)
        self.geo_timer.timeout.connect(self._rebuild)

        self.rot_timer = QtCore.QTimer(self)
        self.rot_timer.setInterval(int(1000 / self.settings.rotation_fps))
        self.rot_timer.timeout.connect(
            lambda: self.render.rotate(self.rot_speed.value()))

        # Appearance retint is debounced so a slider drag coalesces into one
        # rebuild instead of one per tick.
        self._pending_appear = set()
        self.appear_timer = QtCore.QTimer(self)
        self.appear_timer.setSingleShot(True)
        self.appear_timer.setInterval(60)
        self.appear_timer.timeout.connect(self._flush_appearance)

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------
    def _open_dialog(self) -> None:
        dlg = OpenDialog(self)
        if dlg.exec_() == QtWidgets.QDialog.Accepted and dlg.path.text().strip():
            self._load(dlg.spec())

    def _settings_dialog(self) -> None:
        dlg = SettingsDialog(self.settings, self)
        if dlg.exec_() == QtWidgets.QDialog.Accepted:
            dlg.apply_to(self.settings)
            self.geo_timer.setInterval(self.settings.update_delay_ms)
            self.rot_timer.setInterval(int(1000 / self.settings.rotation_fps))

    def _load(self, spec: LoadSpec) -> None:
        try:
            QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
            self.model.load(spec)
        except Exception as exc:  # surface load errors instead of crashing
            QtWidgets.QApplication.restoreOverrideCursor()
            QtWidgets.QMessageBox.critical(self, "Load failed", str(exc))
            return
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()

        self.render.clear_all()
        for panel in self.panels.values():
            panel.setParent(None)
        self.panels.clear()

        self._reset_roi_ranges()

        for atype in self.model.species:
            panel = AtomPanel(atype, self.settings)
            panel.geometryChanged.connect(lambda a: self._schedule_geometry())
            panel.appearanceChanged.connect(self._retint)
            self.atom_container.addWidget(panel)
            self.panels[atype] = panel

        self._rebuild()

    def _reset_roi_ranges(self) -> None:
        """Set ROI sliders to span the full (current) grid resolution."""
        res = self.settings.grid_resolution - 1
        for name, is_max in (("xmin", False), ("xmax", True), ("ymin", False),
                             ("ymax", True), ("zmin", False), ("zmax", True)):
            c = self.roi[name]
            c.widget.blockSignals(True)
            c.widget.setRange(0, res)
            c.widget.setValue(res if is_max else 0)
            c.widget.blockSignals(False)
            self._update_roi_label(name)

    def _update_roi_label(self, name: str) -> None:
        """Show the ROI boundary in Ångström (mapped through the cell)."""
        c = self.roi[name]
        if c.label is None:
            return
        axis = {"x": 0, "y": 1, "z": 2}[name[0]]
        phys = self.model.origin[axis] + c.widget.value() * self.model.spacing[axis]
        c.label.setText(f"{_ROI_LABELS[name]}: {phys:.2f} Å")

    def _change_grid(self) -> None:
        """Rebuild histograms at the newly chosen grid resolution."""
        res = self.grid_spin.value()
        if not self.model.species or res == self.settings.grid_resolution:
            return
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        try:
            self.model.rebuild_grid(res)
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
        self._reset_roi_ranges()
        self._rebuild()

    # ------------------------------------------------------------------
    # Two-tier updates
    # ------------------------------------------------------------------
    def _schedule_geometry(self, *_) -> None:
        self.geo_timer.start()

    def _roi_indices(self) -> Dict[str, int]:
        return {k: self.roi[k].widget.value() for k in
                ("xmin", "xmax", "ymin", "ymax", "zmin", "zmax")}

    def _miller(self) -> MillerParams:
        return MillerParams(
            use=self.miller_on.isChecked(),
            h=self.mh.widget.value(), k=self.mk.widget.value(), l=self.ml.widget.value(),
            thickness=self.mthick.widget.value(), offset=self.moff.widget.value(),
        )

    def _rebuild(self) -> None:
        if not self.model.species:
            return
        # Miller Plane Slice needs an active plane: auto-enable if any atom uses it.
        self._ensure_miller_for_plane_modes()

        first = self.render.first_build
        region = self.model.region(self._roi_indices(), self._miller())
        smooth_before = self.smooth_before.isChecked()

        self.render.clear_all()
        for atype, panel in self.panels.items():
            mode = panel.current_mode()
            app = self._app_for(atype)
            sigma = panel.sigma_value()
            if mode == "Hidden":
                continue
            if mode == "Histogram":
                vol = self.model.volume_data(atype, sigma, region, smooth_before)
                if vol is not None:
                    self.render.show_volume(atype, vol, app)
            elif mode == "Isosurface":
                vol = self.model.volume_data(atype, sigma, region, smooth_before)
                if vol is not None:
                    self.render.show_isosurface(atype, vol, app)
            elif mode == "Averaged Positions":
                self._show_averages(atype, region, app)
            elif mode == "Miller Plane Slice":
                self._show_plane(atype, region, sigma, smooth_before, app)

        self.render.set_focus(region.focal_point, reset=first)

    def _app_for(self, atype: str):
        return self.panels[atype].appearance()

    def _ensure_miller_for_plane_modes(self) -> None:
        wants_plane = any(p.current_mode() == "Miller Plane Slice"
                          for p in self.panels.values())
        if wants_plane and not self.miller_on.isChecked():
            self.miller_on.blockSignals(True)
            self.miller_on.setChecked(True)
            self.miller_on.blockSignals(False)
            self.miller_body.setVisible(True)

    def _retint(self, atype: str) -> None:
        if atype in self.panels:
            self._pending_appear.add(atype)
            self.appear_timer.start()

    def _flush_appearance(self) -> None:
        pending, self._pending_appear = self._pending_appear, set()
        for atype in pending:
            # Appearance change on a mode whose actor doesn't exist yet -> rebuild.
            if not self.render.update_appearance(atype, self._app_for(atype)):
                self._schedule_geometry()

    # ------------------------------------------------------------------
    # Mode builders
    # ------------------------------------------------------------------
    def _show_averages(self, atype, region, app) -> None:
        method = self.panels[atype].average_method()
        pts = self.model.atom_data[atype]["individual_averages"][method]
        if pts.size == 0:
            return
        lo, hi = region.phys_min, region.phys_max
        in_roi = np.all((pts >= lo) & (pts <= hi), axis=1)
        pts = pts[in_roi]
        from ..miller import filter_points
        pts = filter_points(pts, self.model.cell_center, region.miller)
        self.render.show_glyphs(atype, pts, app)

    def _show_plane(self, atype, region, sigma, smooth_before, app) -> None:
        miller = region.miller
        if miller.normal is None:
            return
        center = self.model.cell_center + miller.offset * miller.normal
        size = float(np.linalg.norm(region.phys_max - region.phys_min)) or 1.0
        res = int(self.settings.grid_resolution * self.settings.miller_plane_res_factor)
        mesh = pv.Plane(center=center, direction=miller.normal,
                        i_size=size, j_size=size, i_resolution=res, j_resolution=res)
        values = self.model.sample_on_plane(atype, sigma, mesh.points, smooth_before)
        if values.size == 0:
            return
        self.render.show_plane(atype, mesh, values, app)

    # ------------------------------------------------------------------
    # View helpers
    # ------------------------------------------------------------------
    def _toggle_miller(self, on: bool) -> None:
        self.miller_body.setVisible(on)
        self._schedule_geometry()

    def _view_miller(self) -> None:
        m = self._miller()
        m = MillerParams(True, m.h, m.k, m.l, m.thickness, m.offset)
        self.render.align_miller(m, self.model.cell_center)

    def _toggle_rotation(self, on: bool) -> None:
        if on:
            self.rot_timer.start()
        else:
            self.rot_timer.stop()

    def _apply_stereo(self, *_) -> None:
        self.render.set_stereo(self.stereo_on.isChecked(),
                               self.stereo_mode.currentText())

    # ------------------------------------------------------------------
    def closeEvent(self, event) -> None:
        self.geo_timer.stop()
        self.rot_timer.stop()
        try:
            self.plotter.close()
        except Exception:
            pass
        super().closeEvent(event)
