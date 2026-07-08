"""Per-atom settings panel.

Compact collapsible design: the always-visible header row carries the fast
"what plots how" overview -- ``[arrow + name] [colour chip] ... [mode]`` --
and the body below shows only the controls relevant to the current mode.
Selecting *Hidden* auto-collapses the body.

Emits two kinds of change so the window can route them:
  * geometryChanged   -> mode / sphere size  (needs recompute or re-glyph)
  * appearanceChanged -> colour / opacity / threshold / gamma / cmap / norm
"""
from __future__ import annotations

import random

from PyQt5 import QtCore, QtGui, QtWidgets

from ..config import Settings
from ..render import Appearance
from .widgets import ComboBox, labelled_slider, labelled_spinbox

MODES = ["Hidden", "Histogram", "Isosurface", "Averaged Positions",
         "Miller Plane Slice"]


class AtomPanel(QtWidgets.QFrame):
    geometryChanged = QtCore.pyqtSignal(str)
    appearanceChanged = QtCore.pyqtSignal(str)

    def __init__(self, atype: str, settings: Settings, parent=None):
        super().__init__(parent)
        self.atype = atype
        self.settings = settings
        self.setFrameShape(QtWidgets.QFrame.StyledPanel)

        geo = lambda *_: self.geometryChanged.emit(atype)  # noqa: E731
        app = lambda *_: self.appearanceChanged.emit(atype)  # noqa: E731

        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(4, 2, 4, 4)
        outer.setSpacing(2)

        # -- header: [arrow + name] [colour chip] ......... [mode] --------
        header = QtWidgets.QHBoxLayout()
        self.toggle = QtWidgets.QToolButton()
        self.toggle.setStyleSheet("QToolButton { border: none; font-weight: bold; }")
        self.toggle.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
        self.toggle.setText(atype)
        self.toggle.setCheckable(True)
        self.toggle.setChecked(True)
        self.toggle.setArrowType(QtCore.Qt.DownArrow)
        self.toggle.clicked.connect(self._on_toggle)
        header.addWidget(self.toggle)

        init_color = "#%06x" % random.randint(0, 0xFFFFFF)
        self.color_btn = QtWidgets.QPushButton()
        self.color_btn.setFixedSize(18, 18)
        self.color_btn.setToolTip("Sphere colour (Averaged Positions mode)")
        self._set_chip(init_color)
        self.color_btn.clicked.connect(self._pick_color)
        header.addWidget(self.color_btn)
        header.addStretch()

        self.mode = ComboBox()
        self.mode.addItems(MODES)
        self.mode.currentIndexChanged.connect(self._sync_visibility)
        self.mode.currentIndexChanged.connect(geo)
        header.addWidget(self.mode)
        outer.addLayout(header)

        # -- body: mode-specific controls only -----------------------------
        self.body = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(self.body)
        lay.setContentsMargins(12, 0, 0, 0)
        lay.setSpacing(2)
        outer.addWidget(self.body)

        crow = QtWidgets.QWidget()
        ch = QtWidgets.QHBoxLayout(crow)
        ch.setContentsMargins(0, 0, 0, 0)
        ch.addWidget(QtWidgets.QLabel("Colormap:"))
        self.cmap = ComboBox()
        self.cmap.addItems(settings.colormaps)
        self.cmap.setCurrentText(settings.colormaps[0])
        self.cmap.currentIndexChanged.connect(app)
        ch.addWidget(self.cmap, stretch=1)
        lay.addWidget(crow)
        self.cmap_row = crow

        self.sigma = labelled_slider("Smoothing σ", 0, 20,
                                     settings.gaussian_sigma, lay, geo)

        dr = settings.density_range
        self.density_lower = labelled_slider("Density Lower", dr[0], dr[1],
                                             settings.density_lower_default, lay, app)
        self.density_upper = labelled_slider("Density Upper", dr[0], dr[1],
                                             settings.density_upper_default, lay, app)
        self.opacity = labelled_slider("Opacity (%)", 0, 100,
                                       settings.opacity_default, lay, app)
        sr = settings.sphere_size_range
        self.sphere = labelled_slider("Sphere Size", sr[0], sr[1],
                                      settings.sphere_size_default, lay, geo)

        # position statistic for Averaged Positions mode
        prow = QtWidgets.QWidget()
        ph = QtWidgets.QHBoxLayout(prow)
        ph.setContentsMargins(0, 0, 0, 0)
        ph.addWidget(QtWidgets.QLabel("Position:"))
        self.avg_method = ComboBox()
        self._avg_methods = [("Mode (most visited)", "mode"),
                             ("Circular mean", "mean"),
                             ("Naive mean", "naive")]
        self.avg_method.addItems([label for label, _ in self._avg_methods])
        keys = [m for _, m in self._avg_methods]
        if settings.average_method in keys:
            self.avg_method.setCurrentIndex(keys.index(settings.average_method))
        self.avg_method.setToolTip(
            "Mode: the site the atom occupies most often (two-site hoppers "
            "show at their dominant site). Circular mean: periodic-aware "
            "average. Naive mean: raw average of stored coordinates.")
        self.avg_method.currentIndexChanged.connect(geo)
        ph.addWidget(self.avg_method, stretch=1)
        lay.addWidget(prow)
        self.avg_method_row = prow

        gr = settings.gamma_range
        self.gamma = labelled_spinbox("Opacity Gamma", gr[0], gr[1],
                                      settings.gamma_default, lay, app,
                                      double=True, step=0.1)

        # Isosurface-only controls
        self.iso_shells = labelled_spinbox("Iso Shells", 1, 10,
                                           settings.iso_shells_default, lay, app)
        self.iso_shells.widget.setToolTip(
            "Number of nested contour shells across the density window.")
        self.iso_tol = labelled_spinbox("Band Tolerance", 0.0, 0.45,
                                        settings.iso_tolerance_default, lay, app,
                                        double=True, step=0.05)
        self.iso_tol.widget.setToolTip(
            "Fraction of the density window trimmed off each end, so the "
            "outer/inner shells sit clear of the fuzzy extremes.")
        self.iso_quality = labelled_slider("Surface Quality (%)", 1, 100,
                                           settings.iso_quality_default, lay, app)
        self.iso_quality.widget.setToolTip(
            "Mesh resolution quality. 100% is raw marching cubes. Lower values "
            "decimate the mesh to improve rendering performance.")

        # Separate from opacity: only affects how colour is spread across the
        # density range. Its own line, on by default.
        self.norm = QtWidgets.QCheckBox("Normalise colour in range")
        self.norm.setToolTip("Spread colours across the density-lower..upper "
                             "window instead of the full data range.")
        self.norm.setChecked(True)
        self.norm.toggled.connect(app)
        lay.addWidget(self.norm)

        self._sync_visibility()

    # -- state -----------------------------------------------------------
    def current_mode(self) -> str:
        return self.mode.currentText()

    def sigma_value(self) -> int:
        return self.sigma.widget.value()

    def average_method(self) -> str:
        return self._avg_methods[self.avg_method.currentIndex()][1]

    def appearance(self) -> Appearance:
        return Appearance(
            cmap=self.cmap.currentText(),
            density_lower=self.density_lower.widget.value(),
            density_upper=self.density_upper.widget.value(),
            opacity=self.opacity.widget.value(),
            gamma=self.gamma.widget.value(),
            normalize_in_range=self.norm.isChecked(),
            color=self.color_btn.property("color"),
            sphere_size=self.sphere.widget.value(),
            iso_shells=self.iso_shells.widget.value(),
            iso_tolerance=self.iso_tol.widget.value(),
            iso_quality=self.iso_quality.widget.value(),
        )

    # -- internal --------------------------------------------------------
    def _set_chip(self, color: str) -> None:
        self.color_btn.setStyleSheet(
            f"background-color: {color}; border: 1px solid #888; "
            "border-radius: 3px;")
        self.color_btn.setProperty("color", color)

    def _pick_color(self):
        cur = QtGui.QColor(self.color_btn.property("color"))
        col = QtWidgets.QColorDialog.getColor(initial=cur, parent=self)
        if col.isValid():
            self._set_chip(col.name())
            self.appearanceChanged.emit(self.atype)

    def _on_toggle(self, checked: bool) -> None:
        self._set_open(checked)

    def _set_open(self, open_: bool) -> None:
        self.toggle.setChecked(open_)
        self.toggle.setArrowType(
            QtCore.Qt.DownArrow if open_ else QtCore.Qt.RightArrow)
        self.body.setVisible(open_)

    def _sync_visibility(self):
        mode = self.current_mode()
        density = mode in ("Histogram", "Isosurface", "Miller Plane Slice")
        iso = mode == "Isosurface"
        average = mode == "Averaged Positions"
        self.cmap_row.setVisible(density)
        self.sigma.container.setVisible(density)
        for c in (self.density_lower, self.density_upper, self.opacity, self.gamma):
            c.container.setVisible(density)
        self.norm.setVisible(density)
        for c in (self.iso_shells, self.iso_tol, self.iso_quality):
            c.container.setVisible(iso)
        self.sphere.container.setVisible(average)
        self.avg_method_row.setVisible(average)
        self.color_btn.setVisible(True)
        self._set_open(mode != "Hidden")
