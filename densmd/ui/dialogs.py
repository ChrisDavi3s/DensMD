"""Open-file and Settings dialogs.

``OpenDialog`` collects a ``LoadSpec``: browse to a file, pick/auto-detect its
format and frame slice, then "Scan species" reads just the first frame so the
user can remap element symbols before the real (slow) load happens.
``SettingsDialog`` edits a live ``Settings`` instance in place and saves it to
disk on accept.
"""
from __future__ import annotations

from typing import Dict, Optional

from PyQt5 import QtCore, QtWidgets

from ..config import Settings
from ..io import LoadSpec, guess_format
from ..model import DensityModel
from .widgets import ComboBox, SpinBox


def _parse_type_map(text: str) -> Optional[Dict[str, str]]:
    """Parse ``"H:Li, He:P"`` into a mapping (used by the CLI shim). Empty -> None."""
    text = text.strip()
    if not text:
        return None
    mapping: Dict[str, str] = {}
    for pair in text.replace("\n", ",").split(","):
        if ":" not in pair:
            continue
        src, dst = pair.split(":", 1)
        if src.strip():
            mapping[src.strip()] = dst.strip()
    return mapping or None


class OpenDialog(QtWidgets.QDialog):
    """Collect a :class:`LoadSpec`; remap atom types via a scanned table."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Open Trajectory")
        self.setMinimumWidth(540)
        outer = QtWidgets.QVBoxLayout(self)
        form = QtWidgets.QFormLayout()
        form.setFieldGrowthPolicy(QtWidgets.QFormLayout.AllNonFixedFieldsGrow)
        form.setLabelAlignment(QtCore.Qt.AlignRight)
        outer.addLayout(form)

        path_row = QtWidgets.QHBoxLayout()
        self.path = QtWidgets.QLineEdit()
        browse = QtWidgets.QPushButton("Browse…")
        browse.setFixedWidth(90)
        browse.clicked.connect(self._browse)
        path_row.addWidget(self.path)
        path_row.addWidget(browse)
        form.addRow("File:", path_row)

        self.fmt = ComboBox()
        self.fmt.addItems(["Auto", "ASE", "Pickle"])
        form.addRow("Format:", self.fmt)

        self.slice = QtWidgets.QLineEdit("::5")
        self.slice.setToolTip("ASE frame slice, e.g. ::5 for every 5th frame")
        form.addRow("Frame slice:", self.slice)
        hint = QtWidgets.QLabel("Sub-sample frames, e.g. \"::5\" = every 5th, "
                                "\":\" = all, \"0:100\" = first 100.")
        hint.setStyleSheet("color: gray; font-size: 11px;")
        form.addRow("", hint)

        scan_row = QtWidgets.QHBoxLayout()
        scan_btn = QtWidgets.QPushButton("Scan species")
        scan_btn.clicked.connect(self._scan)
        scan_row.addWidget(scan_btn)
        scan_row.addWidget(QtWidgets.QLabel("edit 'Show as' to remap types"))
        scan_row.addStretch()
        outer.addLayout(scan_row)

        self.table = QtWidgets.QTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels(["In file", "Show as"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.verticalHeader().setVisible(False)
        self.table.setMaximumHeight(180)
        outer.addWidget(self.table)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Open | QtWidgets.QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        outer.addWidget(buttons)

    # -- helpers ---------------------------------------------------------
    def _resolved_fmt(self) -> str:
        fmt = self.fmt.currentText()
        return guess_format(self.path.text()) if fmt == "Auto" else fmt.lower()

    def _browse(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open Trajectory")
        if path:
            self.path.setText(path)
            if self.fmt.currentText() == "Auto":
                self.fmt.setCurrentText("Pickle" if guess_format(path) == "pickle"
                                        else "ASE")

    def _scan(self):
        if not self.path.text().strip():
            return
        base = LoadSpec(self.path.text().strip(), self._resolved_fmt(),
                        self.slice.text().strip() or ":")
        try:
            QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
            species = DensityModel.scan_species(base)
        except Exception as exc:
            QtWidgets.QApplication.restoreOverrideCursor()
            QtWidgets.QMessageBox.warning(self, "Scan failed", str(exc))
            return
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()

        self.table.setRowCount(len(species))
        for r, sym in enumerate(species):
            item = QtWidgets.QTableWidgetItem(sym)
            item.setFlags(item.flags() & ~QtCore.Qt.ItemIsEditable)
            self.table.setItem(r, 0, item)
            self.table.setItem(r, 1, QtWidgets.QTableWidgetItem(sym))

    def _type_map(self) -> Optional[Dict[str, str]]:
        mapping: Dict[str, str] = {}
        for r in range(self.table.rowCount()):
            src = self.table.item(r, 0).text().strip()
            dst_item = self.table.item(r, 1)
            dst = dst_item.text().strip() if dst_item else src
            if dst and dst != src:
                mapping[src] = dst
        return mapping or None

    def spec(self) -> LoadSpec:
        return LoadSpec(
            path=self.path.text().strip(),
            fmt=self._resolved_fmt(),
            frame_slice=self.slice.text().strip() or ":",
            atom_type_map=self._type_map(),
        )


class SettingsDialog(QtWidgets.QDialog):
    """Edit the runtime settings that used to be module constants."""

    def __init__(self, settings: Settings, parent=None):
        super().__init__(parent)
        self.settings = settings
        self.setWindowTitle("Settings")
        form = QtWidgets.QFormLayout(self)

        self.delay = SpinBox()
        self.delay.setRange(0, 1000)
        self.delay.setValue(settings.update_delay_ms)
        form.addRow("Update delay (ms):", self.delay)

        self.fps = SpinBox()
        self.fps.setRange(1, 60)
        self.fps.setValue(settings.rotation_fps)
        form.addRow("Rotation FPS:", self.fps)

        self.plane_res = SpinBox()
        self.plane_res.setRange(1, 16)
        self.plane_res.setValue(settings.miller_plane_res_factor)
        form.addRow("Miller plane res factor:", self.plane_res)

        self.plane_order = SpinBox()
        self.plane_order.setRange(0, 5)
        self.plane_order.setValue(settings.miller_sample_order)
        form.addRow("Miller sample spline order:", self.plane_order)

        self.avg_method = ComboBox()
        self._methods = [("Most-visited site (mode)", "mode"),
                         ("Circular mean", "mean"),
                         ("Naive mean", "naive")]
        self.avg_method.addItems([label for label, _ in self._methods])
        current = [m for _, m in self._methods].index(settings.average_method) \
            if settings.average_method in ("mode", "mean", "naive") else 0
        self.avg_method.setCurrentIndex(current)
        self.avg_method.setToolTip(
            "Mode: the site each atom occupies most often -- a two-site "
            "hopper shows at its dominant site, never in the gap between. "
            "Circular mean: periodic-aware average. Naive mean: raw average "
            "of stored coordinates (boundary hoppers land mid-cell). "
            "Each atom panel can override this per species.")
        form.addRow("Position statistic (default):", self.avg_method)

        self.subsample = SpinBox()
        self.subsample.setRange(1, 1000)
        self.subsample.setValue(settings.average_subsample)
        self.subsample.setToolTip("Use every Nth frame when averaging (speed).")
        form.addRow("Average subsample (reload):", self.subsample)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)

    def apply_to(self, settings: Settings) -> None:
        settings.update_delay_ms = self.delay.value()
        settings.rotation_fps = self.fps.value()
        settings.miller_plane_res_factor = self.plane_res.value()
        settings.miller_sample_order = self.plane_order.value()
        settings.average_method = self._methods[self.avg_method.currentIndex()][1]
        settings.average_subsample = self.subsample.value()
        settings.save()
