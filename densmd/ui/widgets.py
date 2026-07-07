"""Small reusable Qt control factories.

One consistent labelled-slider / spinbox helper instead of three near-copies.
Each returns a lightweight ``Control`` holding the widget and its live label.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

from PyQt5 import QtCore, QtWidgets


@dataclass
class Control:
    container: QtWidgets.QWidget
    widget: QtWidgets.QWidget
    label: Optional[QtWidgets.QLabel] = None

    def value(self):
        w = self.widget
        return w.value() if hasattr(w, "value") else None


def labelled_slider(text: str, lo: int, hi: int, init: int,
                    layout: QtWidgets.QLayout,
                    on_change: Optional[Callable] = None) -> Control:
    box = QtWidgets.QWidget()
    v = QtWidgets.QVBoxLayout(box)
    v.setContentsMargins(0, 2, 0, 2)
    label = QtWidgets.QLabel(f"{text}: {init}")
    slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
    slider.setRange(lo, hi)
    slider.setValue(init)
    slider.valueChanged.connect(lambda val: label.setText(f"{text}: {val}"))
    if on_change:
        slider.valueChanged.connect(on_change)
    v.addWidget(label)
    v.addWidget(slider)
    layout.addWidget(box)
    return Control(box, slider, label)


def labelled_spinbox(text: str, lo: int, hi: int, init: int,
                     layout: QtWidgets.QLayout,
                     on_change: Optional[Callable] = None,
                     double: bool = False, step: float = 1.0) -> Control:
    box = QtWidgets.QWidget()
    h = QtWidgets.QHBoxLayout(box)
    h.setContentsMargins(0, 0, 0, 0)
    spin = QtWidgets.QDoubleSpinBox() if double else QtWidgets.QSpinBox()
    spin.setRange(lo, hi)
    spin.setValue(init)
    if double:
        spin.setSingleStep(step)
    spin.setFixedWidth(90)
    if on_change:
        spin.valueChanged.connect(on_change)
    h.addWidget(QtWidgets.QLabel(f"{text}:"))
    h.addStretch()
    h.addWidget(spin)
    layout.addWidget(box)
    return Control(box, spin)


def group(title: str) -> QtWidgets.QGroupBox:
    box = QtWidgets.QGroupBox(title)
    QtWidgets.QVBoxLayout(box)
    return box


class CollapsibleBox(QtWidgets.QWidget):
    """A titled section that folds away to save panel space.

    Add controls to ``.content_layout``.
    """

    def __init__(self, title: str, expanded: bool = True, parent=None):
        super().__init__(parent)
        self.toggle = QtWidgets.QToolButton()
        self.toggle.setStyleSheet("QToolButton { border: none; font-weight: bold; }")
        self.toggle.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
        self.toggle.setText(title)
        self.toggle.setCheckable(True)
        self.toggle.setChecked(expanded)
        self.toggle.setArrowType(QtCore.Qt.DownArrow if expanded else QtCore.Qt.RightArrow)
        self.toggle.clicked.connect(self._on_toggle)

        self.content = QtWidgets.QWidget()
        self.content_layout = QtWidgets.QVBoxLayout(self.content)
        self.content_layout.setContentsMargins(8, 2, 2, 6)
        self.content.setVisible(expanded)

        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        outer.addWidget(self.toggle)
        outer.addWidget(self.content)

    def _on_toggle(self, checked: bool):
        self.toggle.setArrowType(QtCore.Qt.DownArrow if checked else QtCore.Qt.RightArrow)
        self.content.setVisible(checked)

    def add(self, widget: QtWidgets.QWidget):
        self.content_layout.addWidget(widget)
