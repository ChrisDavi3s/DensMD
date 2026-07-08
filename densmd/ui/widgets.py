"""Small reusable Qt controls.

``labelled_slider`` / ``labelled_spinbox`` are the one consistent way panels
build a labelled control, instead of three hand-rolled near-copies; each
returns a lightweight ``Control`` holding the widget and its live label.
``Slider`` / ``SpinBox`` / ``DoubleSpinBox`` / ``ComboBox`` are the actual
widget classes underneath -- plain QWidget stand-ins except they ignore mouse
wheel input while unfocused, since every one of these lives inside a tall
QScrollArea and would otherwise change value whenever the user scrolls past.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

from PyQt5 import QtCore, QtWidgets


class _NoStrayWheel:
    """Mixin: ignore wheel events unless the widget already has focus.

    Every slider/spinbox/combobox lives in a tall QScrollArea. Without this,
    the mouse wheel changes whatever control happens to be under the cursor
    while the user is just scrolling past it -- e.g. nudging the grid
    resolution spinbox resets every ROI slider back to full range, which
    reads as "the sliders forgot their position". Click/tab into a control
    first and the wheel still works for fine adjustment.
    """

    def wheelEvent(self, event):
        if self.hasFocus():
            super().wheelEvent(event)
        else:
            event.ignore()


class Slider(_NoStrayWheel, QtWidgets.QSlider):
    pass


class SpinBox(_NoStrayWheel, QtWidgets.QSpinBox):
    pass


class DoubleSpinBox(_NoStrayWheel, QtWidgets.QDoubleSpinBox):
    pass


class ComboBox(_NoStrayWheel, QtWidgets.QComboBox):
    pass


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
    slider = Slider(QtCore.Qt.Horizontal)
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
    spin = DoubleSpinBox() if double else SpinBox()
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
