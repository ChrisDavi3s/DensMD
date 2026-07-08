"""Application boot: build the Qt app and show the main window."""
from __future__ import annotations

import sys
from typing import Optional

from PyQt5 import QtWidgets

from .config import Settings
from .io import LoadSpec
from .ui.main_window import MainWindow


def run(spec: Optional[LoadSpec] = None) -> int:
    """Launch DensMD. With no ``spec`` the user opens a file via File > Open."""
    settings = Settings.load()
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow(settings, spec)
    window.resize(settings.window_width, settings.window_height)
    window.show()
    if spec is None:
        window._open_dialog()
    return app.exec_()


if __name__ == "__main__":
    sys.exit(run())
