"""Trajectory loading and input specification.

Replaces the old module-level ``INPUT_FILE_CONFIG`` with a small value object
that the Open dialog fills in, so the app runs without editing source.
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

# Extension -> reader format used for auto-detection in the Open dialog.
_EXT_FORMAT = {
    ".pickle": "pickle", ".pkl": "pickle", ".p": "pickle",
    ".dat": "ase", ".xyz": "ase", ".traj": "ase", ".extxyz": "ase",
    ".cif": "ase", ".vasp": "ase", ".xdatcar": "ase",
}


def guess_format(path: str) -> str:
    """Best-guess reader format from a file extension (defaults to ase)."""
    return _EXT_FORMAT.get(Path(path).suffix.lower(), "ase")


def parse_slice(spec) -> slice:
    """Turn an ASE-style slice string (e.g. ``"::5"``) into a ``slice``."""
    if isinstance(spec, slice):
        return spec
    if not spec:
        return slice(None)
    parts = str(spec).split(":")
    nums = [int(p) if p.strip() else None for p in parts]
    return slice(*nums)


@dataclass
class LoadSpec:
    """Everything needed to load and pre-map a trajectory."""

    path: str
    fmt: str = "ase"                       # 'ase' or 'pickle'
    frame_slice: str = "::5"               # ASE index string
    atom_type_map: Optional[Dict[str, str]] = None
    # Frames used for the averaged-positions mode (workaround for wrapping).
    average_frame_slice: slice = field(default_factory=lambda: slice(0, 1))

    def load_frames(self) -> List:
        """Read frames from disk applying the frame slice. Returns a list."""
        sl = parse_slice(self.frame_slice)
        if self.fmt == "pickle":
            with open(self.path, "rb") as fh:
                frames = pickle.load(fh)
            return frames[sl]
        if self.fmt == "ase":
            from ase.io import read
            frames = read(self.path, index=sl)
            return frames if isinstance(frames, list) else [frames]
        raise ValueError(f"Unsupported file format: {self.fmt!r}")

    def load_first_frame(self):
        """Cheaply read just the first frame (used to scan species)."""
        if self.fmt == "pickle":
            with open(self.path, "rb") as fh:
                frames = pickle.load(fh)
            return frames[0]
        if self.fmt == "ase":
            from ase.io import read
            return read(self.path, index=0)
        raise ValueError(f"Unsupported file format: {self.fmt!r}")

    def mapped_symbols(self, atoms) -> List[str]:
        """Chemical symbols with ``atom_type_map`` applied."""
        symbols = atoms.get_chemical_symbols()
        if not self.atom_type_map:
            return symbols
        return [self.atom_type_map.get(s, s) for s in symbols]
