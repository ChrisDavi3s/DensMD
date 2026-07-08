"""Backwards-compatible launcher.

The application now lives in the ``densmd`` package. This thin shim keeps the
old ``python run_densmd.py`` entry point working. Equivalent to ``python -m densmd``.

Optionally pass a file to open it directly, e.g.:
    python run_densmd.py /path/to/traj.dat --slice ::5 --map H:Li,He:P

Copyright (c) 2025 Chris Davies. Licensed under AGPLv3.
"""
import argparse
import sys

from densmd.app import run
from densmd.io import LoadSpec, guess_format
from densmd.ui.dialogs import _parse_type_map


def _parse_args(argv):
    p = argparse.ArgumentParser(description="DensMD trajectory density viewer")
    p.add_argument("path", nargs="?", help="trajectory file (optional; else use File > Open)")
    p.add_argument("--format", default="auto", choices=["auto", "ase", "pickle"])
    p.add_argument("--slice", default="::5", help="ASE frame slice, e.g. ::5")
    p.add_argument("--map", default="", help="atom type map, e.g. H:Li,He:P")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    spec = None
    if args.path:
        fmt = guess_format(args.path) if args.format == "auto" else args.format
        spec = LoadSpec(path=args.path, fmt=fmt, frame_slice=args.slice,
                        atom_type_map=_parse_type_map(args.map))
    return run(spec)


if __name__ == "__main__":
    sys.exit(main())
