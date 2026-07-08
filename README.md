# DensMD

A simple PyQt5-based application for visualising atomic trajectory data (constant volume) in 3D, designed specifically for molecular dynamics analysis.
![Main Interface](images/main_screen.png)

## Overview

DensMD provides a powerful, quick, interactive GUI for loading molecular dynamics trajectory data, computing density histograms and averaged positions, and visualising the results with various rendering modes and slicing options. The tool is particularly useful for analysing atomic distributions and migrations in crystal structures.

## Features

- **Multiple Visualisation Modes**
  - Voxel density histograms with customisable colour maps, rendered on the
    GPU (fast) or CPU (pixel-exact edges) — switchable in the Render panel
  - Nested isosurface shells extracted from the same density field — fast,
    razor-sharp, with per-shell opacity ramping (shell count, band tolerance
    and surface smoothing are adjustable per atom)
  - Averaged atomic positions with adjustable sphere sizes — by default each
    atom is drawn at its *most-visited site* (periodic-aware mode), so
    two-site hoppers never smear into the gap between sites
  - Per-atom type visualisation settings

- **Miller Plane Slicing**
  - Define custom Miller indices (hkl)
  - Adjustable slice thickness and offset
  - Automatic camera alignment to plane

- **Responsive Rendering**
  - Appearance changes (colour, opacity, gamma) never recompute the density
  - Interactive LOD: coarse sampling while the camera moves, full quality on
    release (toggleable in the Render panel)

## Screenshots

The new loading UI, which allows you to select a file, set a frame slice and optionally remap atom types.

![Main Interface](images/densmd_ui.png)


A miller slice of 1 atom type shown and the average positions of a second atom type plotted.

![Miller Slicing](images/ui_miller_slicing.png)

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/ChrisDavi3s/densmd.git
   cd densmd
   ```

2. Create and activate a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

No source editing required. Launch the app and load a file from the menu:

```bash
python -m densmd        # or: python run_densmd.py
```

On launch an **Open Trajectory** dialog appears. Pick a file (format is
auto-detected from the extension), set a frame slice such as `::5`, and
optionally supply an atom-type remap like `H:Li, He:P, Li:S`. You can reopen it
any time via **File > Open**.

To open a file directly from the command line:

```bash
python run_densmd.py /path/to/traj.dat --slice ::5 --map H:Li,He:P
```

Averaged-atom positions default to the **most-visited site (mode)**: each
atom's trajectory is kernel-density scored with minimum-image distances, and
the atom is drawn at the centre of its dominant cluster. A vibrating atom
shows at its site, a 70/30 two-site hopper shows at the 70 % site, and a
boundary hopper shows at the boundary — no statistic ever lands in a gap the
atom never occupies. A periodic-aware **circular mean** and the **naive
mean** are also available in **File > Settings**, along with a frame
subsample stride. For **NPT** trajectories each frame is remapped into the
mean cell via fractional coordinates, so both the density histograms and the
atom positions stay consistent as the cell fluctuates.

Other compute defaults (smoothing, update delay, plane resolution) also live in
**File > Settings** and persist to `~/.densmd.json`.

## Architecture

```
densmd/
  config.py     runtime settings + JSON persistence
  io.py         trajectory loading, slice parsing, type remap
  miller.py     Miller-plane geometry (normals, masks, filtering)
  unwrap.py     representative positions (PBC-aware mode / circular mean)
  model.py      compute core: histograms, smoothing/region caches, scalar fields
  render.py     PyVista/VTK layer: volumes, isosurfaces, LUTs, stereo, camera
  ui/           Qt widgets, panels, dialogs, main window
  app.py        application boot
```

Updates are split into two tiers. *Geometry* changes (ROI, Miller, smoothing)
recompute a scalar field, debounced. *Appearance* changes (colour, opacity,
gamma, colormap) only re-run a cheap RGBA/LUT remap over cached arrays on the
existing actor, so slider drags are effectively free — no density recompute
and no camera jump.

## Dependencies

- PyQt5
- PyVista / pyvistaqt
- NumPy
- SciPy
- ASE (Atomic Simulation Environment)
- VTK

## Future Development

Done recently:
- Multi-module package (config / io / miller / model / render / ui)
- In-app file loading and settings (no source editing)
- Two-tier update pipeline for fast appearance changes
- Stereo 3D render toggle
- Isosurface shell mode (nested contours, fast + sharp)
- GPU/CPU volume mapper toggle and interactive LOD
- Averaged ion positions done properly: most-visited site (periodic-aware
  mode) by default, with circular-mean and naive-mean options — no more
  hoppers averaging into gaps they never occupy

Still planned:
- Export functionality for images and videos
- Measurement tools for atomic distances and angles

## License

This project is licensed under the GNU Affero General Public License v3.0 - see the [LICENSE](LICENSE) file for details.

## Credits

Developed by Chris Davies (2025)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request
