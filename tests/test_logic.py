"""Static logic tests for the numpy-only core (unittest version).

Heavy GUI/graphics deps (scipy, vtk, pyvista, matplotlib, PyQt5) are stubbed so
the pure numerical logic can be exercised without a display or those installs.

Run with (from the repo root):  python -m unittest discover -s tests -v
"""
import sys
import types
import unittest
import tempfile
import pathlib
import numpy as np

# --- stub heavy deps (must happen before densmd imports) -------------------
def _stub(name, **attrs):
    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    sys.modules[name] = mod
    return mod

scipy = _stub("scipy")
_stub("scipy.ndimage",
      gaussian_filter=lambda a, sigma: a,
      map_coordinates=lambda d, idx, order, mode, cval: np.zeros(idx.shape[1]))
scipy.ndimage = sys.modules["scipy.ndimage"]

class _CMap:
    def __call__(self, x):
        x = np.asarray(x)
        return np.stack([x, x, x, np.ones_like(x)], axis=-1)

_stub("matplotlib")
_stub("matplotlib.pyplot", get_cmap=lambda name: _CMap())

class _TF:
    def __init__(self, *a): self.pts = []
    def AddRGBPoint(self, *a): self.pts.append(a)
    def AddPoint(self, *a): self.pts.append(a)

class _LUT:
    def SetNumberOfTableValues(self, n): self.n = n
    def SetTableRange(self, *a): pass
    def SetTableValue(self, *a): pass
    def Build(self): pass

class _InteractorStyle:
    def __init__(self, *a, **k): pass

_stub("vtk", vtkColorTransferFunction=_TF, vtkPiecewiseFunction=_TF,
      vtkLookupTable=_LUT,
      vtkInteractorStyleTrackballCamera=_InteractorStyle)

_stub("pyvista")

# --- imports under test ---------------------------------------------------
sys.path.insert(0, ".")
from densmd.config import Settings
from densmd import io, miller, model, render, unwrap
from densmd.render import histogram_rgba


class FakeAtoms:
    def __init__(self, syms): self._s = syms
    def get_chemical_symbols(self): return list(self._s)


class TestConfig(unittest.TestCase):
    def test_settings_roundtrip(self):
        p = pathlib.Path(tempfile.mktemp())
        Settings(grid_resolution=123).save(p)
        self.assertEqual(Settings.load(p).grid_resolution, 123)

    def test_settings_default_on_missing_file(self):
        s = Settings.load(pathlib.Path("/nope/nope.json"))
        self.assertEqual(s.grid_resolution, 300)


class TestIO(unittest.TestCase):
    def test_guess_format(self):
        self.assertEqual(io.guess_format("a.dat"), "ase")
        self.assertEqual(io.guess_format("a.pkl"), "pickle")

    def test_parse_slice(self):
        self.assertEqual(io.parse_slice("::5"), slice(None, None, 5))
        self.assertEqual(io.parse_slice(":"), slice(None))

    def test_atom_type_map(self):
        spec = io.LoadSpec("x", atom_type_map={"H": "Li"})
        self.assertEqual(spec.mapped_symbols(FakeAtoms(["H", "O"])), ["Li", "O"])


class TestMiller(unittest.TestCase):
    def test_normal(self):
        mp = miller.MillerParams(True, 1, 0, 0, 2.0, 0.0)
        self.assertTrue(np.allclose(mp.normal, [1, 0, 0]))
        self.assertIsNone(miller.MillerParams(False, 1, 1, 1, 1, 0).normal)

    def test_voxel_centers_and_mask(self):
        mp = miller.MillerParams(True, 1, 0, 0, 2.0, 0.0)
        roi = dict(xmin=0, xmax=4, ymin=0, ymax=4, zmin=0, zmax=4)
        vc = miller.voxel_centers(roi, np.zeros(3), np.ones(3))
        self.assertEqual(vc.shape, (5, 5, 5, 3))
        mask = miller.voxel_mask(vc, np.array([2.0, 2, 2]), mp)
        self.assertEqual(mask.shape, (5, 5, 5))
        self.assertTrue(mask.any())

    def test_filter_points(self):
        mp = miller.MillerParams(True, 1, 0, 0, 2.0, 0.0)
        pts = np.array([[2.0, 2, 2], [10, 2, 2]])
        self.assertEqual(len(miller.filter_points(pts, np.array([2.0, 2, 2]), mp)), 1)


class TestModel(unittest.TestCase):
    def setUp(self):
        self.m = model.DensityModel(Settings(grid_resolution=10, quantile_bins=16))
        m = self.m
        m.species = ["A"]
        m.origin = np.zeros(3); m.spacing = np.ones(3); m.dims = np.array([10, 10, 10])
        m.cell_center = np.array([5.0, 5, 5])
        hist = np.random.rand(10, 10, 10).astype(np.float32)
        m.atom_data = {"A": {"raw_hist": hist,
                             "global_positions": np.zeros((1, 3)),
                             "individual_averages": np.zeros((0, 3))}}
        self.roi = dict(xmin=2, xmax=6, ymin=2, ymax=6, zmin=2, zmax=6)

    def test_quantile_transform_bounds(self):
        q = model.quantile_transform(np.array([0.0, 0, 0, 0, 5, 9], dtype=np.float32))
        self.assertEqual(q.ravel()[0], 0.0)
        self.assertLessEqual(q.max(), 255.0)
        self.assertGreaterEqual(q.min(), 0.0)

    def test_region_and_cache(self):
        mp = miller.MillerParams(False, 1, 1, 1, 1, 0)
        reg = self.m.region(self.roi, mp)
        self.assertIsNone(reg.mask)
        self.assertTrue(np.allclose(reg.phys_min, [2, 2, 2]))
        self.assertIs(self.m.region(self.roi, mp), reg)  # cache hit

    def test_volume_data(self):
        reg = self.m.region(self.roi, miller.MillerParams(False, 1, 1, 1, 1, 0))
        vol = self.m.volume_data("A", 0, reg, smooth_before=True)
        self.assertEqual(vol.data.shape, (5, 5, 5))
        self.assertEqual(vol.quantile.shape, (5, 5, 5))
        self.assertGreaterEqual(vol.quantile.min(), 0.0)
        self.assertLessEqual(vol.quantile.max(), 255.0)
        self.assertIsNone(vol.mask)

    def test_miller_masked_volume_rgba(self):
        regm = self.m.region(self.roi, miller.MillerParams(True, 1, 0, 0, 2.0, 0.0))
        volm = self.m.volume_data("A", 0, regm, smooth_before=True)
        self.assertIsNotNone(volm.mask)
        self.assertEqual(volm.mask.shape, (5, 5, 5))
        rgba = histogram_rgba(volm, render.Appearance(
            density_lower=77, density_upper=178, opacity=100, gamma=0.0))
        self.assertEqual(rgba.shape, (5, 5, 5, 4))
        self.assertEqual(rgba.dtype, np.uint8)
        self.assertTrue(np.all(rgba[~volm.mask, 3] == 0))

    def test_empty_voxels_transparent(self):
        d2 = np.zeros((6, 6, 6), dtype=np.float32)
        d2[2:4, 2:4, 2:4] = np.arange(1, 9).reshape(2, 2, 2) * 5.0
        qv = model.quantile_transform(d2)
        vol2 = model.VolumeData(d2, qv, None, np.zeros(3), np.ones(3))
        r2 = histogram_rgba(vol2, render.Appearance(
            density_lower=77, density_upper=178, opacity=100, gamma=0.0))
        self.assertTrue(np.all(r2[d2 == 0, 3] == 0))
        self.assertEqual(r2[d2 > 0, 3].max(), 255)

    def test_grid_rebuild(self):
        m = self.m
        m._gmin = np.zeros(3); m._gmax = np.array([9.0, 9, 9])
        m.atom_data["A"]["global_positions"] = np.random.rand(500, 3) * 9
        m.rebuild_grid(6)
        self.assertEqual(m.atom_data["A"]["raw_hist"].shape, (6, 6, 6))


class TestUnwrap(unittest.TestCase):
    def setUp(self):
        self.cells = np.stack([np.eye(3) * 10.0] * 4)
        # wrapped x positions 9,1,9,1 (crossing +x boundary each frame)
        self.traj = np.array([[[9.0, 5, 5]], [[1.0, 5, 5]],
                              [[9.0, 5, 5]], [[1.0, 5, 5]]])

    def test_boundary_hopping(self):
        avg = unwrap.averaged_positions(self.traj, self.cells, stride=1, unwrap=True)
        naive = self.traj[:, 0, 0].mean()
        self.assertAlmostEqual(naive, 5.0)  # naive mean is midpoint (wrong)
        self.assertLess(min(avg[0, 0], 10 - avg[0, 0]), 1.0)  # near boundary

    def test_npt_expanding_cell_finite(self):
        cells_npt = np.stack([np.eye(3) * (10 + i) for i in range(4)])
        avg = unwrap.averaged_positions(self.traj, cells_npt, stride=1, unwrap=True)
        self.assertTrue(np.all(np.isfinite(avg)))

    def test_subsample_stride(self):
        avg = unwrap.averaged_positions(self.traj, self.cells, stride=2)
        self.assertTrue(np.all(np.isfinite(avg)))


class TestRenderHelpers(unittest.TestCase):
    def test_alpha_curve_ramps(self):
        a = render.Appearance(density_lower=0, density_upper=255, opacity=100, gamma=1.0)
        alpha = render._alpha_curve(np.array([0.0, 128, 255]), a)
        self.assertEqual(alpha[0], 0)
        self.assertAlmostEqual(alpha[-1], 1.0, places=6)

    def test_gamma_zero_step(self):
        a0 = render.Appearance(density_lower=64, density_upper=200, opacity=50, gamma=0.0)
        al = render._alpha_curve(np.array([0.0, 100, 255]), a0)
        self.assertEqual(al[0], 0)
        self.assertAlmostEqual(al[1], 0.5, places=6)
        self.assertAlmostEqual(al[2], 0.5, places=6)

    def test_density_at(self):
        qv = np.linspace(0, 10, 256)
        self.assertAlmostEqual(render._density_at(qv, 255), 10, places=6)
        self.assertAlmostEqual(render._density_at(qv, 0), 0, places=6)

    def test_color_inputs_clipped(self):
        qv = np.linspace(0, 10, 256)
        ci = render._color_inputs(np.array([-5.0, 0, 5, 10, 20]), qv, 0, 10,
                                  render.Appearance())
        self.assertEqual(ci.min(), 0)
        self.assertEqual(ci.max(), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)