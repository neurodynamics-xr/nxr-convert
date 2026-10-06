"""The atlas producer's own gates beyond nsp's tests (D135): the ladder on an IRREGULAR mesh (nested, equal mass),
the tower-of-cycles FIR bank (a power partition, exact when chunked), — synthetic data only. (``reduce`` as rows: ``test_atlas_rows.py``.)"""
import json

import numpy as np
import pytest
import zarr

from atlas_meshes import icosphere
from nxr_convert.atlas import bands as fb
from nxr_convert.atlas import rollup, scalars
from nxr_convert.atlas.ladder import surface_ladder, vertex_areas


def bumpy_sphere(level=4, seed=0):
    """An icosphere with its radius perturbed, so vertex masses are unequal (a stand-in for a cortex)."""
    p, f = icosphere(level, 1.0)
    rng = np.random.default_rng(seed)
    r = 1 + 0.25 * np.sin(3 * p[:, 0]) * np.cos(2 * p[:, 1]) + 0.05 * rng.standard_normal(len(p))
    return p * r[:, None], f


def test_ladder_bisection_nests_and_halves_mass_on_an_irregular_mesh():
    p, f = bumpy_sphere()
    lad = surface_ladder(p, f)
    mass = vertex_areas(p, f)
    assert lad.finest >= 8
    assert np.isclose(lad.area, mass.sum())
    for l in range(1, lad.finest + 1):
        child, parent = lad.labels[l], lad.labels[l - 1]
        assert np.array_equal(child >> 1, parent)                                      # NESTED: a child's parent is >> 1
        assert np.array_equal(lad.labels[-1] >> (lad.finest - l), child)                # the levels are arithmetic
        m_child, m_parent = lad.tile_areas(l), lad.tile_areas(l - 1)
        assert (np.bincount(child, minlength=2 ** l) > 0).all()                         # no empty tile
        # EQUAL MASS: each half is half its parent, to within the heaviest vertex of that parent
        heaviest = np.zeros(2 ** (l - 1))
        np.maximum.at(heaviest, parent, mass)
        even, odd = m_child[0::2], m_child[1::2]
        assert np.allclose(even + odd, m_parent)
        assert (np.abs(even - m_parent / 2) <= heaviest + 1e-15).all()


def test_ladder_breaks_when_a_gate_fails():
    """The gate can fail: an un-nested labelling is caught (proves the assertion above bites)."""
    p, f = bumpy_sphere()
    lad = surface_ladder(p, f)
    bad = lad.labels[3].copy()
    bad[0] ^= 1                                                                         # move one vertex to its sibling
    assert np.array_equal(bad >> 1, lad.labels[2])                                       # still nested …
    assert not np.array_equal(lad.labels[-1] >> (lad.finest - 3), bad)                   # … but not the arithmetic level


def test_octave_bank_squares_sum_to_one_and_bands_add_to_broadband():
    f = np.linspace(0.05, 300, 20000)
    bands = scalars.octave_bands(1.0, 6)
    g2 = (scalars.band_gains(f, bands) ** 2).sum(0)
    inside = (f > 2 ** 0.25) & (f < 64 / 2 ** 0.25)
    assert np.allclose(g2[inside], 1.0, atol=1e-12)
    # Parseval: band powers of a signal whose spectrum lies inside the bank add to its power
    sf, n = 256.0, 4096
    t = np.arange(n) / sf
    x = (np.sin(2 * np.pi * 5 * t) + 0.5 * np.sin(2 * np.pi * 21 * t))[None, :].astype(np.float32)
    total = sum(float((np.abs(A) ** 2).sum()) for _, A in scalars.analytic_bands(x, sf, bands)) / 2
    assert np.isclose(total, float((x.astype(np.float64) ** 2).sum()), rtol=1e-4)


def test_tower_fir_bank_is_a_power_partition_and_exact_when_chunked():
    sf = 600.0
    levels = fb.levels_for(sf, 1.0, 100.0)
    assert levels
    filters = [fb.analytic_fir(L, sf, levels) for L in levels]
    err = fb.partition_error(filters, sf)
    assert err < 0.1                                                      # 4 cycles: within 10 % (nsp's bound)
    longer = [fb.analytic_fir(L, sf, levels, cycles=12) for L in levels]
    assert fb.partition_error(longer, sf) < err                           # a longer FIR is a closer partition
    F = filters[len(filters) // 2]
    rng = np.random.default_rng(1)
    x = rng.standard_normal((2, 4000))
    pad = np.pad(x, ((0, 0), (F.half, F.half)))
    whole = fb.filter_block(pad, F)
    a, b = 1000, 2500                                                     # one block, padded by H from the record
    part = fb.filter_block(pad[:, a:b + 2 * F.half], F)
    assert np.allclose(part, whole[:, a:b], atol=1e-10)
    good = np.ones(4000, bool)
    good[2000] = False
    c = fb.clean(good, F.half)
    assert not c[:F.half].any() and not c[2000 - F.half:2001 + F.half].any() and c[F.half:2000 - F.half].all()
