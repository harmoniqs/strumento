"""The shared synthetic-readout helpers (``tests/_synthetic_readout.py``) themselves.

The blob-anchored tests (DRAG, DRAG-pair) lean on one synthetic g/e geometry whose angle and
threshold are derived from the centres; these pin that the derivation is self-consistent with
``DispersiveIQ`` rather than trusting hand-entered digits.
"""

from __future__ import annotations

import numpy as np
import pytest

from tests._synthetic_readout import (
    ANGLE,
    E_CENTER,
    G_CENTER,
    GE_BLOBS,
    THRESHOLD,
    BlobGeometry,
    dispersive_iq,
    result_from_populations,
)


def test_derived_angle_and_threshold_separate_g_and_e():
    """Along the derived angle, g projects below the threshold and e above it, symmetrically."""
    ro = dispersive_iq()
    g = float(ro.project(np.asarray(G_CENTER)))
    e = float(ro.project(np.asarray(E_CENTER)))
    assert g < THRESHOLD < e
    assert THRESHOLD - g == pytest.approx(e - THRESHOLD)
    assert ro.discriminate(np.asarray([G_CENTER, E_CENTER])).tolist() == [0, 1]
    assert float(ro.to_population(np.asarray(G_CENTER))) == pytest.approx(0.0, abs=1e-12)
    assert float(ro.to_population(np.asarray(E_CENTER))) == pytest.approx(1.0, abs=1e-12)


def test_angle_is_the_g_to_e_direction():
    """The rotated-I axis is parallel to E − G, so the orthogonal (rotated-Q) component is equal."""
    d = np.subtract(E_CENTER, G_CENTER)
    assert ANGLE == pytest.approx(np.arctan2(d[1], d[0]))
    rot_q = lambda c: -c[0] * np.sin(ANGLE) + c[1] * np.cos(ANGLE)  # noqa: E731
    assert rot_q(G_CENTER) == pytest.approx(rot_q(E_CENTER))
    assert (GE_BLOBS.angle, GE_BLOBS.threshold) == (ANGLE, THRESHOLD)


def test_geometry_works_for_any_centres():
    geo = BlobGeometry(g_center=(1.0, 3.0), e_center=(-2.0, -1.0))
    ro = geo.readout()
    assert float(ro.to_population(np.asarray(geo.g_center))) == pytest.approx(0.0, abs=1e-12)
    assert float(ro.to_population(np.asarray(geo.e_center))) == pytest.approx(1.0, abs=1e-12)
    assert ro.discriminate(np.asarray([geo.g_center, geo.e_center])).tolist() == [0, 1]


def test_readout_with_and_without_centres():
    full = dispersive_iq()
    assert (full.g_center, full.e_center) == (G_CENTER, E_CENTER)
    bare = dispersive_iq(with_centres=False)
    assert (bare.g_center, bare.e_center) == (None, None)
    assert (bare.angle, bare.threshold) == (full.angle, full.threshold)


def test_result_from_populations_round_trips_through_to_population():
    pops = np.array([0.0, 0.25, 0.5, 1.0])
    res = result_from_populations(pops)
    assert res.xpts is None and res.shots is None
    iq = np.stack([res.avgi, res.avgq], axis=-1)
    assert dispersive_iq().to_population(iq) == pytest.approx(pops)

    xs = np.linspace(-1.0, 1.0, pops.size)
    assert np.array_equal(result_from_populations(pops, xpts=xs).xpts, xs)
