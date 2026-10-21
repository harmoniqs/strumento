"""Tests for the cqed DispersiveIQ readout kind."""

import numpy as np
import pytest

from strumento.core.readout import DataSource, Discriminator, ReadoutKind
from strumento.packs.cqed.readout import DispersiveIQ


def test_is_a_readout_kind():
    d = DispersiveIQ(threshold=0.0)
    assert isinstance(d, ReadoutKind)
    assert isinstance(d, Discriminator)
    assert d.data_source is DataSource.QICK_ADC


def test_discriminates_separated_blobs_and_reduces_to_population():
    rng = np.random.default_rng(0)
    n = 2000
    excited = rng.random(n) < 0.3          # ~30% excited
    i = np.where(excited, 1.0, -1.0) + rng.normal(0, 0.1, n)  # g near I=-1, e near I=+1
    q = rng.normal(0, 0.1, n)
    raw = np.stack([i, q], axis=-1)

    d = DispersiveIQ(threshold=0.0, angle=0.0)
    shots = d.discriminate(raw)
    assert np.mean(shots == excited.astype(int)) > 0.98      # clean separation
    assert float(d.reduce(raw)) == pytest.approx(0.3, abs=0.05)


def test_reduce_batches_over_trailing_shot_axis():
    # (n_points, n_shots, 2): reduce -> per-point populations
    raw = np.zeros((5, 100, 2))
    raw[..., 0] = 1.0  # all excited -> population 1.0 per point
    pops = DispersiveIQ(threshold=0.0).reduce(raw)
    assert pops.shape == (5,)
    assert np.allclose(pops, 1.0)


def test_angle_rotation_puts_separation_on_i():
    # separation entirely in Q; a 90-degree rotation brings it onto rotated-I
    raw = np.array([[0.0, 1.0], [0.0, -1.0]])  # one "e"-like (Q=+1), one "g"-like (Q=-1)
    d = DispersiveIQ(threshold=0.0, angle=np.pi / 2)
    assert list(d.discriminate(raw)) == [1, 0]


def test_to_population_batches_over_sweep_axis():
    g, e = (0.0, 0.0), (2.0, 0.0)
    d = DispersiveIQ(threshold=0.0, angle=0.0, g_center=g, e_center=e)
    raw = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])  # I = 0, 1, 2 -> P_e = 0, 0.5, 1
    assert np.allclose(d.to_population(raw), [0.0, 0.5, 1.0])


def test_to_population_without_centres_raises():
    d = DispersiveIQ(threshold=0.0, angle=0.0)  # no blob centres
    with pytest.raises(ValueError, match="g_center"):
        d.to_population(np.array([1.0, 0.0]))


def test_to_population_coincident_centres_raise():
    d = DispersiveIQ(threshold=0.0, angle=0.0, g_center=(1.0, 1.0), e_center=(1.0, 1.0))
    with pytest.raises(ValueError, match="same point"):
        d.to_population(np.array([1.0, 1.0]))


def test_to_level_populations_recovers_a_three_level_iq_mixture():
    d = DispersiveIQ(
        threshold=0.0,
        g_center=(0.0, 0.0),
        e_center=(2.0, 0.0),
        f_center=(0.0, 3.0),
    )
    raw = 0.2 * np.array(d.g_center) + 0.3 * np.array(d.e_center) + 0.5 * np.array(d.f_center)

    populations = d.to_level_populations(raw)

    assert np.allclose(populations, [0.2, 0.3, 0.5])


def test_to_level_populations_rejects_an_ill_conditioned_f_centre():
    d = DispersiveIQ(
        threshold=0.0,
        g_center=(0.0, 0.0),
        e_center=(1.0, 0.0),
        f_center=(1.0, 1.0e-4),
    )

    with pytest.raises(ValueError, match="ill-conditioned"):
        d.to_level_populations(np.array([0.5, 0.25]))
