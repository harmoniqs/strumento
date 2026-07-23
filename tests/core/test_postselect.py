"""Slice 7a — per-shot data + herald post-selection on MeasurementResult (pure, spec R1(ii))."""

from __future__ import annotations

import numpy as np
import pytest

from strumento.core.result import MeasurementResult
from strumento.packs.cqed.readout import DispersiveIQ


def test_no_heralds_is_identity_with_full_survival():
    r = MeasurementResult(xpts=None, avgi=0.5, avgq=0.0, shots=np.array([1, 0, 1, 0]))
    ps = r.postselect()
    assert list(ps.shots) == [1, 0, 1, 0]
    assert ps.meta["postselect_survival"] == 1.0


def test_single_lane_filters_shots():
    shots = np.array([10, 11, 12, 13, 14])          # per-shot data lane
    reset = np.array([True, False, True, True, False])  # herald: keep shots 0,2,3
    r = MeasurementResult(xpts=None, avgi=None, avgq=None, shots=shots,
                          heralds={"reset": reset}, herald_lanes=("reset",))
    ps = r.postselect()
    assert list(ps.shots) == [10, 12, 13]
    assert list(ps.heralds["reset"]) == [True, True, True]   # herald filtered alongside
    assert ps.meta["postselect_survival"] == pytest.approx(3 / 5)
    assert ps.meta["postselect_lanes"] == ("reset",)


def test_multiple_lanes_are_anded():
    shots = np.arange(6)
    a = np.array([1, 1, 1, 0, 1, 0], dtype=bool)
    b = np.array([1, 0, 1, 1, 1, 0], dtype=bool)     # AND -> keep 0, 2, 4
    r = MeasurementResult(xpts=None, avgi=None, avgq=None, shots=shots,
                          heralds={"g": a, "parity": b})   # herald_lanes empty -> use all
    ps = r.postselect()
    assert list(ps.shots) == [0, 2, 4]
    assert ps.meta["postselect_survival"] == pytest.approx(0.5)


def test_lanes_argument_selects_a_subset():
    shots = np.arange(4)
    r = MeasurementResult(xpts=None, avgi=None, avgq=None, shots=shots,
                          heralds={"g": np.array([1, 1, 0, 0], dtype=bool),
                                   "parity": np.array([1, 0, 1, 0], dtype=bool)})
    only_g = r.postselect(lanes=("g",))
    assert list(only_g.shots) == [0, 1]              # ignores the parity lane
    assert only_g.meta["postselect_lanes"] == ("g",)


def test_keep_mask_exposed_for_multidim_sweep():
    # (2 points, 4 shots) herald mask -> compacting would be ragged, so the mask is exposed
    shots = np.arange(8).reshape(2, 4)
    herald = np.array([[1, 1, 0, 1], [0, 1, 1, 1]], dtype=bool)
    r = MeasurementResult(xpts=[0, 1], avgi=None, avgq=None, shots=shots,
                          heralds={"reset": herald}, herald_lanes=("reset",))
    ps = r.postselect()
    assert ps.shots.shape == (2, 4)                  # left intact
    assert ps.meta["keep_mask"].shape == (2, 4)
    assert ps.meta["postselect_survival"] == pytest.approx(6 / 8)


def test_all_rejected_gives_zero_survival():
    shots = np.array([1, 2, 3])
    r = MeasurementResult(xpts=None, avgi=None, avgq=None, shots=shots,
                          heralds={"g": np.array([0, 0, 0], dtype=bool)}, herald_lanes=("g",))
    ps = r.postselect()
    assert ps.shots.size == 0
    assert ps.meta["postselect_survival"] == 0.0


# -- the discriminator -> herald mask bridge ---------------------------------------------


def test_dispersive_iq_herald_builds_pass_mask():
    d = DispersiveIQ(threshold=0.0, angle=0.0)
    # per-shot IQ for a herald event: shots 0,2 are |g> (I<0), shots 1,3 are |e> (I>0)
    iq = np.array([[-1.0, 0.0], [1.0, 0.0], [-0.8, 0.0], [0.9, 0.0]])
    keep_g = d.herald(iq, keep_state=0)
    assert list(keep_g) == [True, False, True, False]     # keep ground-state shots
    keep_e = d.herald(iq, keep_state=1)
    assert list(keep_e) == [False, True, False, True]


def test_end_to_end_herald_then_postselect():
    d = DispersiveIQ(threshold=0.0, angle=0.0)
    rng = np.random.default_rng(3)
    n = 500
    # herald round: ~80% land in |g> (kept); data round: per-shot IQ we keep as shots
    herald_iq = np.stack([np.where(rng.random(n) < 0.8, -1.0, 1.0), np.zeros(n)], axis=-1)
    data_shots = rng.normal(0.5, 0.2, n)
    r = MeasurementResult(
        xpts=None, avgi=None, avgq=None,
        shots=data_shots,
        heralds={"reset": d.herald(herald_iq, keep_state=0)},
        herald_lanes=("reset",),
    )
    ps = r.postselect()
    assert 0.7 < ps.meta["postselect_survival"] < 0.9
    assert ps.shots.size == int(round(ps.meta["postselect_survival"] * n))
