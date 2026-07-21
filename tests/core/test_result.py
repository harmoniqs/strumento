"""Tests for the MeasurementResult seam (spec §18/D24, plan Task 3)."""

import numpy as np

from strumento.core.result import MeasurementResult


def test_as_tuple_returns_xyz():
    xpts, avgi, avgq = np.linspace(0, 1, 5), np.zeros(5), np.ones(5)
    r = MeasurementResult(xpts=xpts, avgi=avgi, avgq=avgq)
    x, i, q = r.as_tuple()
    assert x is xpts and i is avgi and q is avgq


def test_postselect_is_identity_without_lanes():
    r = MeasurementResult(xpts=[0.0], avgi=[0.0], avgq=[0.0])
    assert r.postselect().as_tuple() == r.as_tuple()


def test_meta_environment_key_roundtrips():
    # meta["environment"] is the reserved station-telemetry slot (§18/D24); the Runner stamps
    # it at acquire (Task 19). Reserving it here means every result can carry it from day one.
    r = MeasurementResult(
        xpts=[0.0], avgi=[0.0], avgq=[0.0],
        meta={"environment": {"mxc_temp_mK": 12.3}},
    )
    assert r.meta["environment"]["mxc_temp_mK"] == 12.3


def test_default_meta_is_empty_and_not_shared():
    a = MeasurementResult(xpts=[0], avgi=[0], avgq=[0])
    b = MeasurementResult(xpts=[0], avgi=[0], avgq=[0])
    a.meta["scratch"] = 1
    assert b.meta == {}  # no shared mutable default between instances


def test_raw_and_herald_defaults():
    r = MeasurementResult(xpts=[0], avgi=[0], avgq=[0])
    assert r.raw is None
    assert r.herald_lanes == ()
