"""Slices 5–6 — parity composite/fringe + from_solution OC import (pure)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from strumento.core.device import Device
from strumento.core.fitters import fit_parity
from strumento.core.pulses import Arb, Delay, Pulse, Segment, from_solution
from strumento.core.result import MeasurementResult
from strumento.core.sweeps import Sweep
from strumento.core.wiring import LineRef
from strumento.packs.cqed.calibration import PiPulseReference
from strumento.packs.cqed.experiments.parity import ParityFringe
from strumento.packs.cqed.library.composite import Fock, parity, prepare

YAML = Path(__file__).parents[3] / "devices" / "multimode_demo" / "device.yaml"


@pytest.fixture
def dev():
    return Device.load(YAML)


# -- parity -----------------------------------------------------------------------------


def test_parity_composite_shape(dev):
    seq = parity(dev, "manipulate")
    kinds = [type(op).__name__ for op in seq.ops]
    assert kinds == ["Pulse", "Delay", "Pulse"]
    assert seq.ops[1].duration_us == pytest.approx(1.6)      # calibrated revival
    assert seq.ops[2].phase_deg == pytest.approx(180.0)
    fast = parity(dev, "manipulate", fast=True)
    assert fast.ops[1].duration_us == pytest.approx(0.8)


def test_parity_fringe_analyze_recovers_contrast(dev):
    ph = np.linspace(0.0, 360.0, 121)
    true_contrast, true_phase = 0.8, 30.0
    y = 0.5 + (true_contrast / 2) * np.cos(np.radians(ph + true_phase))
    fit = ParityFringe(dev).analyze(MeasurementResult(xpts=ph, avgi=y, avgq=np.zeros_like(y)))
    assert fit.success
    assert fit.derived["contrast"] == pytest.approx(true_contrast, rel=0.05)


def test_fit_parity_standalone():
    ph = np.linspace(0.0, 360.0, 61)
    y = 1.0 + 0.3 * np.cos(np.radians(ph))
    fit = fit_parity(ph, y)
    assert fit.derived["contrast"] == pytest.approx(0.6, rel=0.05)


# -- from_solution (slice 6) --------------------------------------------------------------


def _toy_solution():
    t = np.linspace(0.0, 1.0, 101)                       # µs
    return {
        "times": t,
        "controls": {
            "qubit": 4.0 * np.sin(np.pi * t) ** 2,                     # MHz Rabi, real
            "cavity": 2.0 * np.exp(1j * 2 * np.pi * t) * np.sin(np.pi * t),  # complex I+jQ
        },
    }


def _drive_map():
    return {
        "qubit": (LineRef("qubit", "drive"), 4000.0),
        "cavity": (("manipulate", "main"), 5200.0),      # bare tuple form also accepted
    }


def test_from_solution_builds_parallel_arb_segment():
    ref = PiPulseReference(pi_gain_frac=0.25, pi_rabi_mhz=8.0)
    seq = from_solution(_toy_solution(), drive_map=_drive_map(), gain_calib=ref)
    assert len(seq.ops) == 1 and isinstance(seq.ops[0], Segment)
    seg = seq.ops[0]
    assert {p.line for p in seg.pulses} == {LineRef("qubit", "drive"), LineRef("manipulate", "main")}
    by_line = {p.line.line: p for p in seg.pulses}
    q = by_line["qubit"]
    assert isinstance(q.envelope, Arb) and q.envelope.qdata is None
    assert np.max(np.abs(q.envelope.idata)) == pytest.approx(1.0)     # normalized samples
    assert q.gain == pytest.approx(ref(4.0))                           # peak -> fraction
    c = by_line["manipulate"]
    assert c.envelope.qdata is not None                                # complex -> I+Q
    assert seq.meta["source"] == "from_solution" and seq.meta["drives"] == ["cavity", "qubit"]


def test_from_solution_resamples_to_dac_rate():
    seq = from_solution(_toy_solution(), drive_map=_drive_map(),
                        gain_calib=lambda a: a / 10.0, dac_rate=400.0)
    n = seq.meta["n_samples"]
    assert n == 401                                       # 1 µs at 400 samples/µs (+1)
    seg = seq.ops[0]
    assert all(len(p.envelope.idata) == n for p in seg.pulses)


def test_from_solution_unknown_drive_raises():
    with pytest.raises(KeyError, match="drive_map"):
        from_solution(_toy_solution(), drive_map={"qubit": (("qubit", "drive"), 4000.0)},
                      gain_calib=lambda a: a)


# -- prepare ------------------------------------------------------------------------------


def test_prepare_fock1_analytic_ladder(dev):
    seq = prepare(dev, Fock(1))
    labels = [op.label for op in seq.ops if isinstance(op, Pulse)]
    assert labels == ["pi_ge", "pi_ef", "f0-g1_pi0"]


def test_prepare_oc_delegates_to_from_solution(dev):
    seq = prepare(dev, Fock(1), via="oc", solution=_toy_solution(), drive_map=_drive_map(),
                  gain_calib=lambda a: a / 20.0)
    assert seq.meta["source"] == "from_solution"


def test_prepare_unknown_analytic_target_raises(dev):
    with pytest.raises(NotImplementedError):
        prepare(dev, Fock(3))


# -- swept phase (compiler support added for the parity fringe) ---------------------------


def test_parity_fringe_sequence_has_swept_phase(dev):
    exp = ParityFringe(dev, phases=Sweep("phase", 0.0, 360.0, 61))
    seq, _ = exp.sequence()
    pulses = [op for op in seq.ops if isinstance(op, Pulse)]
    assert isinstance(pulses[-1].phase_deg, Sweep)
    kinds = [type(op).__name__ for op in seq.ops]
    assert kinds == ["Pulse", "Delay", "Pulse", "Measure"]
    assert isinstance(seq.ops[1], Delay)
