"""Parity composite/fringe + from_solution OC import (pure)."""

from __future__ import annotations

import numpy as np
import pytest

from strumento.core.fitters import fit_parity
from strumento.core.pulses import Arb, Delay, Pulse, Segment, from_solution
from strumento.core.result import MeasurementResult
from strumento.core.sweeps import Sweep
from strumento.core.wiring import LineRef
from strumento.packs.cqed.calibration import PiPulseReference
from strumento.packs.cqed.experiments.parity_fringe import ParityFringe
from strumento.packs.cqed.library.composite import Fock, parity, prepare

# -- parity -----------------------------------------------------------------------------


def test_parity_composite_shape(dev):
    """The bare state-mapping shape, via measure=False (#100: measure=True is the default now --
    see test_parity_composite_measures_by_default)."""
    seq = parity(dev, "cavity", measure=False)
    kinds = [type(op).__name__ for op in seq.ops]
    assert kinds == ["Pulse", "Delay", "Pulse"]
    assert seq.ops[1].duration_us == pytest.approx(1.6)      # calibrated revival
    assert seq.ops[2].phase_deg == pytest.approx(180.0)
    fast = parity(dev, "cavity", fast=True, measure=False)
    assert fast.ops[1].duration_us == pytest.approx(0.8)


def test_parity_composite_measures_by_default(dev):
    """#100: parity() drives the readout probe and measures internally by default, instead of
    leaving `.measure()` to the caller -- every parity()-based experiment now reads out through a
    driven resonator without having to remember dispersive_measure() itself. The state-mapping
    ops (the first three) are unchanged from the measure=False shape -- only the tail changes."""
    seq = parity(dev, "cavity")
    kinds = [type(op).__name__ for op in seq.ops]
    assert kinds == ["Pulse", "Delay", "Pulse", "Pulse", "Measure"]
    assert seq.ops[1].duration_us == pytest.approx(1.6)
    assert seq.ops[2].phase_deg == pytest.approx(180.0)
    assert seq.ops[3].label == "ro_probe"


def test_parity_fast_applies_ac_stark_phase(dev):
    """Fast parity advances the closing half-pi by 360·f_AC·t_rev, on the fast path only.

    The demo fixture leaves revival_stark_shift None, so set it at runtime; the wait is the fast
    revival (revival_time[1]). Slow path is covered by the companion test below.
    """
    mode_cal = dev.calib.modes["cavity"]
    mode_cal.revival_stark_shift = 0.5  # MHz
    t_fast = mode_cal.revival_time[1]
    seq = parity(dev, "cavity", fast=True, measure=False)
    expected = (180.0 + 360.0 * 0.5 * t_fast) % 360.0
    assert seq.ops[2].phase_deg == pytest.approx(expected)


def test_parity_slow_ignores_ac_stark_phase(dev):
    """The slow path applies NO Stark correction even when revival_stark_shift is set (the slow
    revival is fit directly and already absorbs static shifts)."""
    dev.calib.modes["cavity"].revival_stark_shift = 0.5
    seq = parity(dev, "cavity", measure=False)  # fast=False
    assert seq.ops[2].phase_deg == pytest.approx(180.0)


def test_parity_fast_offsets_swept_phase(dev):
    """A swept second_phase is offset by the same constant, preserving the axis binding."""
    mode_cal = dev.calib.modes["cavity"]
    mode_cal.revival_stark_shift = 0.5
    stark = 360.0 * 0.5 * mode_cal.revival_time[1]
    seq = parity(dev, "cavity", fast=True,
                 second_phase=Sweep(start=0.0, stop=360.0, on="phase"), measure=False)
    ph = seq.ops[2].phase_deg
    assert isinstance(ph, Sweep)
    assert ph.on == "phase"
    assert ph.start == pytest.approx(stark)
    assert ph.stop == pytest.approx(360.0 + stark)


def test_parity_fringe_analyze_recovers_contrast(dev):
    ph = np.linspace(0.0, 360.0, 121)
    true_contrast, true_phase = 0.8, 30.0
    y = 0.5 + (true_contrast / 2) * np.cos(np.radians(ph + true_phase))
    fit = ParityFringe(dev).analyze(MeasurementResult(xpts=ph, avgi=y, avgq=np.zeros_like(y)))
    assert fit.success
    assert fit.derived["contrast"] == pytest.approx(true_contrast, rel=0.05)
    # no blob centres on the default readout -> signal-space fit only, no normalization derived
    assert "alpha_scale" not in fit.derived


@pytest.mark.parametrize("phi0, expected_sign", [(180.0, 1.0), (0.0, -1.0)])
def test_parity_fringe_population_path_derives_alpha_scale_and_sign(dev, phi0, expected_sign):
    """With blob centres the fringe fits on P_e and derives alpha_scale/parity_sign/pe_midpoint.

    Build a synthetic averaged-IQ fringe whose population is P_e(phi) = 0.5 - (C/2)cos(phi - phi0):
    the reconstruction samples parity at the composite operating phase (180 deg), so phi0=180 puts
    vacuum at low P_e there (even->low, parity_sign=+1) and phi0=0 puts it high (parity_sign=-1).
    """
    from strumento.packs.cqed.readout import DispersiveIQ

    g, e = np.array([0.0, 0.0]), np.array([1.0, 0.0])  # trivial ruler: rotated-I in [0,1] IS P_e
    C = 0.9
    ph = np.linspace(0.0, 360.0, 181)
    pe = 0.5 - (C / 2) * np.cos(np.radians(ph - phi0))
    # invert to averaged IQ on the g->e segment: I = P_e (angle 0), Q = 0
    avgi, avgq = pe, np.zeros_like(pe)
    readout = DispersiveIQ(threshold=0.0, angle=0.0, g_center=tuple(g), e_center=tuple(e))
    fit = ParityFringe(dev, readout=readout).analyze(
        MeasurementResult(xpts=ph, avgi=avgi, avgq=avgq)
    )
    assert fit.success
    assert fit.derived["alpha_scale"] == pytest.approx(C, rel=0.05)
    assert fit.derived["parity_sign"] == expected_sign
    assert fit.derived["pe_midpoint"] == pytest.approx(0.5, abs=0.02)


def test_fit_parity_standalone():
    ph = np.linspace(0.0, 360.0, 61)
    y = 1.0 + 0.3 * np.cos(np.radians(ph))
    fit = fit_parity(ph, y)
    assert fit.derived["contrast"] == pytest.approx(0.6, rel=0.05)


# -- from_solution ------------------------------------------------------------------------


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
        "cavity": (("cavity", "main"), 5200.0),      # bare tuple form also accepted
    }


def test_from_solution_builds_parallel_arb_segment():
    ref = PiPulseReference(pi_gain_frac=0.25, pi_rabi_mhz=8.0)
    seq = from_solution(_toy_solution(), drive_map=_drive_map(), gain_calib=ref)
    assert len(seq.ops) == 1 and isinstance(seq.ops[0], Segment)
    seg = seq.ops[0]
    assert {p.line for p in seg.pulses} == {LineRef("qubit", "drive"), LineRef("cavity", "main")}
    by_line = {p.line.line: p for p in seg.pulses}
    q = by_line["qubit"]
    assert isinstance(q.envelope, Arb) and q.envelope.qdata is None
    assert np.max(np.abs(q.envelope.idata)) == pytest.approx(1.0)     # normalized samples
    assert q.gain == pytest.approx(ref(4.0))                           # peak -> fraction
    c = by_line["cavity"]
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
        prepare(dev, "cat_state")  # non-Fock target is not implemented


# -- swept phase (compiler support added for the parity fringe) ---------------------------


def test_parity_fringe_sequence_has_swept_phase(dev):
    exp = ParityFringe(dev, phases=Sweep(start=0.0, stop=360.0, on="phase"), points=61)
    seq, _ = exp.sequence()
    kinds = [type(op).__name__ for op in seq.ops]
    # #100: parity()'s own measure=True default now supplies the readout tail; ParityFringe no
    # longer chains its own .measure() (that would double-measure).
    assert kinds == ["Pulse", "Delay", "Pulse", "Pulse", "Measure"]
    assert isinstance(seq.ops[1], Delay)
    assert isinstance(seq.ops[2].phase_deg, Sweep)      # the swept second half-pi
    assert seq.ops[3].label == "ro_probe"               # drives the readout before measuring
