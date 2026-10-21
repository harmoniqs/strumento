"""T1 + T2-Ramsey experiments: sequence shape + analyzer fit recovery (pure)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from strumento.core.device import Device
from strumento.core.result import MeasurementResult
from strumento.core.sweeps import Sweep
from strumento.packs.cqed.experiments.t1 import T1
from strumento.packs.cqed.experiments.t2_ramsey import T2Ramsey
from tests._paths import DEVICES

# -- T1 ---------------------------------------------------------------------------------


def test_t1_sequence_is_pi_wait_measure(dev):
    exp = T1(dev, delays=Sweep(start=0.0, stop=300.0, on="t"), points=61)
    seq, sweeps = exp.sequence()
    kinds = [type(op).__name__ for op in seq.ops]
    # issue #102: a readout probe now precedes the Measure (dev.readout.dispersive_measure()).
    assert kinds == ["Pulse", "Delay", "Pulse", "Measure"]
    assert isinstance(seq.ops[1].duration_us, Sweep)
    assert seq.ops[1].duration_us.on == "t"
    assert sweeps == ()  # the sweep travels in the Delay; the compiler discovers it


def test_t1_analyze_recovers_decay_constant(dev):
    t = np.linspace(0.0, 300.0, 61)
    true_t1 = 85.0
    signal = 0.4 + 1.7 * np.exp(-t / true_t1)
    result = MeasurementResult(xpts=t, avgi=signal, avgq=np.zeros_like(t))
    fit = T1(dev).analyze(result)
    assert fit.success
    assert fit.derived["T1"] == pytest.approx(true_t1, rel=0.05)


# -- T2 Ramsey --------------------------------------------------------------------------


def test_t2_sequence_is_hpi_wait_hpi_measure(dev):
    exp = T2Ramsey(dev, delays=Sweep(start=0.0, stop=40.0, on="t"), points=101,
                   advance_phase_deg=0.0)
    seq, _ = exp.sequence()
    kinds = [type(op).__name__ for op in seq.ops]
    # issue #102: a readout probe now precedes the Measure (dev.readout.dispersive_measure()).
    assert kinds == ["Pulse", "Delay", "Pulse", "Pulse", "Measure"]
    hpi_gain = dev.calib.qubit.pulses["hpi_ge"].gain / 32766
    assert seq.ops[0].gain == pytest.approx(hpi_gain)
    assert seq.ops[2].gain == pytest.approx(hpi_gain)


def test_t2_ramsey_detuning_offsets_both_half_pi_drives(dev):
    # A nonzero ramsey_freq_mhz is a real artificial detuning: both half-pi pulses drive that far
    # off the calibrated f_ge, so the Ramsey fringe oscillates at that known frequency (on
    # resonance there is no fringe, only the T2* envelope).
    f_r = 0.25
    exp = T2Ramsey(dev, delays=Sweep(start=0.0, stop=40.0, on="t"), points=101, ramsey_freq_mhz=f_r)
    assert exp.ramsey_freq_mhz == pytest.approx(f_r)
    hpis = [op for op in exp.sequence()[0].ops
            if type(op).__name__ == "Pulse" and getattr(op, "label", None) == "hpi_ge"]
    assert len(hpis) == 2
    expected = dev.calib.qubit.f_ge + f_r
    assert hpis[0].freq_mhz == pytest.approx(expected)
    assert hpis[1].freq_mhz == pytest.approx(expected)


def test_t2_ramsey_on_resonance_reproduces_the_bare_half_pi_drive(dev):
    # ramsey_freq_mhz=0 (default) must reproduce the on-resonance drive exactly (backward compat).
    exp = T2Ramsey(dev, delays=Sweep(start=0.0, stop=40.0, on="t"), points=101)
    hpis = [op for op in exp.sequence()[0].ops
            if type(op).__name__ == "Pulse" and getattr(op, "label", None) == "hpi_ge"]
    assert all(op.freq_mhz == pytest.approx(dev.calib.qubit.f_ge) for op in hpis)


def test_t2_analyze_recovers_t2_and_detuning(dev):
    t = np.linspace(0.0, 40.0, 201)
    true_t2, f_mhz = 18.0, 0.35
    signal = 0.5 + 0.45 * np.exp(-t / true_t2) * np.cos(2 * np.pi * f_mhz * t + 0.3)
    result = MeasurementResult(xpts=t, avgi=signal, avgq=np.zeros_like(t))
    fit = T2Ramsey(dev).analyze(result)
    assert fit.success
    assert fit.derived["T2"] == pytest.approx(true_t2, rel=0.10)
    assert fit.derived["detuning"] == pytest.approx(f_mhz, rel=0.05)


# -- full go() on the mock (real v2 assembly: swept delay -> add_loop + swept t) ---------


@pytest.mark.qick
def test_t1_full_go_on_mock():
    import sys

    from strumento.core.mock import MockQickSocV2
    sys.path.insert(0, str(Path(__file__).parents[2]))
    from _v2gate import V2_SOCCFG

    dev = Device.load(DEVICES / "loopback_demo" / "device.yaml")
    soc = MockQickSocV2.from_snapshot(V2_SOCCFG)
    # loopback_demo has no dedicated readout line (single-generator instance) — probe on the
    # qubit line instead, same as resonator_spectroscopy's own test override (issue #102).
    outcome = T1(dev, delays=Sweep(start=0.0, stop=100.0, on="t"), points=11,
                 line=("qubit", "drive")).go(
        soc, reps=10, analyze=False
    )
    assert outcome.result.xpts is not None
    # the axis is AS-PLAYED (D37): 10 µs steps are 4300.8 cycles at the testbench's 430.08 MHz
    # tProc clock, and qick truncates the register step to 4300 cycles = 9.99814 µs, so ten
    # steps top at 99.98140 µs — the requested 100.0 was never playable. Numeric golden
    # against the committed snapshot.
    assert np.asarray(outcome.result.xpts)[-1] == pytest.approx(99.98139880952381)


@pytest.mark.qick
def test_t1_compiled_dump_is_json_serializable():
    """A swept delay must not leave tuple mapping keys in saved program provenance."""
    import sys

    from strumento.core.mock import MockQickSocV2
    from strumento.core.program import StrumentoProgram

    sys.path.insert(0, str(Path(__file__).parents[2]))
    from _v2gate import V2_SOCCFG

    dev = Device.load(DEVICES / "loopback_demo" / "device.yaml")
    soc = MockQickSocV2.from_snapshot(V2_SOCCFG)
    exp = T1(
        dev,
        delays=Sweep(start=0.0, stop=100.0, on="t"),
        points=11,
        line=("qubit", "drive"),
    )
    sequence, _ = exp.sequence()
    program = StrumentoProgram(dev, seq=sequence, axes=exp.axes(), reps=10, soc=soc)

    json.dumps(program.dump(), default=str)
