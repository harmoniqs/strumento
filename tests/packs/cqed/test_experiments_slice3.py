"""Slice 3 — T1 + T2-Ramsey experiments: sequence shape + analyzer fit recovery (pure)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from strumento.core.device import Device
from strumento.core.result import MeasurementResult
from strumento.core.sweeps import Sweep
from strumento.packs.cqed.experiments.t1 import T1
from strumento.packs.cqed.experiments.t2_ramsey import T2Ramsey

YAML = Path(__file__).parents[3] / "devices" / "multimode_demo" / "device.yaml"


@pytest.fixture
def dev():
    return Device.load(YAML)


# -- T1 ---------------------------------------------------------------------------------


def test_t1_sequence_is_pi_wait_measure(dev):
    exp = T1(dev, delays=Sweep("t", 0.0, 300.0, 61))
    seq, sweeps = exp.sequence()
    kinds = [type(op).__name__ for op in seq.ops]
    assert kinds == ["Pulse", "Delay", "Measure"]
    assert isinstance(seq.ops[1].duration_us, Sweep)
    assert seq.ops[1].duration_us.name == "t"
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
    exp = T2Ramsey(dev, delays=Sweep("t", 0.0, 40.0, 101), advance_phase_deg=0.0)
    seq, _ = exp.sequence()
    kinds = [type(op).__name__ for op in seq.ops]
    assert kinds == ["Pulse", "Delay", "Pulse", "Measure"]
    hpi_gain = dev.calib.qubit.pulses["hpi_ge"].gain / 32766
    assert seq.ops[0].gain == pytest.approx(hpi_gain)
    assert seq.ops[2].gain == pytest.approx(hpi_gain)


def test_t2_ramsey_detuning_drives_second_hpi_phase(dev):
    # a software detuning advances the second half-pi phase per delay: not expressible as a
    # static phase -> the experiment records it for analysis instead (ramsey_freq_mhz)
    exp = T2Ramsey(dev, delays=Sweep("t", 0.0, 40.0, 101), ramsey_freq_mhz=0.25)
    assert exp.ramsey_freq_mhz == pytest.approx(0.25)


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

    dev = Device.load(Path(__file__).parents[3] / "devices" / "loopback_demo" / "device.yaml")
    soc = MockQickSocV2.from_snapshot(V2_SOCCFG)
    outcome = T1(dev, delays=Sweep("t", 0.0, 100.0, 11)).go(soc, reps=10, analyze=False)
    assert outcome.result.xpts is not None
    assert np.asarray(outcome.result.xpts)[-1] == pytest.approx(100.0)
