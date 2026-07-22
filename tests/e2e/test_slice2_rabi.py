"""Slice 2 end-to-end — amplitude Rabi (plan Task 11).

Structural gate: the AmplitudeRabi sequence compiles into a swept program (add_loop + swept
add_pulse) with the real calibrated ge_pi. Analysis gate: analyze() reduces + fits a synthetic
Rabi trace and recovers the π gain. The full construct+acquire path needs a tProc-v2 soccfg and
is gated (tests/_v2gate.py).
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))
from _recording import RecordingProgram  # noqa: E402
from _v2gate import V2_SOCCFG, requires_v2_soccfg  # noqa: E402

from strumento.core.compiler import Compiler  # noqa: E402
from strumento.core.device import Device  # noqa: E402
from strumento.core.result import MeasurementResult  # noqa: E402
from strumento.packs.cqed.experiments.amplitude_rabi import AmplitudeRabi  # noqa: E402

YAML = Path(__file__).parents[2] / "devices" / "multimode_demo" / "device.yaml"
LOOPBACK_YAML = Path(__file__).parents[2] / "devices" / "loopback_demo" / "device.yaml"


@pytest.fixture
def dev():
    return Device.load(YAML)


def test_rabi_sequence_compiles_to_a_swept_program(dev):
    exp = AmplitudeRabi(dev)  # default gain sweep 0..30000, 51 pts
    seq, sweeps = exp.sequence()
    prog = RecordingProgram()
    c = Compiler(prog, dev.wiring)
    c.declare(seq)
    c.play(seq)

    # one amplitude loop over 51 points
    assert ("add_loop", "amp", 51) in prog.calls
    # the ge_pi pulse's gain is the swept handle (a fraction sweep) on the qubit drive DAC
    add_pulse = next(c for c in prog.calls if c[0] == "add_pulse")
    assert add_pulse[1] == 2  # qubit drive DAC
    assert add_pulse[3]["gain"]["__sweep__"] == "amp"
    # and the readout is triggered
    assert any(c[0] == "trigger" for c in prog.calls)


def test_rabi_analyze_recovers_pi_gain(dev):
    # synthetic averaged-IQ Rabi trace vs the lab-native gain axis
    gains = np.linspace(0, 30000, 51)
    true_period = 20000.0
    avgi = 0.5 + 0.4 * np.cos(2 * np.pi * gains / true_period)
    result = MeasurementResult(xpts=gains, avgi=avgi, avgq=np.zeros_like(gains))

    fit = AmplitudeRabi(dev).analyze(result)
    assert fit.success
    assert fit.derived["pi_gain"] == pytest.approx(true_period / 2, rel=0.05)  # 10000


@requires_v2_soccfg
def test_rabi_full_go_on_mock():
    dev = Device.load(LOOPBACK_YAML)
    from strumento.core.mock import MockQickSocV2

    soc = MockQickSocV2.from_snapshot(V2_SOCCFG)
    outcome = AmplitudeRabi(dev).go(soc, reps=200)
    assert outcome.result.xpts is not None
    assert np.asarray(outcome.result.xpts)[-1] == pytest.approx(30000)  # lab-native axis
    assert outcome.fit is not None
