"""Compiler sweep lowering (plan Task 11, spec §6.4).

Pure: verified against the shared RecordingProgram fake — no qick.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))
from _recording import RecordingProgram  # noqa: E402

from strumento.core.compiler import Compiler  # noqa: E402
from strumento.core.pulses import Gaussian, Pulse, Seq  # noqa: E402
from strumento.core.sweeps import Sweep  # noqa: E402
from strumento.core.wiring import Generator, Line, LineRef, ReadoutWiring, Wiring  # noqa: E402


def _wiring():
    return Wiring(
        lines={
            "qubit": Line(name="qubit", generators=[Generator(role="drive", dac_ch=2, nyquist=1)]),
            "coupler": Line(name="coupler", generators=[
                Generator(role="flux", dac_ch=1, band_hz=(0.0, 1800e6)),
                Generator(role="flux", dac_ch=1, band_hz=(1800e6, 1e12)),
            ]),
        },
        readout=ReadoutWiring(dac_ch=5, adc_chs=[0]),
    )


def _gain_swept_pulse(sweep):
    return Pulse(line=LineRef("qubit", "drive"), freq_mhz=3562.6, gain=sweep,
                 envelope=Gaussian(sigma_us=0.035))


def test_gain_sweep_registers_add_loop_once_and_passes_sweep_value():
    prog = RecordingProgram()
    c = Compiler(prog, _wiring())
    c.declare(Seq().play(_gain_swept_pulse(Sweep("amp", 0.0, 0.9, 51))))

    loops = [c for c in prog.calls if c[0] == "add_loop"]
    assert loops == [("add_loop", "amp", 51)]  # exactly one loop
    # the pulse's gain is the swept handle, not a scalar
    add_pulse = next(c for c in prog.calls if c[0] == "add_pulse")
    assert add_pulse[3]["gain"] == {"__sweep__": "amp", "start": 0.0, "stop": 0.9}


def test_two_pulses_same_sweep_declare_loop_once():
    prog = RecordingProgram()
    c = Compiler(prog, _wiring())
    sw = Sweep("amp", 0.0, 0.9, 51)
    c.declare(Seq().play(_gain_swept_pulse(sw)).play(_gain_swept_pulse(sw)))
    assert sum(1 for c in prog.calls if c[0] == "add_loop") == 1


def test_freq_sweep_within_band_ok():
    prog = RecordingProgram()
    c = Compiler(prog, _wiring())
    p = Pulse(line=LineRef("coupler", "flux"), freq_mhz=Sweep("f", 100.0, 900.0, 21),
              gain=0.1, envelope=Gaussian(0.05))
    c.declare(Seq().play(p))  # both endpoints resolve to DAC 1 (low band) — no error
    assert ("add_loop", "f", 21) in prog.calls


def test_freq_sweep_across_band_boundary_raises():
    prog = RecordingProgram()
    c = Compiler(prog, _wiring())
    # 100 MHz -> low-band generator, 2500 MHz -> high-band generator: two different bands
    p = Pulse(line=LineRef("coupler", "flux"), freq_mhz=Sweep("f", 100.0, 2500.0, 21),
              gain=0.1, envelope=Gaussian(0.05))
    with pytest.raises(ValueError, match="band boundary"):
        c.declare(Seq().play(p))
