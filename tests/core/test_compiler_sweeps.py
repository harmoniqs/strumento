"""Compiler sweep lowering: how swept pulse parameters become program loops (D36).

Pure: verified against the shared RecordingProgram fake — no qick.
"""

import sys
import warnings
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))
from _recording import RecordingProgram  # noqa: E402

from strumento.core.compiler import Compiler  # noqa: E402
from strumento.core.pulses import Gaussian, Pulse, Seq  # noqa: E402
from strumento.core.sweeps import Loop, Sweep  # noqa: E402
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
    c = Compiler(prog, _wiring(), axes=[Loop("amp", 51)])
    c.declare(Seq().play(_gain_swept_pulse(Sweep(start=0.0, stop=0.9, on="amp"))))

    loops = [c for c in prog.calls if c[0] == "add_loop"]
    assert loops == [("add_loop", "amp", 51)]  # exactly one loop
    # the pulse's gain is the swept handle, not a scalar
    add_pulse = next(c for c in prog.calls if c[0] == "add_pulse")
    assert add_pulse[3]["gain"] == {"__sweep__": "amp", "start": 0.0, "stop": 0.9}


def test_two_pulses_same_sweep_declare_loop_once():
    # the legacy per-name form on purpose: its auto-declared loop must still be emitted once
    prog = RecordingProgram()
    c = Compiler(prog, _wiring())
    with pytest.warns(DeprecationWarning, match="legacy per-name Sweep"):
        sw = Sweep("amp", 0.0, 0.9, 51)
    c.declare(Seq().play(_gain_swept_pulse(sw)).play(_gain_swept_pulse(sw)))
    assert sum(1 for c in prog.calls if c[0] == "add_loop") == 1


def test_freq_sweep_within_band_ok():
    prog = RecordingProgram()
    c = Compiler(prog, _wiring(), axes=[Loop("f", 21)])
    p = Pulse(line=LineRef("coupler", "flux"), freq_mhz=Sweep(start=100.0, stop=900.0, on="f"),
              gain=0.1, envelope=Gaussian(0.05))
    c.declare(Seq().play(p))  # both endpoints resolve to DAC 1 (low band) — no error
    assert ("add_loop", "f", 21) in prog.calls


def test_freq_sweep_across_band_boundary_raises():
    prog = RecordingProgram()
    c = Compiler(prog, _wiring(), axes=[Loop("f", 21)])
    # 100 MHz -> low-band generator, 2500 MHz -> high-band generator: two different bands
    p = Pulse(line=LineRef("coupler", "flux"),
              freq_mhz=Sweep(start=100.0, stop=2500.0, on="f"),
              gain=0.1, envelope=Gaussian(0.05))
    with pytest.raises(ValueError, match="band boundary"):
        c.declare(Seq().play(p))


# --- declared loop axes (D36) ---------------------------------------------------


def _joint_pulse(gain_sweep, freq_sweep):
    return Pulse(line=LineRef("qubit", "drive"), freq_mhz=freq_sweep, gain=gain_sweep,
                 envelope=Gaussian(sigma_us=0.035))


def test_declared_axis_joint_sweep_emits_one_loop():
    """The D36 headline: two parameters bound to ONE declared axis step together in a single
    loop — not an outer product. The compiler declares loops only from declared axes."""
    prog = RecordingProgram()
    c = Compiler(prog, _wiring(), axes=[Loop("ax", 51)])
    g = Sweep(start=0.0, stop=0.9, on="ax")
    f = Sweep(start=3550.0, stop=3575.0, on="ax")
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        c.declare(Seq().play(_joint_pulse(g, f)))
    assert not [w for w in record if issubclass(w.category, DeprecationWarning)]

    loops = [c_ for c_ in prog.calls if c_[0] == "add_loop"]
    assert loops == [("add_loop", "ax", 51)]  # ONE loop for TWO swept parameters
    add_pulse = next(c_ for c_ in prog.calls if c_[0] == "add_pulse")
    assert add_pulse[3]["gain"] == {"__sweep__": "ax", "start": 0.0, "stop": 0.9}
    assert add_pulse[3]["freq"] == {"__sweep__": "ax", "start": 3550.0, "stop": 3575.0}


def test_bound_sweep_on_undeclared_axis_raises():
    prog = RecordingProgram()
    c = Compiler(prog, _wiring())  # no axes declared
    p = _joint_pulse(Sweep(start=0.0, stop=0.9, on="nope"), 3562.6)
    with pytest.raises(ValueError, match="undeclared axis"):
        c.declare(Seq().play(p))


def test_legacy_sweep_still_works_but_warns():
    prog = RecordingProgram()
    c = Compiler(prog, _wiring())
    with pytest.warns(DeprecationWarning, match="legacy per-name Sweep"):
        sweep = Sweep("amp", 0.0, 0.9, 51)  # the deprecation fires where the sweep is written
    c.declare(Seq().play(_gain_swept_pulse(sweep)))
    assert ("add_loop", "amp", 51) in prog.calls


def test_compiling_a_legacy_sweep_does_not_warn_a_second_time():
    # #162: the deprecation fires once, where the Sweep is written — not again at compile time
    with pytest.warns(DeprecationWarning, match="legacy per-name Sweep"):
        sweep = Sweep("amp", 0.0, 0.9, 51)
    prog = RecordingProgram()
    c = Compiler(prog, _wiring())
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        c.declare(Seq().play(_gain_swept_pulse(sweep)))
    assert not [w for w in record if issubclass(w.category, DeprecationWarning)]
    # the legacy sweep still auto-declares its per-name loop, exactly as before
    assert [c_ for c_ in prog.calls if c_[0] == "add_loop"] == [("add_loop", "amp", 51)]


def test_declared_but_unused_axis_warns():
    prog = RecordingProgram()
    c = Compiler(prog, _wiring(), axes=[Loop("idle", 10)])
    with pytest.warns(UserWarning, match="no sweeps bound"):
        c.declare(Seq().play(_gain_swept_pulse(0.5)))
    # the declared axis is still emitted — the user said "this is a loop axis"
    assert ("add_loop", "idle", 10) in prog.calls
