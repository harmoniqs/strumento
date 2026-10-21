"""Tests for the compiler (D22 envelope lowering, D23 gain fractions).

The compiler *logic* — walk the IR, resolve generators, dedup envelopes, emit the program build
calls in order — is exercised against a **RecordingProgram** fake, so it needs no qick and no
tProc-v2 firmware. Realized-value readback lives on ``StrumentoProgram.realized`` and is tested
in ``test_program.py``.
"""

import pytest

from strumento.core.compiler import Compiler
from strumento.core.pulses import Gaussian, Pulse, Seq
from strumento.core.wiring import Generator, Line, LineRef, ReadoutWiring, Wiring


class RecordingProgram:
    """A fake program that records the build calls the compiler makes (no qick)."""

    def __init__(self):
        self.calls = []

    def declare_gen(self, ch, **kw):        self.calls.append(("declare_gen", ch, kw))
    def add_gauss(self, ch, name, **kw):    self.calls.append(("add_gauss", ch, name, kw))
    def add_pulse(self, ch, name, **kw):    self.calls.append(("add_pulse", ch, name, kw))
    def add_envelope(self, ch, name, **kw): self.calls.append(("add_envelope", ch, name, kw))
    def pulse(self, ch, name, t=0, **kw):   self.calls.append(("pulse", ch, name, t))
    def delay(self, t, **kw):               self.calls.append(("delay", t))
    def delay_auto(self, t=0, **kw):        self.calls.append(("delay_auto", t))
    def read_and_jump(self, ro_ch=None, component=None, threshold=None, test=None, label=None, **kw):
        self.calls.append(("read_and_jump", ro_ch, component, threshold, test, label))
    def jump(self, label, **kw):            self.calls.append(("jump", label))
    def label(self, label, **kw):           self.calls.append(("label", label))
    def open_loop(self, n, name=None, **kw): self.calls.append(("open_loop", n, name))
    def close_loop(self, **kw):             self.calls.append(("close_loop",))
    def trigger(self, ros=None, t=0, **kw): self.calls.append(("trigger", tuple(ros or ()), t))
    def declare_readout(self, ch, length, **kw): self.calls.append(("declare_readout", ch, length))

    def kinds(self):
        return [c[0] for c in self.calls]


def _wiring():
    return Wiring(
        lines={"qubit": Line(name="qubit", generators=[
            Generator(role="drive", dac_ch=2, nyquist=1),
        ])},
        readout=ReadoutWiring(dac_ch=5, adc_chs=[0]),
    )


def _pulse(gain=0.1, freq=3562.6, sigma=0.035):
    return Pulse(line=LineRef("qubit", "drive"), freq_mhz=freq, gain=gain,
                 envelope=Gaussian(sigma_us=sigma))


def test_lower_seq_emits_gen_envelope_pulse_and_trigger():
    prog = RecordingProgram()
    c = Compiler(prog, _wiring())
    seq = Seq().play(_pulse()).measure()
    c.declare(seq)
    c.play(seq)
    kinds = prog.kinds()
    assert "declare_gen" in kinds
    assert "add_gauss" in kinds        # the Gaussian envelope was lowered
    assert "add_pulse" in kinds
    assert kinds.index("pulse") < kinds.index("trigger")   # play order: pulse then readout
    # the pulse carries the fraction gain unchanged (D23: no int->frac in the compiler)
    add_pulse = next(call for call in prog.calls if call[0] == "add_pulse")
    assert add_pulse[3]["gain"] == pytest.approx(0.1)
    assert add_pulse[3]["freq"] == pytest.approx(3562.6)


def test_measure_triggers_at_the_compiler_supplied_trig_offset():
    """The Measure branch's trigger uses whatever trig_offset the Compiler was built with
    (#98), not a hardcoded 0 — StrumentoProgram resolves calib/wiring and passes it in; the
    compiler itself just plays what it is given (Prior Art: the ``t=0`` literal one line below
    where ``self.wiring.readout.adc_chs[0]`` is already read)."""
    prog = RecordingProgram()
    c = Compiler(prog, _wiring(), trig_offset=0.45)
    seq = Seq().play(_pulse()).measure()
    c.declare(seq)
    c.play(seq)
    trigger = next(call for call in prog.calls if call[0] == "trigger")
    assert trigger == ("trigger", (0,), 0.45)


def test_measure_trigger_offset_defaults_to_zero():
    """No trig_offset supplied to the Compiler -> the trigger still fires at t=0 (AC3: no
    regression for callers that don't pass one, e.g. every pre-existing Compiler(...) call)."""
    prog = RecordingProgram()
    c = Compiler(prog, _wiring())
    seq = Seq().play(_pulse()).measure()
    c.declare(seq)
    c.play(seq)
    trigger = next(call for call in prog.calls if call[0] == "trigger")
    assert trigger == ("trigger", (0,), 0)


def test_equal_envelope_on_same_gen_is_added_once():
    prog = RecordingProgram()
    c = Compiler(prog, _wiring())
    # two pulses, identical Gaussian envelope, same generator
    seq = Seq().play(_pulse(gain=0.1)).play(_pulse(gain=0.2)).measure()
    c.declare(seq)
    n_gauss = sum(1 for k in prog.kinds() if k == "add_gauss")
    assert n_gauss == 1                       # deduped on (gen, envelope) — R6
    n_gen = sum(1 for k in prog.kinds() if k == "declare_gen")
    assert n_gen == 1                         # gen declared once


def test_unregistered_envelope_raises_clear_error():
    from dataclasses import dataclass

    @dataclass(frozen=True)
    class Weird:
        x: float

    prog = RecordingProgram()
    c = Compiler(prog, _wiring())
    seq = Seq().play(Pulse(line=LineRef("qubit", "drive"), freq_mhz=3562.6, gain=0.1,
                           envelope=Weird(1.0)))
    with pytest.raises(TypeError, match="no lowering registered"):
        c.declare(seq)


def test_delay_and_raw_lower():
    prog = RecordingProgram()
    c = Compiler(prog, _wiring())
    marker = []
    seq = Seq().play(_pulse()).wait(0.5).raw(lambda p: marker.append(True))
    c.declare(seq)
    c.play(seq)
    assert ("delay", 0.5) in prog.calls
    assert marker == [True]              # Raw.fn was invoked with the program


def test_shared_dac_with_conflicting_nyquist_raises():
    # dedup on dac_ch alone must not silently first-win when nyquist zones differ
    from strumento.core.wiring import Generator, Line, ReadoutWiring, Wiring

    wiring = Wiring(
        lines={
            "a": Line(name="a", generators=[Generator(role="main", dac_ch=0, nyquist=1)]),
            "b": Line(name="b", generators=[Generator(role="main", dac_ch=0, nyquist=2)]),
        },
        readout=ReadoutWiring(dac_ch=0, adc_chs=[0]),
    )
    prog = RecordingProgram()
    c = Compiler(prog, wiring)
    seq = (
        Seq()
        .play(Pulse(line=LineRef("a", "main"), freq_mhz=1000.0, gain=0.1, envelope=Gaussian(sigma_us=0.03)))
        .play(Pulse(line=LineRef("b", "main"), freq_mhz=5000.0, gain=0.1, envelope=Gaussian(sigma_us=0.03)))
    )
    with pytest.raises(ValueError, match="Nyquist"):
        c.declare(seq)


# -- timeline cursor model (docs/timeline.md) --------------------------------------------


def test_sequential_pulses_advance_the_cursor():
    """play(a).play(b) with no wait: each sequential pulse is followed by a duration-aware
    delay_auto, so b plays after a (not overlapping at t=0)."""
    wiring = _wiring()
    prog = RecordingProgram()
    c = Compiler(prog, wiring)
    p = Pulse(line=LineRef("qubit", "drive"), freq_mhz=3562.6, gain=0.1, envelope=Gaussian(sigma_us=0.03))
    seq = Seq().play(p).play(p)
    c.declare(seq)
    c.play(seq)
    timeline = [k for k in prog.kinds() if k in ("pulse", "delay", "delay_auto", "trigger")]
    # pulse, delay_auto, pulse, delay_auto — each play advances the cursor past itself
    assert timeline == ["pulse", "delay_auto", "pulse", "delay_auto"]


def test_absolute_start_us_does_not_advance_cursor():
    wiring = _wiring()
    prog = RecordingProgram()
    c = Compiler(prog, wiring)
    p = Pulse(line=LineRef("qubit", "drive"), freq_mhz=3562.6, gain=0.1,
              envelope=Gaussian(sigma_us=0.03), start_us=1.5)
    seq = Seq().play(p)
    c.declare(seq)
    c.play(seq)
    timeline = [k for k in prog.kinds() if k in ("pulse", "delay", "delay_auto", "trigger")]
    assert timeline == ["pulse"]                           # absolute placement opts out of auto-advance
    assert next(c for c in prog.calls if c[0] == "pulse")[3] == 1.5   # played at t=start_us


def test_segment_advances_once_past_the_longest():
    wiring = _wiring()
    prog = RecordingProgram()
    c = Compiler(prog, wiring)
    p1 = Pulse(line=LineRef("qubit", "drive"), freq_mhz=3562.6, gain=0.1, envelope=Gaussian(sigma_us=0.03))
    p2 = Pulse(line=LineRef("qubit", "drive"), freq_mhz=3600.0, gain=0.1, envelope=Gaussian(sigma_us=0.05))
    seq = Seq().play_parallel([p1, p2])
    c.declare(seq)
    c.play(seq)
    timeline = [k for k in prog.kinds() if k in ("pulse", "delay", "delay_auto", "trigger")]
    assert timeline.count("pulse") == 2                    # both pulses at the same cursor
    assert timeline.count("delay_auto") == 1               # one advance for the whole segment
    assert timeline == ["pulse", "pulse", "delay_auto"]
