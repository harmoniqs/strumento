"""Tests for the compiler (plan Task 8, D22/D23).

The compiler *logic* — walk the IR, resolve generators, dedup envelopes, emit the program build
calls in order — is exercised against a **RecordingProgram** fake, so it needs no qick and no
tProc-v2 firmware. Only ``quantize`` (unit conversions) needs a real ``QickConfig``; those tests
are marked ``qick`` and use the committed (v1) soccfg — the conversion math is version-agnostic.
"""

import pytest

from strumento.core.compiler import Compiler, quantize, register_envelope
from strumento.core.pulses import Const, Gaussian, Pulse, Seq
from strumento.core.wiring import Generator, Line, LineRef, ReadoutWiring, Wiring


class RecordingProgram:
    """A fake program that records the build calls the compiler makes (no qick)."""

    def __init__(self):
        self.calls = []

    def declare_gen(self, ch, **kw):        self.calls.append(("declare_gen", ch, kw))
    def add_gauss(self, ch, name, **kw):    self.calls.append(("add_gauss", ch, name, kw))
    def add_pulse(self, ch, name, **kw):    self.calls.append(("add_pulse", ch, name, kw))
    def pulse(self, ch, name, t=0, **kw):   self.calls.append(("pulse", ch, name, t))
    def delay(self, t, **kw):               self.calls.append(("delay", t))
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


# --- quantize needs a real QickConfig (version-agnostic math; v1 soccfg is fine) ---

@pytest.mark.qick
def test_quantize_realizes_to_dac_grid():
    from pathlib import Path
    import json
    from qick import QickConfig

    snap = Path(__file__).parents[2] / "devices" / "schuster_multimode" / "soccfg_snapshot.json"
    soccfg = QickConfig(json.loads(snap.read_text()))
    r = quantize(_pulse(gain=0.1, freq=3562.6105), soccfg, _wiring())
    assert r.freq_mhz_actual == pytest.approx(3562.6105, abs=1e-3)  # grid-snapped, near-exact
    assert r.cycles > 0
    assert 0.0 < r.gain_frac <= 1.0
    assert r.gain_frac == pytest.approx(round(0.1 * 32766) / 32766)
