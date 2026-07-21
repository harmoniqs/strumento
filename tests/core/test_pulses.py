"""Tests for the pulse IR + fluent builder (spec §6.1, plan Task 2, D22).

Envelopes are a **typed union** of frozen dataclasses (``Gaussian``/``FlatTop``/``Const``/``Arb``),
each carrying its own params — no flat ``Shape`` enum, no ``None``-field conventions. Frozen ⇒
hashable ⇒ the compiler can dedup on ``(generator, envelope)`` for free (spec R6).
"""

from strumento.core.pulses import (
    Pulse, Delay, Segment, Raw, Measure, Seq, Gaussian, FlatTop, Const, Arb,
)
from strumento.core.wiring import LineRef


def _p():
    return Pulse(
        line=LineRef("qubit", "drive"), freq_mhz=3562.6, gain=0.10,
        envelope=Gaussian(sigma_us=0.035),
    )


def test_builder_order():
    s = Seq().play(_p()).wait(0.5).measure()
    kinds = [type(op).__name__ for op in s.ops]
    assert kinds == ["Pulse", "Delay", "Measure"]


def test_play_parallel_makes_segment():
    s = Seq().play_parallel([_p(), _p()])
    assert isinstance(s.ops[0], Segment) and len(s.ops[0].pulses) == 2


def test_envelope_is_frozen_hashable():
    assert hash(Gaussian(sigma_us=0.035)) == hash(Gaussian(sigma_us=0.035))   # dedup key (R6)
    assert Gaussian(sigma_us=0.035) != FlatTop(ramp_sigma_us=0.01, length_us=1.0)


# --- extra coverage beyond the plan's three ---

def test_play_flattens_a_seq():
    inner = Seq().play(_p()).wait(0.2)
    s = Seq().play(_p()).play(inner).measure()
    kinds = [type(op).__name__ for op in s.ops]
    assert kinds == ["Pulse", "Pulse", "Delay", "Measure"]


def test_concatenation_with_add():
    a = Seq().play(_p())
    b = Seq().wait(0.5).measure()
    s = a + b
    kinds = [type(op).__name__ for op in s.ops]
    assert kinds == ["Pulse", "Delay", "Measure"]
    # __add__ must not mutate the operands
    assert len(a.ops) == 1 and len(b.ops) == 2


def test_raw_appends_raw():
    s = Seq().raw(lambda prog: None)
    assert isinstance(s.ops[0], Raw)


def test_all_envelopes_hashable_including_arb():
    # Arb carries samples but stays hashable (coerced to tuples) so it works as a dedup key too.
    for env in (Gaussian(0.035), FlatTop(0.01, 1.0), Const(0.5), Arb([0.0, 1.0, 0.0])):
        assert hash(env) == hash(env)
    assert Arb([0.0, 1.0]) == Arb([0.0, 1.0])
    assert Arb([0.0, 1.0]) != Arb([0.0, 2.0])


def test_pulse_defaults_and_fields():
    p = _p()
    assert p.phase_deg == 0.0 and p.start_us is None and p.label is None
    p2 = Pulse(line=LineRef("qubit", "drive"), freq_mhz=100.0, gain=0.5,
               envelope=Const(0.1), phase_deg=90.0, start_us=0.2, label="x90")
    assert (p2.phase_deg, p2.start_us, p2.label) == (90.0, 0.2, "x90")


def test_measure_carries_kwargs():
    s = Seq().measure(kind="dispersive", shots=1000)
    m = s.ops[0]
    assert isinstance(m, Measure)
    assert m.kind == "dispersive" and m.params["shots"] == 1000
