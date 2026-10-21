"""Tests for ReadoutOps' dispersive_measure() helper (issue #99, part of #96).

``dispersive_measure()`` bundles ``Seq().play(self.probe(...)).measure(...)`` behind one call,
reusing ``probe()``'s own defaulting logic rather than duplicating it (issue #99's Key
Decision) -- these tests assert the compiled op sequence matches the equivalent hand-written
``.play(probe(...)).measure(...)`` call, for both defaulted and overridden arguments, plus
composability with ``Seq.__add__``.
"""

from __future__ import annotations

from strumento.core.pulses import Seq
from strumento.core.wiring import LineRef


def test_dispersive_measure_matches_hand_composed_play_probe_measure_with_defaults(dev):
    # AC1 (exists), AC2 (defaults match probe()'s own) and AC3 (same compiled ops as the
    # manual .play(probe(...)).measure() call) all fall out of one structural comparison.
    got = dev.readout.dispersive_measure()
    expected = Seq().play(dev.readout.probe(start_us=0.0)).measure()
    assert got.ops == expected.ops


def test_dispersive_measure_matches_hand_composed_with_overridden_arguments(dev):
    # Every argument dispersive_measure() takes must reach the same place a hand-written
    # .play(probe(...)).measure(kind=..., **params) call would put it -- this is what proves
    # pass-through rather than a second, possibly-diverging copy of probe()'s own defaulting.
    line = LineRef("readout", "probe")
    got = dev.readout.dispersive_measure(
        freq=7010.0, gain=0.4, length=3.0, line=line, kind="dispersive", shots=1000,
    )
    expected = Seq().play(
        dev.readout.probe(freq=7010.0, gain=0.4, length=3.0, line=line, start_us=0.0)
    ).measure(kind="dispersive", shots=1000)
    assert got.ops == expected.ops


def test_dispersive_measure_returns_a_seq_composable_with_add(dev):
    # AC4: existing_seq + dev.readout.dispersive_measure() must compose via Seq.__add__
    # without mutating either operand (Seq's own __add__ contract).
    existing = Seq().play(dev.qubit.ge_pi())
    existing_ops_before = list(existing.ops)
    expected_tail = Seq().play(dev.readout.probe(start_us=0.0)).measure().ops

    combined = existing + dev.readout.dispersive_measure()

    assert isinstance(combined, Seq)
    assert combined.ops == existing_ops_before + expected_tail
    assert existing.ops == existing_ops_before
