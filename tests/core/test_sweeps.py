"""Tests for the Sweep axis (plan Task 11, D23)."""

import pytest

from strumento.core.sweeps import Sweep


def test_sweep_fields():
    s = Sweep("amp", 0, 30000, 51)
    assert (s.name, s.start, s.stop, s.points) == ("amp", 0, 30000, 51)


def test_xpts_defaults_to_linspace_over_start_stop():
    s = Sweep("amp", 0, 30000, 51)
    x = s.xpts
    assert len(x) == 51
    assert x[0] == pytest.approx(0) and x[-1] == pytest.approx(30000)


def test_mapped_converts_endpoints_but_keeps_lab_native_xpts():
    # D23: user sweeps lab-native int codes; the factory maps endpoints to a v2 fraction for the
    # compiler, but the experiment axis (xpts) must stay the original lab-native values.
    s = Sweep("amp", 0, 30000, 51)
    frac = s.mapped(lambda code: code / 32766)
    assert frac.start == pytest.approx(0.0)
    assert frac.stop == pytest.approx(30000 / 32766)
    assert frac.points == 51
    assert frac.xpts[0] == pytest.approx(0) and frac.xpts[-1] == pytest.approx(30000)  # lab-native


def test_sweep_is_frozen_hashable():
    assert hash(Sweep("amp", 0, 1, 3)) == hash(Sweep("amp", 0, 1, 3))
    assert Sweep("amp", 0, 1, 3) != Sweep("amp", 0, 2, 3)
