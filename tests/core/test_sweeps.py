"""Tests for the Sweep axis (gains as v2 fractions, D23) and declared Loop axes (D36)."""

import inspect
import warnings

import pytest

from strumento.core.sweeps import Loop, Sweep

# The legacy per-name form (D36, deprecated) still works: these cases pin its fields, display
# axis and mapping, so they construct it on purpose and expect its deprecation warning.
LEGACY = "legacy per-name Sweep"


def test_sweep_fields():
    with pytest.warns(DeprecationWarning, match=LEGACY):
        s = Sweep("amp", 0, 30000, 51)
    assert (s.name, s.start, s.stop, s.points) == ("amp", 0, 30000, 51)


def test_xpts_defaults_to_linspace_over_start_stop():
    with pytest.warns(DeprecationWarning, match=LEGACY):
        s = Sweep("amp", 0, 30000, 51)
    x = s.xpts
    assert len(x) == 51
    assert x[0] == pytest.approx(0) and x[-1] == pytest.approx(30000)


def test_mapped_converts_endpoints_but_keeps_lab_native_xpts():
    # D23: user sweeps lab-native int codes; the factory maps endpoints to a v2 fraction for the
    # compiler, but the experiment axis (xpts) must stay the original lab-native values.
    with pytest.warns(DeprecationWarning, match=LEGACY):
        s = Sweep("amp", 0, 30000, 51)
    frac = s.mapped(lambda code: code / 32766)
    assert frac.start == pytest.approx(0.0)
    assert frac.stop == pytest.approx(30000 / 32766)
    assert frac.points == 51
    assert frac.xpts[0] == pytest.approx(0) and frac.xpts[-1] == pytest.approx(30000)  # lab-native


def test_sweep_is_frozen_hashable():
    with pytest.warns(DeprecationWarning, match=LEGACY):
        assert hash(Sweep("amp", 0, 1, 3)) == hash(Sweep("amp", 0, 1, 3))
        assert Sweep("amp", 0, 1, 3) != Sweep("amp", 0, 2, 3)


# --- declared loop axes (D36) -------------------------------------------------


def test_loop_axis_fields():
    ax = Loop("ax", 51)
    assert (ax.name, ax.points) == ("ax", 51)


def test_loop_axis_rejects_bad_points():
    with pytest.raises(ValueError, match="point count"):
        Loop("ax", 0)
    with pytest.raises(ValueError, match="point count"):
        Loop("ax", 2.5)


def test_bound_sweep_carries_range_and_axis_only():
    s = Sweep(start=0.0, stop=0.9, on="ax")
    assert (s.start, s.stop, s.on) == (0.0, 0.9, "ax")
    assert s.points is None and s.name is None


def test_bound_sweep_rejects_points():
    with pytest.raises(ValueError, match="point count"):
        Sweep(start=0, stop=1, on="ax", points=3)


def test_unbound_sweep_requires_points_and_name():
    with pytest.raises(ValueError, match="point count"):
        Sweep("amp", 0, 1)  # legacy form without points
    with pytest.raises(ValueError, match="name"):
        Sweep(start=0, stop=1, points=3)  # legacy form without a name


def test_bound_sweep_xpts_raises_without_a_loop():
    s = Sweep(start=0.0, stop=0.9, on="ax")
    with pytest.raises(ValueError, match="Loop"):
        _ = s.xpts


def test_mapped_preserves_axis_binding_and_inverse():
    s = Sweep(start=0, stop=30000, on="ax")
    m = s.mapped(lambda code: code / 32766, inverse=lambda frac: frac * 32766)
    assert m.on == "ax"
    assert m.start == pytest.approx(0.0) and m.stop == pytest.approx(30000 / 32766)
    assert m.points is None
    assert m._lab_inverse is not None and m._lab_inverse(0.5) == pytest.approx(0.5 * 32766)


def test_axis_values_linspace():
    s = Sweep(start=100.0, stop=200.0, on="ax")
    x = s.axis_values(5)
    assert len(x) == 5 and x[0] == pytest.approx(100) and x[-1] == pytest.approx(200)


# --- the legacy form's deprecation (D36, #162) ----------------------------------


def _deprecations(record):
    return [w for w in record if issubclass(w.category, DeprecationWarning)]


def test_legacy_sweep_warns_once_at_the_line_that_wrote_it():
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        line = inspect.currentframe().f_lineno + 1
        Sweep("amp", 0, 30000, 51)
    [w] = _deprecations(record)
    assert "deprecated (D36)" in str(w.message)
    assert "Loop('amp', 51)" in str(w.message) and "on='amp'" in str(w.message)
    assert w.filename == __file__
    assert w.lineno == line


def test_bound_sweep_does_not_warn():
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        Sweep(start=0, stop=30000, on="amp")
    assert not _deprecations(record)


def test_mapping_a_legacy_sweep_does_not_warn_again():
    # mapped() builds a new Sweep inside strumento; the user's own line already warned
    with pytest.warns(DeprecationWarning, match=LEGACY):
        s = Sweep("amp", 0, 30000, 51)
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        m = s.mapped(lambda code: code / 32766).mapped(lambda frac: frac * 2)
    assert not _deprecations(record)
    assert m.points == 51 and m.name == "amp"
