"""The three meanings of ``gain=`` in the cqed ops factories, checked against the code (#185).

Every public method of a cqed ops class (``TransmonOps``, ``ReadoutOps``, ``ModeOps``,
``CouplerOps``, ``SidebandOps``, and any ops class added to ``strumento.packs.cqed.library``
later) that takes a ``gain`` parameter is found by introspection, then:

* built on ``multimode_demo`` with ``gain=None``, a number and a ``Sweep``, and the resulting
  pulse's ``gain`` is compared with what the "Gain units" section of the
  :mod:`strumento.packs.cqed.library` docstring says: the calibrated DAC integer code divided by
  ``maxv``; the number as is; the sweep's integer-code endpoints each divided by ``maxv``;
* checked to refuse a gain ``Sweep`` whose endpoint looks like a fraction (``0 < |v| < 1``) and
  a number above its role's limit;
* checked to point to that section from its docstring's ``gain`` entry.

A new factory with a ``gain`` parameter fails :func:`test_every_gain_factory_has_a_case` until
it gets a row in ``CASES``, so it can't skip either check. Pure lane: no qick needed.
"""

from __future__ import annotations

import importlib
import inspect
import pkgutil
import re

import pytest

import strumento.packs.cqed.library as cqed_library
from strumento import Device, Sweep
from strumento.core.pulses import Pulse, Seq
from strumento.packs.cqed.calibration import gain_int_to_frac
from strumento.packs.cqed.library._gain import maxv_for
from strumento.packs.cqed.safety import GainOutOfRange

POINTER = re.compile(r"\bgain\b.*\bsee\b.*Gain units.*:mod:`strumento\.packs\.cqed\.library`")


def _gain_factories() -> dict[str, object]:
    """``{"Class.method": function}`` for every public ops method with a ``gain`` parameter."""
    found: dict[str, object] = {}
    for info in pkgutil.iter_modules(cqed_library.__path__):
        module = importlib.import_module(f"{cqed_library.__name__}.{info.name}")
        for cls_name, cls in inspect.getmembers(module, inspect.isclass):
            if cls.__module__ != module.__name__ or not cls_name.endswith("Ops"):
                continue
            for name, fn in inspect.getmembers(cls, inspect.isfunction):
                if not name.startswith("_") and "gain" in inspect.signature(fn).parameters:
                    found[f"{cls_name}.{name}"] = fn
    return found


FACTORIES = _gain_factories()

#: ``"Class.method"`` -> (device attribute, positional args, calibrated code on the device).
CASES = {
    "TransmonOps.ge_pi": ("qubit", (), lambda d: d.calib.qubit.pulses["pi_ge"].gain),
    "TransmonOps.ge_hpi": ("qubit", (), lambda d: d.calib.qubit.pulses["hpi_ge"].gain),
    "TransmonOps.ef_pi": ("qubit", (), lambda d: d.calib.qubit.pulses["pi_ef"].gain),
    "TransmonOps.ef_hpi": ("qubit", (), lambda d: d.calib.qubit.pulses["hpi_ef"].gain),
    "ReadoutOps.probe": ("readout", (), lambda d: d.calib.readout.gain),
    "ReadoutOps.dispersive_measure": ("readout", (), lambda d: d.calib.readout.gain),
    "ModeOps.displace": ("cavity", (), lambda d: d.calib.modes["cavity"].displace.gain),
    "CouplerOps.swap": ("coupler", ("cavity-S1",), lambda d: d.calib.swaps["cavity-S1"].gain),
    "SidebandOps.f0g1": ("sideband", (), lambda d: d.calib.multiphoton["f0-g1"].pi[0].gain),
    "SidebandOps.multiphoton": (
        "sideband", ("f0-g1",), lambda d: d.calib.multiphoton["f0-g1"].pi[0].gain,
    ),
}

#: Integer-code sweep endpoints, inside every role's limit on the demo (3000 / 32766 ≈ 0.09).
SWEEP_START, SWEEP_STOP = 1000, 3000
NUMBER = 0.25


def _build(dev: Device, key: str, **kw) -> Pulse:
    """Call the factory and return the pulse that carries the gain (a ``Seq`` holds one)."""
    attr, args, _ = CASES[key]
    out = getattr(getattr(dev, attr), key.split(".")[1])(*args, **kw)
    if isinstance(out, Seq):
        pulses = [op for op in out.ops if isinstance(op, Pulse)]
        assert len(pulses) == 1, f"{key} returned a Seq with {len(pulses)} pulses"
        out = pulses[0]
    assert isinstance(out, Pulse)
    return out


def _maxv(dev: Device, pulse: Pulse) -> float:
    freq = pulse.freq_mhz.start if isinstance(pulse.freq_mhz, Sweep) else pulse.freq_mhz
    return maxv_for(dev, pulse.line, freq)


def test_every_gain_factory_has_a_case():
    assert set(FACTORIES) == set(CASES), (
        "a cqed ops factory takes gain= but has no row in CASES (or a row names a factory that "
        f"no longer exists): missing {sorted(set(FACTORIES) - set(CASES))}, "
        f"stale {sorted(set(CASES) - set(FACTORIES))}"
    )


@pytest.mark.parametrize("key", sorted(CASES))
def test_gain_none_is_the_calibrated_code_as_a_fraction(multimode_dev, key):
    pulse = _build(multimode_dev, key)
    code = CASES[key][2](multimode_dev)
    assert code >= 1, "the calibrated gain is a DAC integer code, not a fraction"
    assert pulse.gain == gain_int_to_frac(code, _maxv(multimode_dev, pulse))


@pytest.mark.parametrize("key", sorted(CASES))
def test_gain_number_is_a_fraction_used_as_is(multimode_dev, key):
    assert _build(multimode_dev, key, gain=NUMBER).gain == NUMBER


@pytest.mark.parametrize("key", sorted(CASES))
def test_gain_sweep_is_integer_codes_converted_per_endpoint(multimode_dev, key):
    pulse = _build(multimode_dev, key, gain=Sweep(start=SWEEP_START, stop=SWEEP_STOP, on="amp"))
    maxv = _maxv(multimode_dev, pulse)
    assert isinstance(pulse.gain, Sweep)
    assert (pulse.gain.start, pulse.gain.stop) == (SWEEP_START / maxv, SWEEP_STOP / maxv)
    assert pulse.gain.on == "amp"


@pytest.mark.parametrize("key", sorted(CASES))
def test_gain_sweep_with_a_fraction_endpoint_is_refused(multimode_dev, key):
    with pytest.raises(GainOutOfRange, match="looks like a v2 fraction"):
        _build(multimode_dev, key, gain=Sweep(start=0, stop=0.3, on="amp"))


@pytest.mark.parametrize("key", sorted(CASES))
def test_gain_number_above_the_limit_is_refused(multimode_dev, key):
    # An integer code passed as a number is read as a fraction, so it fails the limit.
    code = CASES[key][2](multimode_dev)
    with pytest.raises(GainOutOfRange, match="exceeds safe limit"):
        _build(multimode_dev, key, gain=code)


def test_the_library_docstring_states_the_three_meanings():
    doc = inspect.getdoc(cqed_library) or ""
    assert "Gain units" in doc
    section = doc[doc.index("Gain units"):]
    for phrase in ("None", "DAC integer code", "gain_int_to_frac", "fraction of full scale",
                   "as is", "Sweep", "each endpoint"):
        assert phrase in section, f"the 'Gain units' section does not mention {phrase!r}"


def test_the_gain_units_docs_say_an_override_faces_the_limit():
    """An override's own gain is checked like every other pulse (#187): no doc calls it a
    variation that skips the limit."""
    from pathlib import Path

    doc = inspect.getdoc(cqed_library) or ""
    section = " ".join(doc[doc.index("Gain units"):].split())
    assert "not re-checked" not in section
    assert "faces the drive limit like every other pulse" in section
    assert "One variation" in section and "Two variations" not in section
    docs = Path(__file__).resolve().parents[3] / "docs"
    for name in ("getting-started.md", "cqed-cookbook.md"):
        text = " ".join((docs / name).read_text().split())
        assert "two variations" not in text, f"docs/{name} still lists the override as a variation"
        assert "returns the override's own gain for `gain=None`, and" not in text


@pytest.mark.parametrize("key", sorted(FACTORIES))
def test_every_gain_factory_points_to_the_gain_units_section(key):
    doc = inspect.getdoc(FACTORIES[key]) or ""
    assert any(POINTER.search(line) for line in doc.splitlines()), (
        f"{key}'s docstring needs a gain entry pointing to the 'Gain units' section, e.g. "
        "'gain : see \"Gain units\" in :mod:`strumento.packs.cqed.library`.'"
    )
