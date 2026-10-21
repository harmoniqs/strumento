"""The per-element gain ceilings have a hard cap, and the pack declares them (#190).

``qubit.max_drive_gain``, ``readout.max_readout_gain`` and ``modes.<name>.max_displace_gain``
replace the pack default limit for their element. Without an upper bound, a calibration could
raise them to full scale and every factory would build pulses above the pack default. So:

* each ceiling has one absolute cap per role in ``safety.py``, above the role's default limit and
  below full scale, and the schema bounds the field to ``0 < limit <= cap``;
* a calibration that sets a ceiling above its cap fails ``Device.from_config``, located at the
  field — through a device YAML and through the library's store write-back alike;
* a ceiling at or below its cap behaves as it always did: the factories use it as the limit;
* the pack declares its ceiling paths (``CALIBRATION_CEILINGS``), derived from a marker on the
  fields themselves, so the MCP session can refuse writes to them without naming any field.

Pure lane: no qick needed.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

import strumento.packs.cqed as cqed
from strumento.core.config import apply_updates
from strumento.core.device import Device
from strumento.core.store import FileConfigStore, record_calibration
from strumento.packs.cqed import safety
from strumento.packs.cqed.calibration import GAIN_CEILING_MARK, Calibration, ceiling_paths
from strumento.packs.cqed.experiments._common import drive_gain_limit
from strumento.packs.cqed.safety import GainOutOfRange
from tests._paths import MULTIMODE_YAML, SHIPPED_DEVICE_YAMLS
from tests.packs.cqed.test_calibration_units import _fields

#: (calibration path, the role's default limit, the role's hard cap)
CEILINGS = {
    "qubit": ("qubit.max_drive_gain", safety.MAX_QUBIT_DRIVE_GAIN, safety.QUBIT_DRIVE_GAIN_CAP),
    "readout": ("readout.max_readout_gain", safety.MAX_READOUT_GAIN, safety.READOUT_GAIN_CAP),
    "mode": (
        "modes.cavity.max_displace_gain",
        safety.MAX_DISPLACEMENT_GAIN,
        safety.DISPLACEMENT_GAIN_CAP,
    ),
}


def _config_with(updates: dict):
    config, _ = apply_updates(Device.load(MULTIMODE_YAML).to_config(), updates)
    return config


# -- the caps -------------------------------------------------------------------------------


@pytest.mark.parametrize("role", list(CEILINGS))
def test_each_cap_sits_above_its_default_and_at_most_full_scale(role):
    _, default, cap = CEILINGS[role]
    assert default < cap <= 1.0


def test_the_default_limits_are_unchanged():
    """Out of scope for #190: the caps are new, the defaults stay as they were."""
    assert (safety.MAX_QUBIT_DRIVE_GAIN, safety.MAX_READOUT_GAIN, safety.MAX_DISPLACEMENT_GAIN) == (
        0.5, 0.6, 0.4,
    )


def test_the_safety_docstring_describes_the_rule_it_enforces():
    doc = safety.__doc__ or ""
    assert "unsafe=True" not in doc and "unsafe=" not in doc
    assert "cap" in doc


# -- the schema bound ------------------------------------------------------------------------


@pytest.mark.parametrize("role", list(CEILINGS))
def test_a_ceiling_above_its_cap_fails_to_compose_naming_the_field(role):
    path, _, cap = CEILINGS[role]
    with pytest.raises(ValidationError) as err:
        Device.from_config(_config_with({path: cap + 1e-6}))
    locs = {".".join(str(p) for p in e["loc"]) for e in err.value.errors()}
    assert path in locs, f"the error is not located at {path}: {sorted(locs)}"


@pytest.mark.parametrize("value", [0.0, -0.1])
@pytest.mark.parametrize("role", list(CEILINGS))
def test_a_ceiling_must_be_positive(role, value):
    path, _, _ = CEILINGS[role]
    with pytest.raises(ValidationError) as err:
        Device.from_config(_config_with({path: value}))
    locs = {".".join(str(p) for p in e["loc"]) for e in err.value.errors()}
    assert path in locs


@pytest.mark.parametrize("role", list(CEILINGS))
def test_a_ceiling_at_its_cap_or_below_loads_and_is_the_value_set(role):
    path, default, cap = CEILINGS[role]
    for value in (cap, default, 0.1):
        device = Device.from_config(_config_with({path: value}))
        node = device.calib.model_dump()
        for key in path.split("."):
            node = node[key]
        assert node == value


def test_a_qubit_ceiling_at_the_cap_is_the_limit_the_factories_use():
    """At the cap the per-element limit replaces the default exactly as before #190."""
    cap = safety.QUBIT_DRIVE_GAIN_CAP
    device = Device.from_config(_config_with({"qubit.max_drive_gain": cap}))
    assert drive_gain_limit(device) == cap
    maxv = device.calib.qubit.pulses["pi_ge"].gain / device.qubit.ge_pi().gain
    # a pi gain above the default limit but under the cap now builds, one above the cap does not
    between = (safety.MAX_QUBIT_DRIVE_GAIN + cap) / 2
    raised = Device.from_config(_config_with({
        "qubit.max_drive_gain": cap, "qubit.pulses.pi_ge.gain": between * maxv,
    }))
    assert raised.qubit.ge_pi().gain == pytest.approx(between)
    over = Device.from_config(_config_with({
        "qubit.max_drive_gain": cap, "qubit.pulses.pi_ge.gain": (cap + 0.01) * maxv,
    }))
    with pytest.raises(GainOutOfRange):
        over.qubit.ge_pi()


def test_an_unset_ceiling_still_means_the_pack_default():
    device = Device.load(MULTIMODE_YAML)
    assert device.calib.qubit.max_drive_gain is None
    assert drive_gain_limit(device) == safety.MAX_QUBIT_DRIVE_GAIN


@pytest.mark.parametrize("yaml_path", SHIPPED_DEVICE_YAMLS, ids=lambda p: p.parent.name)
def test_no_shipped_device_sets_a_ceiling_above_its_cap(yaml_path):
    Device.load(yaml_path)  # a ceiling above its cap would fail to compose here


# -- the library write-back path (#191 validates before the ref moves) -----------------------


def _store_state(root) -> tuple:
    log = root / "log" / "main.jsonl"
    return (
        (root / "refs" / "main").read_text(),
        sorted(p.name for p in (root / "objects").iterdir()),
        log.read_bytes() if log.exists() else b"",
    )


@pytest.fixture
def store(tmp_path):
    root = tmp_path / "store"
    s = FileConfigStore(root)
    base = Device.load(MULTIMODE_YAML).to_config()
    s.set_ref("main", s.put(base))
    return root, s, base


@pytest.mark.parametrize("role", list(CEILINGS))
def test_an_over_cap_ceiling_written_through_the_library_is_refused_before_the_ref_moves(
    store, role
):
    root, s, base = store
    path, _, cap = CEILINGS[role]
    state = _store_state(root)
    with pytest.raises(ValidationError, match=path.rsplit(".", 1)[1]):
        record_calibration(s, "main", base, {path: cap + 0.05}, source="manual")
    assert _store_state(root) == state


@pytest.mark.parametrize("role", list(CEILINGS))
def test_a_human_can_still_set_a_ceiling_up_to_its_cap_through_the_library(store, role):
    _, s, base = store
    path, _, cap = CEILINGS[role]
    cid, config = record_calibration(s, "main", base, {path: cap}, source="manual")
    assert s.resolve_ref("main") == cid
    Device.from_config(config)


# -- the pack declares its ceilings ----------------------------------------------------------


def test_the_pack_declares_its_ceiling_paths():
    assert set(cqed.CALIBRATION_CEILINGS) == {
        "qubit.max_drive_gain", "readout.max_readout_gain", "modes.*.max_displace_gain",
    }
    assert tuple(cqed.CALIBRATION_CEILINGS) == ceiling_paths(Calibration)


def test_every_gain_limit_field_is_marked_as_a_ceiling():
    """A new ``max_*_gain`` field is a ceiling: it must carry the mark (and so be declared)."""
    for model, name in _fields():
        if name.startswith("max_") and name.endswith("_gain"):
            extra = model.model_fields[name].json_schema_extra or {}
            assert extra.get(GAIN_CEILING_MARK) is True, (
                f"{model.__name__}.{name} is a gain limit but is not marked as a ceiling"
            )


# -- a raised ceiling reaches every experiment, and the refusal says how to raise it ---------


def _raised_device(updates: dict):
    return Device.from_config(_config_with(updates))


@pytest.mark.parametrize(
    ("module", "cls"),
    [
        ("qubit_spectroscopy", "QubitSpectroscopy"),
        ("ef_spectroscopy", "EFSpectroscopy"),
        ("number_splitting", "NumberSplitting"),
        ("tls_scan", "TlsScan"),
    ],
)
def test_a_raised_drive_ceiling_reaches_the_qubit_probe_experiments(module, cls):
    """A probe tone on the qubit drive faces the qubit's calibrated limit, not the pack default."""
    import importlib

    exp_cls = getattr(importlib.import_module(f"strumento.packs.cqed.experiments.{module}"), cls)
    with pytest.raises(GainOutOfRange):
        exp_cls(Device.load(MULTIMODE_YAML), probe_gain=0.55).sequence()
    exp_cls(_raised_device({"qubit.max_drive_gain": 0.6}), probe_gain=0.55).sequence()  # no error


def test_a_raised_drive_ceiling_reaches_the_gain_ladder():
    from strumento.packs.cqed.experiments.gain_ladder_spectroscopy import GainLadderSpectroscopy

    gains = [int(0.55 * 32766)]
    with pytest.raises(GainOutOfRange):
        GainLadderSpectroscopy(Device.load(MULTIMODE_YAML), gains=gains).sequence()
    raised = _raised_device({"qubit.max_drive_gain": 0.6})
    GainLadderSpectroscopy(raised, gains=gains).sequence()  # no error


def test_a_raised_displacement_ceiling_reaches_cavity_spectroscopy():
    from strumento.packs.cqed.experiments.cavity_spectroscopy import CavitySpectroscopy

    with pytest.raises(GainOutOfRange):
        CavitySpectroscopy(Device.load(MULTIMODE_YAML), mode_name="cavity", probe_gain=0.45).sequence()
    raised = _raised_device({"modes.cavity.max_displace_gain": 0.5})
    CavitySpectroscopy(raised, mode_name="cavity", probe_gain=0.45).sequence()  # no error


@pytest.mark.parametrize(
    ("build", "field", "cap"),
    [
        (lambda d: d.qubit.ge_pi(gain=0.6), "qubit.max_drive_gain", safety.QUBIT_DRIVE_GAIN_CAP),
        (lambda d: d.readout.probe(gain=0.7), "readout.max_readout_gain", safety.READOUT_GAIN_CAP),
        (
            lambda d: d.cavity.displace(gain=0.5),
            "modes.cavity.max_displace_gain",
            safety.DISPLACEMENT_GAIN_CAP,
        ),
    ],
)
def test_the_refusal_names_the_ceiling_to_raise_and_its_cap(build, field, cap):
    with pytest.raises(GainOutOfRange) as err:
        build(Device.load(MULTIMODE_YAML))
    assert field in str(err.value) and f"up to {cap}" in str(err.value)


def test_a_sweep_refusal_names_the_ceiling_to_raise():
    from strumento.core.sweeps import Sweep

    with pytest.raises(GainOutOfRange) as err:
        Device.load(MULTIMODE_YAML).qubit.ge_pi(gain=Sweep(start=0, stop=20000, on="amp"))
    assert "qubit.max_drive_gain" in str(err.value)


@pytest.mark.parametrize(
    "build",
    [lambda d: d.coupler.swap("cavity-S1", gain=0.6), lambda d: d.sideband.f0g1(gain=0.6)],
)
def test_a_role_without_a_ceiling_says_so(build):
    with pytest.raises(GainOutOfRange) as err:
        build(Device.load(MULTIMODE_YAML))
    assert "no per-element override" in str(err.value)
