"""The cqed calibration schema refuses values no device can have (#191).

A calibration value reaches a device two ways: a device YAML (``Device.load``) and a store
write-back (``record_calibration``). Both compose through the same pydantic model, so the
constraints live on the fields themselves:

* no NaN or infinite number anywhere in the calibration;
* every time (a µs field) is ``>= 0``;
* every gain stored as a DAC integer code is ``>= 0``.

The gain *ceilings* (``max_*_gain``, a fraction of full scale) are bounded by their hard caps
(#190, ``test_gain_ceilings.py``); ``trig_offset``'s upper bound is in ``test_trig_offset_bounds.py``
(#195).
Pure lane: no qick needed.
"""

from __future__ import annotations

import math
import re

import pytest
from pydantic import ValidationError

from strumento.core.device import Device
from strumento.packs.cqed.calibration import Calibration
from tests._paths import MULTIMODE_YAML, SHIPPED_DEVICE_YAMLS
from tests.packs.cqed.test_calibration_units import _fields, _models_in

# -- the YAML door ---------------------------------------------------------------------------

#: (yaml line in devices/multimode_demo/device.yaml, its replacement, the field the error names)
BAD_YAML_VALUES = {
    "nan-time": ("relax_delay: 500", "relax_delay: .nan", "readout.relax_delay"),
    "inf-time": ("trig_offset: 0.5", "trig_offset: .inf", "readout.trig_offset"),
    "inf-frequency": ("f_ge: 4000.0", "f_ge: -.inf", "qubit.f_ge"),
    "negative-time": ("T1: 100.0", "T1: -5.0", "qubit.T1"),
}


def _write_variant(tmp_path, old: str, new: str):
    text = MULTIMODE_YAML.read_text(encoding="utf-8")
    assert text.count(old) == 1, f"{old!r} must appear exactly once in {MULTIMODE_YAML.name}"
    text = text.replace(old, new)
    snapshot = (MULTIMODE_YAML.parent / "../_fixtures/soccfg_v2_cqed.json").resolve()
    text = re.sub(r"^soccfg_snapshot: .*$", f"soccfg_snapshot: {snapshot}", text, flags=re.M)
    path = tmp_path / "device.yaml"
    path.write_text(text, encoding="utf-8")
    return path


@pytest.mark.parametrize("case", list(BAD_YAML_VALUES.values()), ids=list(BAD_YAML_VALUES))
def test_a_device_yaml_with_a_bad_value_fails_to_load_naming_the_field(tmp_path, case):
    old, new, field = case
    with pytest.raises(ValidationError) as err:
        Device.load(_write_variant(tmp_path, old, new))
    locs = {".".join(str(p) for p in e["loc"]) for e in err.value.errors()}
    assert field in locs, f"the error is not located at {field}: {sorted(locs)}"


def test_the_unmodified_variant_still_loads(tmp_path):
    """The YAML rewrite itself is sound: with no value changed the device composes."""
    Device.load(_write_variant(tmp_path, "relax_delay: 500", "relax_delay: 500"))


@pytest.mark.parametrize("yaml_path", SHIPPED_DEVICE_YAMLS, ids=lambda p: p.parent.name)
def test_every_shipped_device_still_loads(yaml_path):
    Device.load(yaml_path)


# -- the schema contract ---------------------------------------------------------------------


def _is_float_leaf(model, name) -> bool:
    return not _models_in(model.model_fields[name].annotation)


def _unit(model, name) -> str:
    return (model.model_fields[name].description or "").split(" — ", 1)[0]


def _schema_props(model, name) -> list[dict]:
    """The JSON Schema branches a field's value can take (``anyOf`` flattened; arrays unwrapped)."""
    prop = model.model_json_schema()["properties"][name]
    branches = prop.get("anyOf", [prop])
    out = []
    for b in branches:
        if b.get("type") == "array":
            out.extend(b.get("prefixItems", []) or [b.get("items", {})])
        else:
            out.append(b)
    return [b for b in out if b.get("type") in ("number", "integer")]


TIME_FIELDS = [f for f in _fields() if _is_float_leaf(*f) and _unit(*f).startswith("µs")]
GAIN_CODE_FIELDS = [
    f for f in _fields() if _is_float_leaf(*f) and _unit(*f) == "DAC integer code"
]


def test_the_constrained_field_sets_are_what_the_schema_declares():
    """Non-vacuity: the enumeration finds the time and gain-code fields this issue names."""
    names = {f"{m.__name__}.{n}" for m, n in TIME_FIELDS}
    assert {
        "ReadoutCal.relax_delay", "ReadoutCal.trig_offset", "ReadoutCal.readout_length",
        "QubitCal.T1", "PulseCal.sigma", "ModeCal.revival_time", "SwapCal.length",
    } <= names
    assert {f"{m.__name__}.{n}" for m, n in GAIN_CODE_FIELDS} == {
        "PulseCal.gain", "ReadoutCal.gain", "SidebandPulseCal.gain", "SwapCal.gain",
    }


@pytest.mark.parametrize(
    "field", TIME_FIELDS + GAIN_CODE_FIELDS, ids=lambda f: f"{f[0].__name__}.{f[1]}"
)
def test_every_time_and_gain_code_field_publishes_minimum_zero(field):
    model, name = field
    branches = _schema_props(model, name)
    assert branches, f"{model.__name__}.{name} has no numeric schema branch"
    for b in branches:
        assert b.get("minimum") == 0, f"{model.__name__}.{name}: schema branch {b} lacks minimum 0"


def _calib(**readout):
    base = {"frequency": 750.0, "gain": 800, "readout_length": 1.5,
            "trig_offset": 0.45, "relax_delay": 250}
    return {"qubit": {"f_ge": 4000.0}, "readout": {**base, **readout}}


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf], ids=repr)
@pytest.mark.parametrize("path", [
    ("readout", "frequency"), ("readout", "phase"), ("readout", "threshold"),
    ("readout", "g_center_iq"), ("qubit", "drag_alpha"), ("couplings", "qubit-cavity"),
])
def test_no_float_in_the_calibration_takes_a_non_finite_value(path, value):
    """Not only the bounded fields: every float, including tuples and dict values, is finite."""
    d = _calib()
    section, key = path
    d.setdefault(section, {})[key] = (value, 0.0) if key == "g_center_iq" else value
    with pytest.raises(ValidationError, match="finite"):
        Calibration(**d)


@pytest.mark.parametrize("value", [-1e-9, -5.0])
def test_a_negative_time_is_refused(value):
    with pytest.raises(ValidationError, match="greater than or equal to 0"):
        Calibration(**_calib(trig_offset=value))


def test_zero_is_a_valid_time_and_gain():
    c = Calibration(**_calib(trig_offset=0.0, gain=0))
    assert c.readout.trig_offset == 0.0 and c.readout.gain == 0
