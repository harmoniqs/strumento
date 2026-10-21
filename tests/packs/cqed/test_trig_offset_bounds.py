"""``trig_offset`` is bounded to ``0 <= trig_offset <= 10`` µs (#195).

``trig_offset`` has been in µs since #98. Two fields feed the readout trigger — the calibration
field ``calib.readout.trig_offset`` (cqed ``ReadoutCal``) and the wiring fallback
``wiring.readout.trig_offset`` (core ``ReadoutWiring``) — and both carry the same bound. A large
value is almost always a clock-cycle count from before #98 (``400`` cycles used to be ~0.9 µs;
it now means 400 µs, and on a board every readout would be noise without an error), so the
refusal names the unit and says so.

Every door is covered: a device YAML, a model built in code, a store write-back and the
time-of-flight write-back, which goes through that same store validation. Pure lane: no qick.
"""

from __future__ import annotations

import re

import pytest
from pydantic import ValidationError

from strumento.core.device import Device
from strumento.core.fitters import FitResult
from strumento.core.store import FileConfigStore, record_calibration
from strumento.core.wiring import ReadoutWiring
from strumento.packs.cqed.calibration import Calibration, ReadoutCal
from strumento.packs.cqed.experiments.time_of_flight import TimeOfFlight
from tests._paths import MULTIMODE_YAML

OUT_OF_RANGE = [400, 400.0, -5, -5.0, 10.000001]
IN_RANGE = [0, 0.0, 0.5, 10, 10.0]


def _locs(err: ValidationError) -> set[str]:
    return {".".join(str(p) for p in e["loc"]) for e in err.errors()}


def _messages(err: ValidationError, loc: str) -> list[str]:
    return [e["msg"] for e in err.errors() if ".".join(str(p) for p in e["loc"]) == loc]


def _calib(**readout):
    base = {"frequency": 750.0, "gain": 800, "readout_length": 1.5,
            "trig_offset": 0.45, "relax_delay": 250}
    return {"qubit": {"f_ge": 4000.0}, "readout": {**base, **readout}}


# -- the calibration field -----------------------------------------------------------------


@pytest.mark.parametrize("value", OUT_OF_RANGE, ids=repr)
def test_the_calibration_field_refuses_an_out_of_range_value_naming_the_unit(value):
    with pytest.raises(ValidationError) as err:
        Calibration(**_calib(trig_offset=value))
    assert "readout.trig_offset" in _locs(err.value)
    msgs = _messages(err.value, "readout.trig_offset")
    assert msgs and all("µs" in m for m in msgs), msgs


@pytest.mark.parametrize("value", IN_RANGE, ids=repr)
def test_the_calibration_field_accepts_an_in_range_value(value):
    assert Calibration(**_calib(trig_offset=value)).readout.trig_offset == value


def test_a_cycle_count_looking_value_says_so():
    with pytest.raises(ValidationError) as err:
        Calibration(**_calib(trig_offset=400))
    (msg,) = _messages(err.value, "readout.trig_offset")
    assert "10 µs" in msg and "clock-cycle" in msg and "#98" in msg, msg


def test_a_negative_value_keeps_the_ge_zero_wording():
    """#191's ``>= 0`` refusal keeps its wording; the unit is added to it."""
    with pytest.raises(ValidationError) as err:
        Calibration(**_calib(trig_offset=-5))
    (msg,) = _messages(err.value, "readout.trig_offset")
    assert "greater than or equal to 0 µs" in msg, msg


def test_the_calibration_schema_publishes_the_bound():
    prop = ReadoutCal.model_json_schema()["properties"]["trig_offset"]
    assert prop["minimum"] == 0 and prop["maximum"] == 10
    assert prop["description"].startswith("µs — ")


# -- the wiring fallback -------------------------------------------------------------------


@pytest.mark.parametrize("value", OUT_OF_RANGE, ids=repr)
def test_the_wiring_fallback_refuses_an_out_of_range_value_naming_the_unit(value):
    with pytest.raises(ValidationError) as err:
        ReadoutWiring(dac_ch=5, adc_chs=[0], trig_offset=value)
    assert "trig_offset" in _locs(err.value)
    msgs = _messages(err.value, "trig_offset")
    assert msgs and all("µs" in m for m in msgs), msgs


@pytest.mark.parametrize("value", [*IN_RANGE, None], ids=repr)
def test_the_wiring_fallback_accepts_an_in_range_value_or_none(value):
    assert ReadoutWiring(dac_ch=5, adc_chs=[0], trig_offset=value).trig_offset == value


def test_the_wiring_schema_publishes_the_bound():
    prop = ReadoutWiring.model_json_schema()["properties"]["trig_offset"]
    (number,) = [b for b in prop["anyOf"] if b.get("type") == "number"]
    assert number["minimum"] == 0 and number["maximum"] == 10
    assert prop["description"].startswith("µs — ")


# -- the YAML door: both fields, through Device.load ---------------------------------------


def _write_variant(tmp_path, old: str, new: str):
    text = MULTIMODE_YAML.read_text(encoding="utf-8")
    assert text.count(old) == 1, f"{old!r} must appear exactly once in {MULTIMODE_YAML.name}"
    text = text.replace(old, new)
    snapshot = (MULTIMODE_YAML.parent / "../_fixtures/soccfg_v2_cqed.json").resolve()
    text = re.sub(r"^soccfg_snapshot: .*$", f"soccfg_snapshot: {snapshot}", text, flags=re.M)
    path = tmp_path / "device.yaml"
    path.write_text(text, encoding="utf-8")
    return path


_CALIB_LINE = "    trig_offset: 0.5      #"
_WIRING_LINE = "    adc_chs: [0]\n"


@pytest.mark.parametrize("value", ["400", "-5"])
def test_a_device_yaml_with_an_out_of_range_calibration_value_fails_to_load(tmp_path, value):
    path = _write_variant(tmp_path, _CALIB_LINE, f"    trig_offset: {value}      #")
    with pytest.raises(ValidationError) as err:
        Device.load(path)
    assert "readout.trig_offset" in _locs(err.value)
    assert all("µs" in m for m in _messages(err.value, "readout.trig_offset"))


@pytest.mark.parametrize("value", ["400", "-5"])
def test_a_device_yaml_with_an_out_of_range_wiring_fallback_fails_to_load(tmp_path, value):
    path = _write_variant(tmp_path, _WIRING_LINE, f"{_WIRING_LINE}    trig_offset: {value}\n")
    with pytest.raises(ValidationError) as err:
        Device.load(path)
    locs = _locs(err.value)
    assert any(loc.endswith("readout.trig_offset") for loc in locs), sorted(locs)
    assert any("µs" in e["msg"] for e in err.value.errors())


@pytest.mark.parametrize("value", ["0", "0.5", "10"])
def test_a_device_yaml_with_an_in_range_wiring_fallback_loads(tmp_path, value):
    dev = Device.load(_write_variant(tmp_path, _WIRING_LINE,
                                     f"{_WIRING_LINE}    trig_offset: {value}\n"))
    assert dev.wiring.readout.trig_offset == float(value)


# -- the write-back doors ------------------------------------------------------------------


def _store_with_base(tmp_path, dev):
    store = FileConfigStore(tmp_path / "store")
    base = dev.to_config()
    store.put(base)
    store.set_ref("main", base.content_id())
    return store, base


def _snapshot(root):
    return {p.relative_to(root): p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()}


@pytest.mark.parametrize("value", [400, -5])
def test_a_store_write_back_of_an_out_of_range_value_is_refused(tmp_path, dev, value):
    store, base = _store_with_base(tmp_path, dev)
    before = _snapshot(tmp_path / "store")
    with pytest.raises(ValidationError) as err:
        record_calibration(store, "main", base, {"readout.trig_offset": value}, source="bad")
    assert any(loc.endswith("readout.trig_offset") for loc in _locs(err.value))
    assert _snapshot(tmp_path / "store") == before
    assert store.resolve_ref("main") == base.content_id()


@pytest.mark.parametrize("tof_us", [400.0, -5.0])
def test_a_time_of_flight_write_back_out_of_range_is_refused_and_stored_nowhere(
    tmp_path, dev, tof_us
):
    store, base = _store_with_base(tmp_path, dev)
    before = _snapshot(tmp_path / "store")
    tof = TimeOfFlight(dev, window_us=3.0)
    fit = FitResult(params={}, stderr={}, derived={"tof_us": tof_us}, success=True)
    with pytest.raises(ValidationError) as err:
        tof.write_back(fit, store, "main")
    assert any("µs" in e["msg"] for e in err.value.errors())
    assert _snapshot(tmp_path / "store") == before        # no object, no provenance, no ref move
    assert tof.device is dev                              # the session keeps its old belief
    assert dev.calib.readout.trig_offset == 0.5


def test_a_time_of_flight_write_back_at_the_bound_is_stored(tmp_path, dev):
    store, _ = _store_with_base(tmp_path, dev)
    tof = TimeOfFlight(dev, window_us=3.0)
    fit = FitResult(params={}, stderr={}, derived={"tof_us": 10.0}, success=True)
    assert tof.write_back(fit, store, "main").calib.readout.trig_offset == 10.0
