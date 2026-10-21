"""``MeasurementResult`` wire form + the machine-readable contract schemas (D35).

The wire form is what leaves the process (a JSON payload for an agent, a store, a job server);
the schemas are how a consumer knows its shape without reading Python. Both are asserted here
against *real* instances — a fit off the real fitter, real device YAMLs, a real operator export —
so a schema cannot drift away from the objects it claims to describe.
"""

from __future__ import annotations

import json
from dataclasses import asdict, fields
from pathlib import Path

import numpy as np
import pytest
import yaml

from strumento.core.config import DeviceConfig
from strumento.core.device import Device
from strumento.core.fitters import FitResult, fit_rabi
from strumento.core.result import MeasurementResult
from strumento.core.schemas import CONTRACTS, DeviceSource, SchemaError, schema_for, validate
from tests._paths import SHIPPED_DEVICE_YAMLS

ROOT = Path(__file__).parents[2]
DEVICE_YAMLS = SHIPPED_DEVICE_YAMLS  # git-tracked only: a local untracked folder never leaks in


# -- MeasurementResult wire form -----------------------------------------------------------


def _rich_result() -> MeasurementResult:
    rng = np.random.default_rng(0)
    shots = (rng.random((3, 8)) < 0.4).astype(np.int8)
    return MeasurementResult(
        xpts=np.linspace(0.0, 1.0, 3),
        avgi=np.array([1.0, 2.0, 3.0]),
        avgq=np.array([-1.0, -2.0, -3.0]),
        shots=shots,
        heralds={"reset": np.array([True, False, True, True, True, False, True, True])},
        herald_lanes=("reset",),
        meta={"reps": 100, "environment": {"mxc_mk": 0.012}},
    )


def test_to_wire_is_json_safe():
    wire = _rich_result().to_wire()
    text = json.dumps(wire)  # no default= hook: every leaf is already a JSON primitive
    assert json.loads(text) == wire
    assert isinstance(wire["avgi"], list) and isinstance(wire["shots"], list)
    assert wire["heralds"]["reset"][0] is True  # heralds are bool lists, not 1/0 ints
    assert wire["herald_lanes"] == ["reset"]


def test_wire_round_trip_preserves_shots_and_heralds():
    original = _rich_result()
    back = MeasurementResult.from_wire(original.to_wire())
    assert np.array_equal(np.asarray(back.xpts), np.asarray(original.xpts))
    assert np.array_equal(np.asarray(back.avgi), np.asarray(original.avgi))
    assert np.array_equal(np.asarray(back.avgq), np.asarray(original.avgq))
    assert np.array_equal(np.asarray(back.shots), np.asarray(original.shots))
    assert set(back.heralds) == {"reset"}
    assert np.array_equal(back.heralds["reset"], original.heralds["reset"])
    assert np.asarray(back.heralds["reset"]).dtype == bool
    assert back.herald_lanes == ("reset",)
    assert back.meta == original.meta


def test_wire_is_idempotent():
    wire = _rich_result().to_wire()
    assert MeasurementResult.from_wire(wire).to_wire() == wire


def test_wire_handles_the_single_point_case():
    mr = MeasurementResult(xpts=None, avgi=0.5, avgq=-0.5)
    wire = mr.to_wire()
    assert wire["xpts"] is None and wire["shots"] is None and wire["heralds"] == {}
    back = MeasurementResult.from_wire(wire)
    assert back.xpts is None and back.avgi == 0.5 and back.herald_lanes == ()


def test_meta_is_coerced_json_safe():
    mr = MeasurementResult(
        xpts=None,
        avgi=[0.0],
        avgq=[0.0],
        meta={
            "mask": np.array([True, False]),      # arrays → lists
            "survival": np.float64(0.5),          # numpy scalars → python
            "lanes": ("reset", "leakage"),        # tuples → lists
            "path": Path("/tmp/x"),               # anything else → str
            "nested": {"n": np.int64(3)},
        },
    )
    meta = mr.to_wire()["meta"]
    json.dumps(meta)
    assert meta["mask"] == [True, False]
    assert meta["survival"] == 0.5 and isinstance(meta["survival"], float)
    assert meta["lanes"] == ["reset", "leakage"]
    assert meta["path"] == "/tmp/x"
    assert meta["nested"]["n"] == 3


def test_postselected_result_survives_the_wire():
    # postselect stamps meta and filters shots — the composed object must still serialize.
    mr = _rich_result()
    flat = MeasurementResult(
        xpts=None,
        avgi=[0.0],
        avgq=[0.0],
        shots=np.asarray(mr.shots)[0],
        heralds={"reset": mr.heralds["reset"]},
        herald_lanes=("reset",),
    )
    wire = flat.postselect().to_wire()
    json.dumps(wire)
    assert wire["meta"]["postselect_survival"] == pytest.approx(6 / 8)
    assert len(wire["shots"]) == 6


# -- schemas -------------------------------------------------------------------------------


def test_every_contract_emits_an_object_schema():
    assert set(CONTRACTS) == {"device", "fit", "result", "operator-model"}
    for name in CONTRACTS:
        schema = schema_for(name)
        json.dumps(schema)  # a schema is itself a JSON document
        assert schema["type"] == "object"
        assert schema["properties"], name


def test_unknown_contract_is_typed_and_lists_valid_names():
    with pytest.raises(SchemaError, match="unknown contract"):
        schema_for("qubit")
    try:
        schema_for("qubit")
    except SchemaError as err:
        for name in CONTRACTS:
            assert name in str(err)


def test_the_calibration_contract_is_pack_scoped_and_core_asks_the_pack():
    from strumento.core.schemas import PACK_CONTRACTS, calibration_schema
    from strumento.packs.cqed import CALIBRATION

    assert "calibration" in PACK_CONTRACTS and "calibration" not in CONTRACTS
    assert calibration_schema("cqed") == CALIBRATION.model_json_schema()
    with pytest.raises(SchemaError, match="pack-scoped.*installed packs: .*cqed"):
        schema_for("calibration")  # no pack named → a helpful error, not "unknown contract"


def test_a_pack_without_a_calibration_model_has_no_calibration_contract(monkeypatch):
    import types

    import strumento.core.device as device_mod
    from strumento.core.schemas import calibration_schema

    bare = types.ModuleType("bare_pack")  # a pack that exports no CALIBRATION
    monkeypatch.setattr(device_mod, "installed_packs", lambda: ["bare"])
    monkeypatch.setattr(device_mod, "_load_pack", lambda name: bare)
    with pytest.raises(SchemaError, match="'bare' exports no CALIBRATION"):
        calibration_schema("bare")


def test_device_schema_matches_the_config_dataclass():
    # the schema is a second description of DeviceConfig's source form; assert it cannot drift
    assert set(DeviceSource.model_fields) == {f.name for f in fields(DeviceConfig)}
    props = schema_for("device")["properties"]
    assert set(props) == set(DeviceSource.model_fields)
    assert set(schema_for("device")["required"]) == {"pack", "topology", "wiring", "calibration"}


@pytest.mark.parametrize("path", DEVICE_YAMLS, ids=lambda p: p.parent.name)
def test_shipped_device_yaml_validates(path: Path):
    raw = yaml.safe_load(path.read_text())
    validate("device", raw)  # the on-disk source form (soccfg_snapshot as a path)
    # …and every key it carries is a property the emitted schema declares
    assert set(raw) <= set(schema_for("device")["properties"])


@pytest.mark.parametrize("path", DEVICE_YAMLS, ids=lambda p: p.parent.name)
def test_loaded_device_config_validates(path: Path):
    validate("device", Device.load(path).to_config().to_dict())  # snapshot resolved to a dict


def test_device_schema_rejects_a_malformed_instance():
    with pytest.raises(SchemaError, match="device"):
        validate("device", {"topology": [], "wiring": {}, "calibration": {}})  # no pack


def test_fit_schema_validates_a_real_fit():
    x = np.linspace(0, 30000, 61)
    y = 0.5 + 0.4 * np.cos(2 * np.pi * x / 16000.0)
    fit = fit_rabi(x, y)
    validate("fit", asdict(fit))
    assert set(schema_for("fit")["required"]) == {"params", "stderr", "derived", "success"}


def test_fit_schema_accepts_json_values_in_derived():
    """``derived`` carries JSON values, not only numbers (issue #174, option (a))."""
    derived = {
        "protocol": "bb1",
        "T1": 42.5,
        "n": 3,
        "refused": None,
        "converged": True,
        "confusion_matrix": [[0.9, 0.1], [0.2, 0.8]],
        "provenance": {"source": "calibration", "keys": ["a", "b"]},
    }
    fit = FitResult(params={}, stderr={}, derived=derived, success=True)
    assert validate("fit", asdict(fit)).derived == derived


def test_fit_schema_still_describes_derived_numbers_as_numbers():
    schema = schema_for("fit")
    value = schema["properties"]["derived"]["additionalProperties"]
    if "$ref" in value:
        value = schema["$defs"][value["$ref"].rsplit("/", 1)[-1]]
    kinds = {branch.get("type") for branch in value["anyOf"]}
    assert {"number", "integer", "string", "boolean", "null", "array", "object"} <= kinds


def test_fit_schema_rejects_a_non_json_derived_value():
    with pytest.raises(SchemaError, match="fit"):
        validate("fit", {"params": {}, "stderr": {}, "derived": {"x": object()}, "success": True})


def test_result_schema_validates_a_real_wire_form():
    validate("result", _rich_result().to_wire())
    validate("result", MeasurementResult(xpts=None, avgi=0.0, avgq=0.0).to_wire())
    with pytest.raises(SchemaError):
        validate("result", {"avgi": [0.0]})  # avgq missing


def test_operator_model_schema_validates_a_real_export():
    dev = Device.load(ROOT / "devices" / "loopback_demo" / "device.yaml")
    model = dev.to_operator_model(frame="rotating", cutoffs={"qubit": 3}, include=["qubit"])
    validate("operator-model", model.to_dict())


def test_module_imports_without_qick():
    import subprocess
    import sys

    code = (
        "import sys; sys.modules['qick']=None; "
        "import strumento.core.schemas, strumento.core.result; print('ok')"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0 and "ok" in r.stdout, r.stderr
