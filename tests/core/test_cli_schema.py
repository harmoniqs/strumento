"""``strumento schema`` (D35) — the typed contracts as a CLI verb.

Same conventions as every other verb: JSON with ``--json``, errors as data (never a traceback),
nonzero exit on failure.
"""

from __future__ import annotations

import json

import pytest
import yaml

from strumento.cli import main
from strumento.core.schemas import CONTRACTS
from tests._paths import SHIPPED_DEVICE_YAMLS

DEVICE_YAMLS = SHIPPED_DEVICE_YAMLS  # git-tracked only: a local untracked folder never leaks in


def _json_out(capsys) -> dict:
    return json.loads(capsys.readouterr().out)


@pytest.mark.parametrize("contract", sorted(CONTRACTS))
def test_schema_emits_each_contract(contract, capsys):
    assert main(["schema", contract, "--json"]) == 0
    out = _json_out(capsys)
    assert out["ok"] and out["contract"] == contract
    assert out["schema"]["type"] == "object" and out["schema"]["properties"]


def test_schema_text_mode_prints_the_schema_document(capsys):
    assert main(["schema", "result"]) == 0
    schema = json.loads(capsys.readouterr().out)  # the text form is the schema itself
    assert schema["type"] == "object" and "avgi" in schema["properties"]


def test_unknown_contract_is_errors_as_data(capsys):
    assert main(["schema", "nope", "--json"]) == 1
    out = _json_out(capsys)
    assert out["ok"] is False
    assert "nope" in out["error"]
    for name in CONTRACTS:
        assert name in out["error"]


@pytest.mark.parametrize("path", DEVICE_YAMLS, ids=lambda p: p.parent.name)
def test_shipped_devices_agree_with_the_emitted_device_schema(path, capsys):
    """The D35 acceptance gate: every shipped instance matches the schema the CLI emits."""
    assert main(["schema", "device", "--json"]) == 0
    schema = _json_out(capsys)["schema"]
    raw = yaml.safe_load(path.read_text())
    assert set(schema["required"]) <= set(raw)      # every required field present
    assert set(raw) <= set(schema["properties"])    # no undeclared field

    from strumento.core.schemas import validate

    validate("device", raw)  # and it validates against the model the schema is emitted from


# -- the pack-scoped calibration contract (#184) ------------------------------------------


def _installed_packs() -> list[str]:
    from importlib.metadata import entry_points

    return sorted(ep.name for ep in entry_points(group="strumento.packs"))


def test_calibration_schema_for_a_pack_carries_its_units(capsys):
    """``schema calibration --pack cqed`` publishes the pack's CALIBRATION model, units and all."""
    assert main(["schema", "calibration", "--pack", "cqed", "--json"]) == 0
    out = _json_out(capsys)
    assert out["ok"] and out["contract"] == "calibration" and out["pack"] == "cqed"
    defs = out["schema"]["$defs"]
    assert "MHz" in defs["QubitCal"]["properties"]["f_ge"]["description"]
    assert "µs" in defs["ReadoutCal"]["properties"]["relax_delay"]["description"]


@pytest.mark.parametrize(
    "model, field, cap",
    [
        ("QubitCal", "max_drive_gain", "QUBIT_DRIVE_GAIN_CAP"),
        ("ReadoutCal", "max_readout_gain", "READOUT_GAIN_CAP"),
        ("ModeCal", "max_displace_gain", "DISPLACEMENT_GAIN_CAP"),
    ],
)
def test_calibration_schema_publishes_the_gain_ceiling_bounds(model, field, cap, capsys):
    """Each per-element gain limit is bounded ``0 < limit <= cap`` and marked a ceiling (#190)."""
    from strumento.packs.cqed import safety

    assert main(["schema", "calibration", "--pack", "cqed", "--json"]) == 0
    prop = _json_out(capsys)["schema"]["$defs"][model]["properties"][field]
    number = next(b for b in prop["anyOf"] if b.get("type") == "number")
    assert number["exclusiveMinimum"] == 0
    assert number["maximum"] == getattr(safety, cap) <= 1.0
    assert prop["gain_ceiling"] is True


def test_calibration_schema_publishes_the_trig_offset_bound(capsys):
    """``readout.trig_offset`` is bounded ``0 <= trig_offset <= 10`` µs (#195)."""
    from strumento.core.wiring import TRIG_OFFSET_MAX_US

    assert main(["schema", "calibration", "--pack", "cqed", "--json"]) == 0
    prop = _json_out(capsys)["schema"]["$defs"]["ReadoutCal"]["properties"]["trig_offset"]
    assert prop["minimum"] == 0
    assert prop["maximum"] == TRIG_OFFSET_MAX_US == 10
    assert prop["description"].startswith("µs — ")


def test_device_schema_publishes_the_wiring_trig_offset_bound(capsys):
    """The wiring fallback ``wiring.readout.trig_offset`` carries the same bound (#195)."""
    assert main(["schema", "device", "--json"]) == 0
    schema = _json_out(capsys)["schema"]
    prop = schema["$defs"]["ReadoutWiring"]["properties"]["trig_offset"]
    number = next(b for b in prop["anyOf"] if b.get("type") == "number")
    assert number["minimum"] == 0 and number["maximum"] == 10


@pytest.mark.parametrize("pack", _installed_packs())
def test_every_installed_pack_publishes_its_calibration_schema(pack, capsys):
    assert main(["schema", "calibration", "--pack", pack, "--json"]) == 0
    out = _json_out(capsys)
    assert out["ok"] and out["pack"] == pack
    assert out["schema"]["type"] == "object" and out["schema"]["properties"]


@pytest.mark.parametrize(
    "argv",
    [["schema", "calibration"], ["schema", "calibration", "--pack", "nosuch"]],
    ids=["no-pack", "unknown-pack"],
)
def test_calibration_without_a_known_pack_is_errors_as_data(argv, capsys):
    assert main([*argv, "--json"]) == 1
    out = _json_out(capsys)
    assert out["ok"] is False and "calibration" in out["error"]
    for pack in _installed_packs():
        assert pack in out["error"]  # tells the caller what it could have asked for


@pytest.mark.parametrize("contract", sorted(CONTRACTS))
def test_pack_agnostic_contracts_keep_their_payload_shape(contract, capsys):
    """Adding the pack-scoped contract changes nothing about the four pack-agnostic ones."""
    assert main(["schema", contract, "--json"]) == 0
    assert set(_json_out(capsys)) == {"ok", "contract", "schema"}


def test_a_pack_on_a_pack_agnostic_contract_is_errors_as_data(capsys):
    assert main(["schema", "result", "--pack", "cqed", "--json"]) == 1
    out = _json_out(capsys)
    assert out["ok"] is False and "calibration" in out["error"]
