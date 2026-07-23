"""CLI — every command drives main(argv) in-process; JSON output is the agent contract."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from strumento.cli import main

ROOT = Path(__file__).parents[2]
MULTIMODE = str(ROOT / "devices" / "multimode_demo" / "device.yaml")
V1_SNAP = str(ROOT / "devices" / "_fixtures" / "soccfg_v1_zcu216.json")
V2_SNAP = str(ROOT / "devices" / "_fixtures" / "soccfg_v2_testbench.json")


def _json_out(capsys) -> dict:
    return json.loads(capsys.readouterr().out)


def test_packs_lists_entry_points(capsys):
    assert main(["packs", "--json"]) == 0
    out = _json_out(capsys)
    assert {"cqed", "spin", "tweezer"} <= set(out["packs"])


def test_validate_ok_and_error(tmp_path, capsys):
    assert main(["validate", MULTIMODE, "--json"]) == 0
    out = _json_out(capsys)
    assert out["ok"] and "qubit" in out["elements"]

    bad = tmp_path / "bad.yaml"
    bad.write_text("pack: cqed\nwiring: {lines: {}, readout: {dac_ch: 0, adc_chs: [0]}}\n"
                   "topology: []\ncalibration: {nope: 1}\n")
    assert main(["validate", str(bad), "--json"]) == 1
    out = _json_out(capsys)
    assert not out["ok"] and "error" in out


def test_info_reports_wiring_and_namespaces(capsys):
    assert main(["info", MULTIMODE, "--json"]) == 0
    out = _json_out(capsys)
    assert out["elements"]["qubit"] == "Transmon"
    assert "sideband" in out["namespaces"]
    assert out["wiring"]["qubit"][0]["dac_ch"] == 2
    assert out["readout"] == {"dac_ch": 5, "adc_chs": [0]}


def test_soccfg_v1_vs_v2_verdict(capsys):
    assert main(["soccfg", V1_SNAP, "--json"]) == 0
    v1 = _json_out(capsys)
    assert v1["is_v2"] is False and v1["board"] == "ZCU216"

    assert main(["soccfg", V2_SNAP, "--json"]) == 0
    v2 = _json_out(capsys)
    assert v2["is_v2"] is True and v2["tproc_type"] == "qick_processor"


@pytest.mark.parametrize("kind,gen,key", [
    ("t1", lambda t: 0.2 + 1.5 * np.exp(-t / 42.0), "T1"),
    ("rabi", lambda t: 0.5 + 0.4 * np.exp(-t / 300.0) * np.cos(2 * np.pi * t / 120.0), "pi_gain"),
])
def test_fit_csv(tmp_path, capsys, kind, gen, key):
    t = np.linspace(0.0, 240.0, 121)
    csv = tmp_path / "data.csv"
    np.savetxt(csv, np.column_stack([t, gen(t)]), delimiter=",", header="x,y", comments="")
    assert main(["fit", kind, str(csv), "--json"]) == 0
    out = _json_out(capsys)
    assert out["success"] and key in out["derived"]


def test_compile_lists_lowering_calls(capsys):
    assert main(["compile", MULTIMODE, "--json"]) == 0
    out = _json_out(capsys)
    calls = [c["call"] for c in out["calls"]]
    assert "declare_gen" in calls and "add_pulse" in calls and "add_loop" in calls
    # qick is installed in the dev env, so realized values are present; without qick it's None
    if out["realized"] is not None:
        assert out["realized"]["cycles"] > 0
