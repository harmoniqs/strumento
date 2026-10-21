"""CLI — every command drives main(argv) in-process; JSON output is the agent contract."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from strumento.cli import main
from tests._paths import SHIPPED_DEVICE_YAMLS

ROOT = Path(__file__).parents[2]
MULTIMODE = str(ROOT / "devices" / "multimode_demo" / "device.yaml")
LOOPBACK = str(ROOT / "devices" / "loopback_demo" / "device.yaml")
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
    assert out["wiring"]["qubit"][0]["dac_ch"] == 1
    assert out["readout"] == {"dac_ch": 0, "adc_chs": [0]}


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
    # multimode_demo now ships a v2 snapshot (soccfg_v2_cqed.json), so realized readback
    # returns actual values when qick is installed (without qick it is None)
    if out["realized"] is not None:
        assert "freq_mhz_actual" in out["realized"]


def test_compile_realized_reads_back_on_v2_loopback(capsys):
    pytest.importorskip("qick")
    assert main(["compile", LOOPBACK, "--json"]) == 0
    out = _json_out(capsys)
    r = out["realized"]
    assert r is not None and "unavailable" not in r
    assert r["freq_mhz_actual"] == pytest.approx(4000.0, abs=1e-3)
    assert r["gain_frac"] == pytest.approx(8192 / 32766)
    assert r["length_us_actual"] >= 0.12


# -- check: the exit code carries the verdict (issue #129) --------------------------------
#
# Purpose-built tmp devices, derived from loopback_demo, so these cases don't flip when a
# shipped demo's wiring or snapshot changes.


def _check_device(tmp_path, *, dac_ch: int = 0, snapshot: dict | None = None,
                  attach: bool = True, adc_chs: list[int] | None = None) -> str:
    """A loopback_demo copy: ``dac_ch`` rewires the drive, ``adc_chs`` the readout ADCs;
    ``snapshot`` replaces the soccfg."""
    text = Path(LOOPBACK).read_text(encoding="utf-8")
    text = text.replace("{role: drive, dac_ch: 0,", f"{{role: drive, dac_ch: {dac_ch},")
    if adc_chs is not None:
        text = text.replace("adc_chs: [0]", f"adc_chs: {adc_chs}")
    if snapshot is not None:
        snap = tmp_path / "soccfg.json"
        snap.write_text(json.dumps(snapshot), encoding="utf-8")
        text = text.replace("../_fixtures/soccfg_v2_testbench.json", str(snap))
    else:
        text = text.replace("../_fixtures/soccfg_v2_testbench.json", V2_SNAP)
    if not attach:
        text = "\n".join(ln for ln in text.splitlines() if not ln.startswith("soccfg_snapshot:"))
    path = tmp_path / "device.yaml"
    path.write_text(text, encoding="utf-8")
    return str(path)


def _testbench(**tproc) -> dict:
    snap = json.loads(Path(V2_SNAP).read_text(encoding="utf-8"))
    snap["tprocs"][0].update(tproc)
    return snap


def test_check_errors_exit_nonzero(tmp_path, capsys):
    yaml = _check_device(tmp_path, dac_ch=7)  # the testbench has only gen 0
    assert main(["check", yaml]) != 0
    assert "FAIL" in capsys.readouterr().out


def test_check_clean_pass_exits_zero(tmp_path, capsys):
    fabric = _testbench()["gens"][0]["f_fabric"]
    yaml = _check_device(tmp_path, snapshot=_testbench(f_time=fabric))
    assert main(["check", yaml, "--json"]) == 0
    out = _json_out(capsys)
    assert out["ok"] is True and out["issues"] == [] and out["warnings"] == []


def test_check_warnings_only_exits_zero(tmp_path, capsys):
    yaml = _check_device(tmp_path, snapshot=_testbench(f_time=1.0))  # f_time ≠ any f_fabric
    assert main(["check", yaml, "--json"]) == 0
    out = _json_out(capsys)
    assert out["ok"] is True and out["issues"] == []
    assert any("f_time" in w for w in out["warnings"])


# -- check: channels are snapshot list indices, not tProc ports (issue #135) --------------
#
# qick's declare_gen / declare_readout take the index into the snapshot's gens / readouts
# list; ``tproc_ch`` is the tProc port behind it. These snapshots pull the two apart.


def _readout_on_port(port: int) -> dict:
    """The testbench, with its single readout moved to tProc input port ``port``."""
    snap = _testbench()
    snap["readouts"][0]["tproc_ch"] = port
    return snap


def _gens_on_ports(*ports: int) -> dict:
    """The testbench, its one generator cloned once per entry of ``ports`` (tproc_ch = port)."""
    snap = _testbench()
    proto = snap["gens"][0]
    snap["gens"] = [{**proto, "tproc_ch": port} for port in ports]
    return snap


def test_check_readout_on_port_6_accepts_adc_index_0(tmp_path, capsys):
    yaml = _check_device(tmp_path, snapshot=_readout_on_port(6), adc_chs=[0])
    assert main(["check", yaml, "--json"]) == 0
    out = _json_out(capsys)
    assert out["ok"] is True and out["issues"] == []


def test_check_readout_on_port_6_rejects_adc_index_1(tmp_path, capsys):
    yaml = _check_device(tmp_path, snapshot=_readout_on_port(6), adc_chs=[1])
    assert main(["check", yaml, "--json"]) == 1
    out = _json_out(capsys)
    assert out["ok"] is False
    [issue] = out["issues"]
    assert "adc_ch=1" in issue and "[0]" in issue and "6" not in issue.split("have")[-1]


def test_check_readout_port_number_is_not_a_valid_adc_index(tmp_path, capsys):
    yaml = _check_device(tmp_path, snapshot=_readout_on_port(6), adc_chs=[6])
    assert main(["check", yaml, "--json"]) == 1
    assert any("adc_ch=6" in i for i in _json_out(capsys)["issues"])


def test_check_dac_ch_validated_by_gen_list_position(tmp_path, capsys):
    # gens at list positions 0, 1 sit on tProc ports 5, 3 — dac_ch 1 is position 1: valid
    yaml = _check_device(tmp_path, dac_ch=1, snapshot=_gens_on_ports(5, 3))
    assert main(["check", yaml, "--json"]) == 0
    out = _json_out(capsys)
    assert out["ok"] is True and out["issues"] == []


def test_check_dac_ch_equal_to_a_tproc_port_but_out_of_range_fails(tmp_path, capsys):
    yaml = _check_device(tmp_path, dac_ch=5, snapshot=_gens_on_ports(5, 3))
    assert main(["check", yaml, "--json"]) == 1
    out = _json_out(capsys)
    [issue] = out["issues"]
    assert "dac_ch=5" in issue and "have [0, 1]" in issue


def test_check_reports_counts_from_the_snapshot_lists(tmp_path, capsys):
    # two gens on the same tProc port are still two generators
    yaml = _check_device(tmp_path, snapshot=_gens_on_ports(4, 4))
    assert main(["check", yaml, "--json"]) == 0
    out = _json_out(capsys)
    assert out["n_gens"] == 2 and out["n_readouts"] == 1


def test_check_loopback_demo_passes(capsys):
    assert main(["check", LOOPBACK]) == 0
    assert capsys.readouterr().out.startswith("PASS")


def _declares_snapshot(device_yaml: Path) -> bool:
    """True when the device YAML names a ``soccfg_snapshot`` — the only filter on the list below."""
    doc = yaml.safe_load(device_yaml.read_text(encoding="utf-8")) or {}
    return bool(doc.get("soccfg_snapshot"))


# Every shipped demo that declares a snapshot must pass `check` against it. Devices without one
# (today the spin and tweezer demos) can't be checked at all, so they are filtered out here by
# that declaration alone — a demo picks this test up automatically the moment it gains one.
SNAPSHOT_DEMOS = [p for p in SHIPPED_DEVICE_YAMLS if _declares_snapshot(p)]


def test_snapshot_demo_filter_selects_the_cqed_and_loopback_demos():
    names = {p.parent.name for p in SNAPSHOT_DEMOS}
    assert {"loopback_demo", "multimode_demo"} <= names


@pytest.mark.parametrize("device_yaml", SNAPSHOT_DEMOS, ids=lambda p: p.parent.name)
def test_shipped_demo_with_a_snapshot_passes_check(device_yaml, capsys):
    code = main(["check", str(device_yaml), "--json"])
    out = _json_out(capsys)
    assert out["issues"] == []
    assert out["ok"] is True and code == 0


def test_check_no_snapshot_exits_nonzero(tmp_path, capsys):
    yaml = _check_device(tmp_path, attach=False)
    assert main(["check", yaml]) != 0
    assert "no soccfg snapshot" in capsys.readouterr().out


@pytest.mark.parametrize("case", ["errors", "pass", "warnings", "no-snapshot"])
def test_check_json_verdict_agrees_with_exit_code(tmp_path, capsys, case):
    yaml = {
        "errors": lambda: _check_device(tmp_path, dac_ch=7),
        "pass": lambda: _check_device(tmp_path),
        "warnings": lambda: _check_device(tmp_path, snapshot=_testbench(f_time=1.0)),
        "no-snapshot": lambda: _check_device(tmp_path, attach=False),
    }[case]()
    code = main(["check", yaml, "--json"])
    out = _json_out(capsys)
    assert isinstance(out["ok"], bool)
    assert out["ok"] is (code == 0)
    assert out["ok"] is (case in ("pass", "warnings"))
