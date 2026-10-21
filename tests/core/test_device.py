"""Tests for Device and loading a device instance from YAML (D21 namespaces)."""

import subprocess
import sys

import pytest

from strumento.core.device import Device
from strumento.core.wiring import LineRef
from strumento.packs.cqed.library.transmon import TransmonOps
from tests._paths import DEVICES

YAML = DEVICES / "multimode_demo" / "device.yaml"


def test_load_resolves_qubit_drive():
    dev = Device.load(YAML)
    assert dev.wiring.resolve(LineRef("qubit", "drive"), freq_mhz=3562).dac_ch == 1
    assert dev.wiring.resolve(LineRef("qubit", "sideband"), freq_mhz=2010).dac_ch == 5


def test_load_is_locale_independent(tmp_path, monkeypatch):
    """Device files are UTF-8 by authorship (µ, — in comments/values); loading must never
    depend on the interpreter's locale (#45). Reproduced faithfully in a subprocess with
    UTF-8 mode off and a C locale — the condition a bare/embedded interpreter (PythonCall
    from Strumento.jl, a container with LANG unset) actually presents. Monkeypatching
    locale in-process does NOT reproduce it on 3.14 (the locale layer short-circuits to
    UTF-8), so this runs the real interpreter the real way."""
    import os
    import shutil

    instance = tmp_path / "multimode_demo"
    instance.mkdir()
    yaml_copy = instance / "device.yaml"  # one level down: its ../_fixtures ref stays inside tmp_path
    shutil.copy(YAML, yaml_copy)
    fixtures = tmp_path / "_fixtures"
    fixtures.mkdir()
    for snap in (YAML.parent.parent / "_fixtures").glob("*.json"):
        shutil.copy(snap, fixtures / snap.name)

    code = (
        "from strumento.core.device import Device\n"
        f"Device.load({str(yaml_copy)!r})\n"
    )
    env = {k: v for k, v in os.environ.items() if not k.startswith(("LC_", "LANG", "PYTHON"))}
    env.update({"LANG": "C", "LC_ALL": "C", "PYTHONUTF8": "0"})
    proc = subprocess.run(
        [sys.executable, "-X", "utf8=0", "-c", code],
        capture_output=True, text=True, env=env,
    )
    assert proc.returncode == 0, (
        f"Device.load failed under a C-locale interpreter (the embedded-Python condition):\n"
        f"{proc.stderr[-800:]}"
    )


def test_calibration_populated():
    dev = Device.load(YAML)
    assert dev.calib.qubit.f_ge == pytest.approx(4000.0)
    assert dev.calib.qubit.pulses["pi_ge"].gain == 8192
    # 7000.0, not the fixture's earlier 750.0: that value sat outside this
    # generator's declared nyquist:2 band ([4792, 9585] MHz on this fixture's
    # soccfg) and was implausibly low for a 3D-cQED readout resonator anyway
    # (see devices/multimode_demo/device.yaml's header).
    assert dev.calib.readout.frequency == pytest.approx(7000.0)


def test_soccfg_snapshot_is_plain_dict():
    dev = Device.load(YAML)
    assert isinstance(dev.soccfg_snapshot, dict)
    assert dev.soccfg_snapshot["board"] == "ZCU216"
    # maxv is present so the pack can convert int-code gains -> fractions (D23)
    assert dev.soccfg_snapshot["gens"][0]["maxv"] == 32766


def test_topology_composed_and_namespace_bound():
    dev = Device.load(YAML)
    # 1 transmon + 1 cavity + 2 storage + coupler + readout = 6 elements
    assert len(dev.elements) == 6
    # D21: dev.qubit is a TransmonOps bound to the *named* transmon element
    assert isinstance(dev.qubit, TransmonOps)
    assert dev.qubit.element is dev.elements["qubit"]
    assert dev.qubit.device is dev
    # modes and the coupler carry ops classes and bind too
    from strumento.packs.cqed.library.coupler import CouplerOps
    from strumento.packs.cqed.library.mode import ModeOps

    assert isinstance(dev.cavity, ModeOps)
    assert isinstance(dev.coupler, CouplerOps)


def test_all_elements_are_wireable():
    # load() validates every element's LineRefs against the wiring; a good instance loads clean.
    dev = Device.load(YAML)
    for el in dev.elements.values():
        for ref in el.lines():
            # each declared line/role must exist in the wiring
            assert ref.line in dev.wiring.lines
            assert any(g.role == ref.role for g in dev.wiring.lines[ref.line].generators)


def test_import_does_not_pull_in_qick():
    # core/device.py must be importable with qick absent (pure layer).
    code = "import sys; sys.modules['qick']=None; import strumento.core.device; print('ok')"
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0 and "ok" in r.stdout, r.stderr
