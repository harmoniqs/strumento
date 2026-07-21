"""Tests for Device + instance load (spec §5, plan Task 5, D21)."""

import subprocess
import sys
from pathlib import Path

import pytest

from strumento.core.device import Device
from strumento.core.wiring import LineRef
from strumento.packs.cqed.library.transmon import TransmonOps

YAML = Path(__file__).parents[2] / "devices" / "schuster_multimode" / "device.yaml"


def test_load_resolves_qubit_drive():
    dev = Device.load(YAML)
    assert dev.wiring.resolve(LineRef("qubit", "drive"), freq_mhz=3562).dac_ch == 2
    assert dev.wiring.resolve(LineRef("qubit", "sideband"), freq_mhz=2010).dac_ch == 0


def test_calibration_populated():
    dev = Device.load(YAML)
    assert dev.calib.qubit.f_ge == pytest.approx(3562.6105332943075)
    assert dev.calib.qubit.pulses["pi_ge"].gain == 3289
    assert dev.calib.readout.frequency == pytest.approx(749.4328673317561)


def test_soccfg_snapshot_is_plain_dict():
    dev = Device.load(YAML)
    assert isinstance(dev.soccfg_snapshot, dict)
    assert dev.soccfg_snapshot["board"] == "ZCU216"
    # maxv is present so the pack can convert int-code gains -> fractions (Task 6)
    assert dev.soccfg_snapshot["gens"][0]["maxv"] == 32766


def test_topology_composed_and_namespace_bound():
    dev = Device.load(YAML)
    # 1 transmon + 1 manipulate + 7 storage + coupler + readout = 11 elements
    assert len(dev.elements) == 11
    # D21: dev.qubit is a TransmonOps bound to the *named* transmon element
    assert isinstance(dev.qubit, TransmonOps)
    assert dev.qubit.element is dev.elements["qubit"]
    assert dev.qubit.device is dev
    # elements whose ops land in a later slice are NOT bound as namespaces yet
    assert not hasattr(dev, "manipulate")
    assert not hasattr(dev, "coupler")


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
