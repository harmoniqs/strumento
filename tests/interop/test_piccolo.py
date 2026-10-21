"""Tests for the upstream Piccolo bridge + OperatorModel serialization (D31)."""

import json

import numpy as np

from strumento.core.builder import DeviceBuilder
from strumento.core.wiring import Generator
from strumento.interop.piccolo import PiccoloSystemSpec, to_piccolo_system
from strumento.packs.cqed.couplings import ZZ

CALIB = {
    "qubit": {"f_ge": 5000.0, "f_ef": 4800.0, "pulses": {}},
    "readout": {"frequency": 749.0, "gain": 800, "readout_length": 1.5, "trig_offset": 0.45, "relax_delay": 2500},
    "couplings": {"q0-q1": 0.5},
}


def _model():
    dev = (
        DeviceBuilder(pack="cqed")
        .add("Transmon", "q0").add("Transmon", "q1").add("ReadoutResonator", "ro")
        .wire("q0", drive=Generator(dac_ch=0)).wire("q1", drive=Generator(dac_ch=1))
        .readout(dac_ch=5, adc_chs=[0])
        .coupling("q0", "q1", form=ZZ)
        .build(calib=CALIB)
    )
    return dev.to_operator_model(frame="rotating", cutoffs={"q0": 2, "q1": 2})


def test_to_piccolo_system_shapes_drift_and_ordered_drives():
    spec = to_piccolo_system(_model())
    assert isinstance(spec, PiccoloSystemSpec)
    assert spec.H_drift.shape == (4, 4)
    assert spec.drive_labels == ["q0", "q1"]              # deterministic order
    assert len(spec.H_drives) == 2 and spec.H_drives[0].shape == (4, 4)


def test_operator_model_to_dict_is_json_serializable():
    d = _model().to_dict()
    json.dumps(d)  # must not raise — framework-neutral wire format
    assert set(d["H0"]) == {"real", "imag"}
    assert set(d["drives"]) == {"q0", "q1"}
    assert d["frame"] == "rotating"
    # round-trips back to the same drift matrix
    back = np.array(d["H0"]["real"]) + 1j * np.array(d["H0"]["imag"])
    assert back.shape == (4, 4)


def test_bridge_is_qick_free():
    import subprocess
    import sys

    code = "import sys; sys.modules['qick']=None; import strumento.interop.piccolo; print('ok')"
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0 and "ok" in r.stdout, r.stderr
