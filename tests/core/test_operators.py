"""Tests for the Hamiltonian export seam (D31)."""

import numpy as np
import pytest

from strumento.core.builder import DeviceBuilder
from strumento.core.operators import HilbertSpace, OperatorModel, annihilation
from strumento.core.wiring import Generator
from strumento.packs.cqed.couplings import ZZ

CALIB = {
    "qubit": {"f_ge": 5000.0, "f_ef": 4800.0, "pulses": {}},
    "readout": {"frequency": 749.0, "gain": 800, "readout_length": 1.5, "trig_offset": 0.45, "relax_delay": 2500},
    "couplings": {"q0-q1": 0.5},
}


def test_annihilation_matrix_elements():
    a = annihilation(3)
    assert a.shape == (3, 3)
    assert a[0, 1] == pytest.approx(1.0)
    assert a[1, 2] == pytest.approx(np.sqrt(2))


def test_hilbertspace_embedding_and_labels():
    sp = HilbertSpace(["q", "m"], [2, 3])
    assert sp.dim == 6
    assert np.allclose(np.diag(sp.ops("q").n).real, [0, 0, 0, 1, 1, 1])  # n_q ⊗ I_m
    assert len(sp.labels()) == 6 and sp.labels()[0] == (0, 0)


def _two_transmon_zz_device():
    return (
        DeviceBuilder(pack="cqed")
        .add("Transmon", "q0")
        .add("Transmon", "q1")
        .add("ReadoutResonator", "ro")
        .wire("q0", drive=Generator(dac_ch=0))
        .wire("q1", drive=Generator(dac_ch=1))
        .readout(dac_ch=5, adc_chs=[0])
        .coupling("q0", "q1", form=ZZ)
        .build(calib=CALIB)
    )


def test_to_operator_model_shape_hermiticity_and_drives():
    dev = _two_transmon_zz_device()
    model = dev.to_operator_model(frame="rotating", cutoffs={"q0": 2, "q1": 2})
    assert isinstance(model, OperatorModel)
    assert model.H0.shape == (4, 4)                       # 2 qubits × 2 levels
    assert np.allclose(model.H0, model.H0.conj().T)       # hermitian
    # ReadoutResonator has no Hamiltonian template → excluded from the drift
    assert set(model.dims) == {"q0", "q1"}
    # each driven line becomes a drive operator
    assert set(model.drives) == {"q0", "q1"}
    assert model.drive_map["q0"] == "q0:x"
    # the ZZ coupling (strength 0.5 from calib) is present: diag(1,-1,-1,1)·2π·0.5
    assert not np.allclose(model.H0, 0)
    assert np.allclose(np.diag(model.H0).real, 2 * np.pi * 0.5 * np.array([1, -1, -1, 1]))


def test_lab_frame_adds_the_linear_term():
    dev = _two_transmon_zz_device()
    rot = dev.to_operator_model(frame="rotating", cutoffs={"q0": 2, "q1": 2})
    lab = dev.to_operator_model(frame="lab", cutoffs={"q0": 2, "q1": 2})
    # lab frame keeps ω·n; rotating drops it → they differ
    assert not np.allclose(rot.H0, lab.H0)


def test_include_extracts_a_subgraph():
    dev = _two_transmon_zz_device()
    model = dev.to_operator_model(frame="rotating", cutoffs={"q0": 3}, include=["q0"])
    assert model.H0.shape == (3, 3)          # just q0's 3-level space
    assert set(model.dims) == {"q0"}
    assert set(model.drives) == {"q0"}


def test_missing_cutoff_raises():
    dev = _two_transmon_zz_device()
    with pytest.raises(ValueError, match="cutoffs missing"):
        dev.to_operator_model(frame="rotating", cutoffs={"q0": 2})  # q1 missing
