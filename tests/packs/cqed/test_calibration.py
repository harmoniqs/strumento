"""Tests for the cqed element types + calibration schema (D21 namespaces, D23 gain units).

Values are representative multimode-device calibration values used as test fixtures, written
as scalars (not one-element lists).
"""

import math

import pytest

from strumento.core.wiring import LineRef
from strumento.packs.cqed.calibration import Calibration, gain_int_to_frac
from strumento.packs.cqed.elements import BosonicMode, Coupler, ReadoutResonator, Transmon
from strumento.packs.cqed.library.transmon import TransmonOps


def _calib_dict():
    return {
        "qubit": {
            "f_ge": 3562.6105332943075,
            "f_ef": 3419.036012060942,
            "pulses": {
                "pi_ge": {"gain": 3289, "sigma": 0.035, "length": 0.0, "type": "gauss"},
                "hpi_ge": {"gain": 1629, "sigma": 0.035, "length": 0.0, "type": "gauss"},
            },
        },
        "readout": {
            "frequency": 749.4328673317561,
            "gain": 800,
            "readout_length": 1.5,
            "trig_offset": 0.45,
            "relax_delay": 2500,
        },
    }


def test_calibration_roundtrips_minimal_dict():
    c = Calibration(**_calib_dict())
    assert c.qubit.f_ge == pytest.approx(3562.6105332943075)
    pi = c.qubit.pulses["pi_ge"]
    assert (pi.gain, pi.sigma, pi.length, pi.type) == (3289, 0.035, 0.0, "gauss")
    assert c.readout.frequency == pytest.approx(749.4328673317561)
    assert c.readout.readout_length == 1.5
    # round-trips back to a dict
    assert c.model_dump()["qubit"]["pulses"]["pi_ge"]["gain"] == 3289


def test_readout_blob_centres_and_parity_sign_roundtrip():
    # Wigner-parity normalization plan: ReadoutCal g/e centres + ModeCal parity_sign are additive,
    # strict-schema fields that round-trip through the calibration model.
    d = _calib_dict()
    d["readout"]["g_center_iq"] = [1.0, 2.0]
    d["readout"]["e_center_iq"] = [-3.0, 4.0]
    d["modes"] = {"cavity": {"revival_time": [1.6], "alpha_scale": 0.9, "parity_sign": -1.0}}
    c = Calibration(**d)
    assert c.readout.g_center_iq == (1.0, 2.0)
    assert c.readout.e_center_iq == (-3.0, 4.0)
    assert c.modes["cavity"].alpha_scale == pytest.approx(0.9)
    assert c.modes["cavity"].parity_sign == -1.0
    dumped = c.model_dump()
    assert tuple(dumped["readout"]["g_center_iq"]) == (1.0, 2.0)
    assert dumped["modes"]["cavity"]["parity_sign"] == -1.0


def test_readout_blob_centres_default_none_and_parity_sign_plus_one():
    c = Calibration(**_calib_dict())
    assert c.readout.g_center_iq is None and c.readout.e_center_iq is None
    # a mode with no parity_sign set defaults to the +1 convention
    c2 = Calibration(**{**_calib_dict(), "modes": {"cavity": {"revival_time": [1.6]}}})
    assert c2.modes["cavity"].parity_sign == 1.0
    assert c2.modes["cavity"].alpha_scale == 1.0


def test_readout_f_blob_centre_roundtrips():
    d = _calib_dict()
    d["readout"]["f_center_iq"] = [1.25, -3.5]

    calibration = Calibration(**d)

    assert calibration.readout.f_center_iq == (1.25, -3.5)
    assert tuple(calibration.model_dump()["readout"]["f_center_iq"]) == (1.25, -3.5)


def test_sideband_transfer_metrics_roundtrip():
    d = _calib_dict()
    d["multiphoton"] = {
        "f0-g1": {
            "pi": [{
                "frequency": 2006.54,
                "gain": 4000,
                "length": 0.53,
                "ramp_sigma": 0.005,
                "type": "flat_top",
                "transfer_g_population": 0.93,
                "transfer_initial_mapped_e_population": 0.95,
                "transfer_residual_mapped_e_population": 0.07,
            }],
        },
    }

    calibration = Calibration(**d)
    pulse = calibration.multiphoton["f0-g1"].pi[0]

    assert pulse.transfer_g_population == pytest.approx(0.93)
    assert pulse.transfer_initial_mapped_e_population == pytest.approx(0.95)
    assert pulse.transfer_residual_mapped_e_population == pytest.approx(0.07)


def test_gain_int_to_frac():
    # D23: lab-native int codes convert to a v2 fraction at the pack boundary.
    assert gain_int_to_frac(3289, 32766) == pytest.approx(3289 / 32766)
    assert math.isclose(gain_int_to_frac(3289, 32766), 0.10038, abs_tol=1e-4)
    assert gain_int_to_frac(0, 32766) == 0.0
    assert gain_int_to_frac(32766, 32766) == pytest.approx(1.0)


def test_element_types_construct_and_expose_lines():
    t = Transmon(name="qubit", f_ge=3562.6, f_ef=3419.0, pulses={})
    assert LineRef("qubit", "drive") in t.lines()

    m = BosonicMode(name="cavity", role="cavity")
    assert LineRef("cavity", "main") in m.lines()

    cpl = Coupler(name="coupler")
    assert set(cpl.lines()) == {LineRef("coupler", "low"), LineRef("coupler", "high")}

    ro = ReadoutResonator(name="readout")
    # readout is carried by ReadoutWiring, not a plain Line — it drives no LineRefs
    assert ro.lines() == []


def test_element_ops_cls_is_topology_derived():
    # D21: element TYPE -> ops class (instance name is irrelevant here).
    assert Transmon.ops_cls is TransmonOps
    # element types whose ops land in a later slice have no ops class bound yet
    from strumento.packs.cqed.library.coupler import CouplerOps
    from strumento.packs.cqed.library.mode import ModeOps

    assert BosonicMode.ops_cls is ModeOps
    assert Coupler.ops_cls is CouplerOps
