"""Tests for the cqed element types + calibration schema (plan Task 4, D21/D23).

Values mirror the real Schuster multimode v1 config (``.refs-v1/.../hardware_config.yml``),
with the v1 one-element-list idiom flattened to scalars.
"""

import math

import pytest

from strumento.packs.cqed.calibration import Calibration, gain_int_to_frac
from strumento.packs.cqed.elements import Transmon, BosonicMode, Coupler, ReadoutResonator
from strumento.packs.cqed.library.transmon import TransmonOps
from strumento.core.wiring import LineRef


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
            "trig_offset": 450,
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


def test_gain_int_to_frac():
    # D23: v1 lab-native int codes convert to a v2 fraction at the pack boundary.
    assert gain_int_to_frac(3289, 32766) == pytest.approx(3289 / 32766)
    assert math.isclose(gain_int_to_frac(3289, 32766), 0.10038, abs_tol=1e-4)
    assert gain_int_to_frac(0, 32766) == 0.0
    assert gain_int_to_frac(32766, 32766) == pytest.approx(1.0)


def test_element_types_construct_and_expose_lines():
    t = Transmon(name="qubit", f_ge=3562.6, f_ef=3419.0, pulses={})
    assert LineRef("qubit", "drive") in t.lines()

    m = BosonicMode(name="manipulate", role="manipulate")
    assert LineRef("manipulate", "main") in m.lines()

    cpl = Coupler(name="coupler")
    assert set(cpl.lines()) == {LineRef("coupler", "low"), LineRef("coupler", "high")}

    ro = ReadoutResonator(name="readout")
    # readout is carried by ReadoutWiring, not a plain Line — it drives no LineRefs
    assert ro.lines() == []


def test_element_ops_cls_is_topology_derived():
    # D21: element TYPE -> ops class (instance name is irrelevant here).
    assert Transmon.ops_cls is TransmonOps
    # element types whose ops land in a later slice have no ops class bound yet
    assert BosonicMode.ops_cls is None
    assert Coupler.ops_cls is None
