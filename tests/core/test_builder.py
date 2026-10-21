"""Tests for the DeviceBuilder: config-as-code that produces the canonical config data (D30)."""

import pytest

from strumento.core.builder import DeviceBuilder
from strumento.core.device import Device
from strumento.core.wiring import Generator, LineRef
from strumento.packs.cqed.library.transmon import TransmonOps

CALIB = {
    "qubit": {"f_ge": 3562.6, "f_ef": 3419.0, "pulses": {"pi_ge": {"gain": 3289, "sigma": 0.035}}},
    "readout": {"frequency": 749.4, "gain": 800, "readout_length": 1.5, "trig_offset": 0.45, "relax_delay": 2500},
}


def _cqed_builder():
    return (
        DeviceBuilder(pack="cqed")
        .add("Transmon", "qubit")
        .add("Coupler", "coupler")
        .add("ReadoutResonator", "readout")
        .wire("qubit", drive=Generator(dac_ch=1, nyquist=1), sideband=Generator(dac_ch=5))
        .wire("coupler", low=Generator(dac_ch=4, band_hz=(0.0, 1.8e9)),
                          high=Generator(dac_ch=4, band_hz=(1.8e9, 1e12)))
        .readout(dac_ch=0, adc_chs=[0])
    )


def test_builder_produces_a_valid_device():
    dev = _cqed_builder().build(calib=CALIB)
    assert isinstance(dev, Device)
    assert dev.wiring.resolve(LineRef("qubit", "drive"), 3562).dac_ch == 1
    assert dev.wiring.resolve(LineRef("qubit", "sideband"), 2010).dac_ch == 5
    assert dev.calib.qubit.f_ge == 3562.6
    assert isinstance(dev.qubit, TransmonOps)  # D21 namespace bound, same as load
    assert set(dev.elements) == {"qubit", "coupler", "readout"}


def test_role_is_filled_from_the_wire_key():
    dev = _cqed_builder().build(calib=CALIB)
    assert {g.role for g in dev.wiring.lines["qubit"].generators} == {"drive", "sideband"}


def test_builder_matches_yaml_load(tmp_path):
    # Device ⇄ YAML ⇄ builder must yield the identical model.
    yaml_text = """
pack: cqed
wiring:
  lines:
    qubit:
      name: qubit
      generators:
        - {role: drive, dac_ch: 1, nyquist: 1}
        - {role: sideband, dac_ch: 5, nyquist: 1}
    coupler:
      name: coupler
      generators:
        - {role: low,  dac_ch: 4, band_hz: [0.0, 1.8e9]}
        - {role: high, dac_ch: 4, band_hz: [1.8e9, 1.0e12]}
  readout: {dac_ch: 0, adc_chs: [0]}
topology:
  - {type: Transmon, name: qubit}
  - {type: Coupler, name: coupler}
  - {type: ReadoutResonator, name: readout}
calibration:
  qubit: {f_ge: 3562.6, f_ef: 3419.0, pulses: {pi_ge: {gain: 3289, sigma: 0.035}}}
  readout: {frequency: 749.4, gain: 800, readout_length: 1.5, trig_offset: 0.45, relax_delay: 2500}
"""
    p = tmp_path / "device.yaml"
    p.write_text(yaml_text)

    built = _cqed_builder().build(calib=CALIB)
    loaded = Device.load(p)

    assert built.wiring == loaded.wiring        # Pydantic structural equality
    assert built.calib == loaded.calib
    assert sorted(built.elements) == sorted(loaded.elements)


def test_pack_agnostic_builds_a_tweezer_device():
    # the builder never names Transmon — it composes over the loaded pack's element types
    dev = (
        DeviceBuilder(pack="tweezer")
        .add("AODAxis", "aod_x", axis="x")
        .add("AtomArray", "array", n_traps=88)
        .wire("aod_x", mux=Generator(dac_ch=0))
        .readout(dac_ch=0, adc_chs=[0])
        .build(calib={"aod_x": {"center_mhz": 100, "span_mhz": 20},
                      "aod_y": {"center_mhz": 100, "span_mhz": 20}})
    )
    assert set(dev.elements) == {"aod_x", "array"}
    assert dev.wiring.resolve(LineRef("aod_x", "mux"), 100).dac_ch == 0


def test_calib_stays_data_and_build_validates_it():
    # calibration values are passed as data, never authored on the builder
    with pytest.raises(Exception):
        _cqed_builder().build(calib={"qubit": {"f_ge": 3562.6}})  # missing readout section


def test_missing_readout_raises():
    b = DeviceBuilder(pack="cqed").add("Transmon", "qubit").wire("qubit", drive=Generator(dac_ch=2))
    with pytest.raises(ValueError, match="readout"):
        b.build(calib=CALIB)
