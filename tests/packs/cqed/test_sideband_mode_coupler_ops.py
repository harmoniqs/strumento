"""Sideband / mode / coupler ops + alias binding (pure)."""

from __future__ import annotations

import pytest

from strumento.core.pulses import FlatTop, Gaussian, Seq
from strumento.core.wiring import LineRef
from strumento.packs.cqed.calibration import MultiphotonCal, SidebandPulseCal

MAXV = 32766


def test_sideband_alias_bound_from_instance_yaml(dev):
    # dev.sideband exists because the YAML declares the alias, not because the pack hardcodes it
    p = dev.sideband.f0g1()
    assert p.line == LineRef("qubit", "sideband")
    assert p.freq_mhz == pytest.approx(2100.0)
    assert p.gain == pytest.approx(12000 / MAXV)
    assert p.envelope == FlatTop(ramp_sigma_us=0.01, length_us=0.8)


def test_multiphoton_indexes_by_photon_number(dev):
    p1 = dev.sideband.multiphoton("f0-g1", "pi", photon=1)
    assert p1.freq_mhz == pytest.approx(2088.0)
    assert p1.gain == pytest.approx(12500 / MAXV)
    with pytest.raises(IndexError):
        dev.sideband.multiphoton("f0-g1", "hpi", photon=3)
    with pytest.raises(KeyError):
        dev.sideband.multiphoton("g0-e1")


def test_multiphoton_preserves_a_gaussian_calibration(dev):
    dev.calib.multiphoton["g0-e1"] = MultiphotonCal(
        pi=[SidebandPulseCal(frequency=3562.0, gain=3319, sigma=0.035, type="gauss")],
    )

    pulse = dev.sideband.multiphoton("g0-e1")

    assert pulse.envelope == Gaussian(sigma_us=0.035)


def test_multiphoton_rejects_a_zero_length_flat_top_calibration():
    with pytest.raises(ValueError, match="length > 0"):
        SidebandPulseCal(frequency=2100.0, gain=12000, length=0.0, ramp_sigma=0.01)


def test_flat_top_multiphoton_keeps_its_default_ramp_sigma():
    calibration = SidebandPulseCal(frequency=2100.0, gain=12000, length=0.8)

    assert calibration.ramp_sigma == pytest.approx(0.005)


def test_mode_ops_displace(dev):
    p = dev.cavity.displace()
    assert p.line == LineRef("cavity", "main")
    assert p.freq_mhz == pytest.approx(5200.0)
    assert p.gain == pytest.approx(6000 / MAXV)
    assert p.envelope == Gaussian(sigma_us=0.040)


def test_mode_ops_displace_alpha_converts_native_gain_code_to_a_v2_fraction(dev):
    pulse = dev.cavity.displace_alpha(1.0)

    assert pulse.gain == pytest.approx(3000 / MAXV)


@pytest.mark.qick
def test_gaussian_multiphoton_compiles_without_a_flat_top_length(dev):
    from strumento.core.program import StrumentoProgram  # needs qick; keep the module pure

    dev.calib.multiphoton["g0-e1"] = MultiphotonCal(
        pi=[SidebandPulseCal(frequency=3562.0, gain=3319, sigma=0.035, type="gauss")],
    )

    StrumentoProgram(dev, seq=Seq().play(dev.sideband.multiphoton("g0-e1")), reps=1)


@pytest.mark.qick
def test_displace_alpha_realizes_its_native_gain_code_on_tprocv2(dev):
    from strumento.core.program import StrumentoProgram  # needs qick; keep the module pure

    program = StrumentoProgram(dev, seq=Seq().play(dev.cavity.displace_alpha(1.0)), reps=1)
    waveform = next(iter(program.pulses.values())).waveforms[0]

    assert waveform.gain.start == 3000


def test_coupler_swap_band_gated(dev):
    p = dev.coupler.swap("cavity-S1")
    assert p.line == LineRef("coupler", "low")        # 800 MHz < 1800 MHz band edge
    assert p.freq_mhz == pytest.approx(800.0)
    assert p.envelope == FlatTop(ramp_sigma_us=0.01, length_us=2.4)
    hpi = dev.coupler.swap("cavity-S1", "hpi")
    assert hpi.envelope.length_us == pytest.approx(1.2)  # 50:50 beamsplitter = half length
    with pytest.raises(KeyError):
        dev.coupler.swap("cavity-S9")


def test_coupler_swap_frequency_sweep_within_one_band_is_fine(dev):
    from strumento.core.sweeps import Sweep
    p = dev.coupler.swap("cavity-S1", freq=Sweep(start=700.0, stop=900.0, on="f"))
    assert p.line == LineRef("coupler", "low")  # both endpoints < 1800 MHz


def test_coupler_swap_frequency_sweep_straddling_band_boundary_raises(dev):
    """Regression: _role_for used to resolve the generator from only the
    sweep's .start -- a scan crossing the 1.8 GHz low/high boundary silently compiled
    onto whichever generator .start picked, with no check that .stop was still in band."""
    from strumento.core.sweeps import Sweep
    with pytest.raises(ValueError, match="straddles a coupler band boundary"):
        dev.coupler.swap("cavity-S1", freq=Sweep(start=1700.0, stop=1900.0, on="f"))
