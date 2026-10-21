"""Tests for TransmonOps pulse factories + per-instance binding (D21 namespaces, D23 gains)."""

import pytest

from strumento.core.pulses import Gaussian
from strumento.core.wiring import LineRef
from strumento.packs.cqed.calibration import gain_int_to_frac

MAXV = 32766  # from the committed soccfg snapshot


def test_ge_pi_from_calibration(dev):
    p = dev.qubit.ge_pi()
    assert p.line == LineRef("qubit", "drive")
    assert p.freq_mhz == pytest.approx(dev.calib.qubit.f_ge)
    assert p.envelope == Gaussian(sigma_us=dev.calib.qubit.pulses["pi_ge"].sigma)
    # D23: gain is a v2 FRACTION, not the int code 8192
    assert p.gain == pytest.approx(gain_int_to_frac(8192, MAXV))
    assert p.gain == pytest.approx(8192 / MAXV)
    assert p.gain != 8192


def test_ge_hpi_ef_pi_and_ef_hpi_use_right_freq_and_gain(dev):
    hpi = dev.qubit.ge_hpi()
    assert hpi.freq_mhz == pytest.approx(dev.calib.qubit.f_ge)
    assert hpi.gain == pytest.approx(4096 / MAXV)

    ef = dev.qubit.ef_pi()
    assert ef.freq_mhz == pytest.approx(dev.calib.qubit.f_ef)  # ef transition, not ge
    assert ef.gain == pytest.approx(7600 / MAXV)

    # ef_hpi mirrors ge_hpi exactly, but on the ef transition (issue #89): same gain-units
    # conversion + safety validation, reading the "hpi_ef" calib key, frequency from f_ef.
    ef_hpi = dev.qubit.ef_hpi()
    assert ef_hpi.freq_mhz == pytest.approx(dev.calib.qubit.f_ef)  # ef transition, like ef_pi
    assert ef_hpi.gain == pytest.approx(3800 / MAXV)
    assert ef_hpi.envelope == Gaussian(sigma_us=dev.calib.qubit.pulses["hpi_ef"].sigma)


def test_ef_hpi_missing_hpi_ef_calibration_raises():
    """Issue #89 AC2: a device whose calibration omits ``hpi_ef`` gets the same bare
    ``KeyError`` every other factory already raises for a missing pulse key (mirrors
    ``dev.coupler.swap("cavity-S9")`` in test_sideband_mode_coupler_ops.py). Key Decision:
    no new error-handling path -- ef_hpi reuses TransmonOps._pulse's existing ``qc.pulses[key]``
    lookup, same as ge_hpi/ef_pi, so this is inherited behavior, not new handling."""
    from strumento.core.builder import DeviceBuilder
    from strumento.core.wiring import Generator

    b = DeviceBuilder(pack="cqed")
    b.add("Transmon", "qubit")
    b.add("ReadoutResonator", "readout")
    b.wire("qubit", drive=Generator(dac_ch=2, nyquist=1))
    b.readout(dac_ch=5, adc_chs=[0])
    calib = {
        "qubit": {"f_ge": 4000.0, "f_ef": 3820.0, "pulses": {
            "pi_ge": {"gain": 8192, "sigma": 0.03}}},  # deliberately no "hpi_ef" key
        "readout": {"frequency": 750.0, "gain": 1000, "readout_length": 2.0,
                    "trig_offset": 0.4, "relax_delay": 500},
    }
    dev_without_hpi_ef = b.build(calib=calib)
    with pytest.raises(KeyError):
        dev_without_hpi_ef.qubit.ef_hpi()


def test_overrides(dev):
    # a raw fraction passed as gain is used as-is (override convention, D23)
    p = dev.qubit.ge_pi(gain=0.5, freq=4000.0)
    assert p.gain == 0.5
    assert p.freq_mhz == 4000.0
    # phase override -> phase_deg
    assert dev.qubit.ge_pi(phase=90).phase_deg == 90


def test_binding_is_instance_scoped(dev):
    # dev.qubit ops are bound to *this* named element (D21)
    assert dev.qubit.element is dev.elements["qubit"]
    assert dev.qubit.element.name == "qubit"


def test_gain_sweep_maps_int_codes_to_fraction_keeps_lab_axis(dev):
    # D23: a gain Sweep is lab-native int codes -> fraction sweep for the IR, lab-native xpts kept.
    # Sweep endpoints must stay within the qubit_drive safety limit
    # (0.5 fraction = 16383 int code) -- 16000 is safely under it, unlike the old 30000 this
    # test used before sweep-endpoint validation was wired in (that value now raises, covered
    # by tests/packs/cqed/test_safety.py::test_sweep_gain_exceeding_limit_raises_at_factory).
    from strumento.core.sweeps import Sweep

    # the legacy per-name form on purpose: it is the form that pins the lab axis it is mapped
    # from (bound sweeps carry the inverse map instead, D36)
    with pytest.warns(DeprecationWarning, match="legacy per-name Sweep"):
        sweep = Sweep("amp", 0, 16000, 51)
    p = dev.qubit.ge_pi(gain=sweep)
    assert isinstance(p.gain, Sweep)
    assert p.gain.start == pytest.approx(0.0)
    assert p.gain.stop == pytest.approx(16000 / MAXV)  # converted to a v2 fraction
    assert p.gain.points == 51
    assert p.gain.xpts[0] == pytest.approx(0) and p.gain.xpts[-1] == pytest.approx(16000)


# -- DRAG: the factory consumes calib.qubit.drag_alpha -------------------------------------------


def test_ge_pi_stays_gaussian_when_drag_alpha_unset(dev):
    """Backward-compat: with drag_alpha=None (the shipped default) ge_pi is a plain Gaussian.

    This is the guard that DRAG is opt-in — nothing moves until a device calibrates the coefficient.
    """
    assert dev.calib.qubit.drag_alpha is None
    assert dev.qubit.ge_pi().envelope == Gaussian(sigma_us=dev.calib.qubit.pulses["pi_ge"].sigma)
    assert dev.qubit.ge_hpi().envelope == Gaussian(sigma_us=dev.calib.qubit.pulses["hpi_ge"].sigma)


def test_ge_pulses_emit_drag_arb_when_drag_alpha_set(dev):
    """With drag_alpha set, ge_pi/ge_hpi emit the shared DRAG Arb (I=Gaussian, Q=α·dGaussian/dt)."""
    from strumento.core.pulses import Arb
    from strumento.packs.cqed.library._drag import drag_arb

    dev.calib.qubit.drag_alpha = 0.5
    sigma = dev.calib.qubit.pulses["pi_ge"].sigma
    dac_rate = dev.qubit._dac_rate(dev.calib.qubit.f_ge)

    p = dev.qubit.ge_pi()
    assert isinstance(p.envelope, Arb)
    # The shipped shape IS the shared-builder shape at this α, so it equals what the calibration
    # measured: the factory and DRAGCalibration build the pulse with the same function.
    assert p.envelope == drag_arb(sigma, 0.5, dac_rate=dac_rate)
    # Q channel carries the derivative (non-zero), I channel is the peak-normalized Gaussian.
    assert p.envelope.qdata is not None and any(abs(q) > 1e-9 for q in p.envelope.qdata)
    assert max(p.envelope.idata) == pytest.approx(1.0)


def test_ge_drag_arb_matches_calibration_envelope(dev):
    """The factory's DRAG pulse equals DRAGCalibration's at the same α — same builder, same DAC rate.

    Not just the normalized profile at matched sample counts: the calibration used to pin 64 samples
    (10 ns on a 6.4 GS/s DAC vs the 4σ gate), which passed a shape-only check while playing a 14x
    shorter pulse.
    """
    from strumento.packs.cqed.experiments.drag_calibration import _drag_envelope

    dev.calib.qubit.drag_alpha = 0.7
    sigma = dev.calib.qubit.pulses["hpi_ge"].sigma
    dac_rate = dev.qubit._dac_rate(dev.calib.qubit.f_ge)
    assert dac_rate is not None  # the fixture carries a snapshot — otherwise this test is vacuous
    env, gain_scale = _drag_envelope(sigma, 0.7, dac_rate=dac_rate)
    assert gain_scale == 1.0
    assert env == dev.qubit.ge_hpi().envelope


def test_ge_drag_duration_matches_the_gaussian_it_replaces(dev):
    """The DRAG Arb spans the same 4σ window as the Gaussian — turning DRAG on shifts no timing."""
    from strumento.core.compiler import _GAUSS_LENGTH_SIGMAS

    dev.calib.qubit.drag_alpha = 0.5
    sigma = dev.calib.qubit.pulses["pi_ge"].sigma
    dac_rate = dev.qubit._dac_rate(dev.calib.qubit.f_ge)
    n = len(dev.qubit.ge_pi().envelope.idata)
    # n samples at dac_rate must reproduce the Gaussian's total length (4σ) within a sample.
    expected_us = _GAUSS_LENGTH_SIGMAS * sigma
    assert n / dac_rate == pytest.approx(expected_us, abs=1.0 / dac_rate)


def test_ef_pulses_ignore_drag_alpha(dev):
    """ef pulses stay Gaussian even with drag_alpha set — DRAG applies to the ge pulses only."""
    dev.calib.qubit.drag_alpha = 0.5
    assert dev.qubit.ef_pi().envelope == Gaussian(sigma_us=dev.calib.qubit.pulses["pi_ef"].sigma)
    assert dev.qubit.ef_hpi().envelope == Gaussian(sigma_us=dev.calib.qubit.pulses["hpi_ef"].sigma)


# --- Pulse overrides must honour freq/gain like the calibrated path ----------------------------
#
# Regression: the override branch returned the stored Pulse untouched apart from
# phase. ErrorAmplification sweeps by calling ``pulse(freq=Sweep)`` / ``pulse(gain=Sweep)`` — with an
# override that swept NOTHING (a flat, "perfectly robust" fringe that reads as a robust gate). AllXY's identity gate is ``ge_pi(gain=0.0)`` — with an override it played a full π.


def _arb_override(dev):
    from strumento.core.pulses import Arb, Pulse
    from strumento.core.wiring import LineRef

    return Pulse(line=LineRef("qubit", "drive"), freq_mhz=dev.calib.qubit.f_ge, gain=0.2,
                 phase_deg=0.0, envelope=Arb(idata=tuple([0.0, 0.5, 1.0, 0.5, 0.0] * 4)),
                 label="override_pi_ge")


def _ops_with(dev, key="pi_ge"):
    from strumento.packs.cqed.library.transmon import TransmonOps

    return TransmonOps(dev, dev.qubit.element, overrides={key: _arb_override(dev)})


def test_override_honours_scalar_gain_as_raw_fraction(dev):
    ops = _ops_with(dev)
    assert ops.ge_pi().gain == pytest.approx(0.2)            # default: the override's own gain
    assert ops.ge_pi(gain=0.0).gain == 0.0                    # AllXY identity: a zero-gain gate
    assert ops.ge_pi(gain=0.1).gain == pytest.approx(0.1)
    assert ops.ge_pi(gain=0.0).envelope == _arb_override(dev).envelope  # same slot length


def test_override_honours_freq_scalar_and_sweep(dev):
    from strumento.core.sweeps import Sweep

    ops = _ops_with(dev)
    assert ops.ge_pi(freq=4001.0).freq_mhz == 4001.0
    sw = Sweep(start=3999.0, stop=4001.0, on="err_amp")
    assert ops.ge_pi(freq=sw).freq_mhz is sw                   # ErrorAmplification frequency sweep


def test_override_gain_sweep_maps_int_codes_like_the_calibrated_path(dev):
    from strumento.core.sweeps import Sweep

    ops = _ops_with(dev)
    sw = Sweep(start=0, stop=16000, on="amp")
    p = ops.ge_pi(gain=sw)
    assert isinstance(p.gain, Sweep)
    ref = dev.qubit.ge_pi(gain=sw).gain                        # calibrated-path mapping
    assert p.gain.start == pytest.approx(ref.start) and p.gain.stop == pytest.approx(ref.stop)


def test_override_gain_faces_the_same_safety_limits(dev):
    from strumento.core.sweeps import Sweep
    from strumento.packs.cqed.safety import GainOutOfRange

    ops = _ops_with(dev)
    with pytest.raises(GainOutOfRange):
        ops.ge_pi(gain=0.9)
    with pytest.raises(GainOutOfRange):
        ops.ge_pi(gain=Sweep(start=0, stop=30000, on="amp"))


def test_override_refuses_a_length_it_cannot_honour(dev):
    with pytest.raises(ValueError, match="length"):
        _ops_with(dev).ge_pi(length=0.2)


def test_override_phase_still_applies_with_other_kwargs(dev):
    p = _ops_with(dev).ge_pi(phase=90.0, gain=0.0)
    assert p.phase_deg == 90.0 and p.gain == 0.0


# --- An override's own gain faces the drive limit too (#187) ----------------------------------
# With gain=None the override path used to return the override pulse's own gain unchecked, while
# the calibrated path checks even its calibrated value and an explicit gain= on an override is
# checked. An override is a fraction already (the Pulse IR's unit), so it is checked as one.

#: Every pulse key an override can replace -> (factory method, calibrated frequency field).
_OVERRIDE_KEYS = {
    "pi_ge": ("ge_pi", "f_ge"),
    "hpi_ge": ("ge_hpi", "f_ge"),
    "pi_ef": ("ef_pi", "f_ef"),
    "hpi_ef": ("ef_hpi", "f_ef"),
}


def _override_at(dev, key, gain):
    from strumento.core.pulses import Arb, Pulse

    freq = getattr(dev.calib.qubit, _OVERRIDE_KEYS[key][1])
    return Pulse(line=LineRef("qubit", "drive"), freq_mhz=freq, gain=gain, phase_deg=0.0,
                 envelope=Arb(idata=(0.0, 0.5, 1.0, 0.5, 0.0)), label=f"override_{key}")


def _play(dev, key, gain, **kw):
    from strumento.packs.cqed.library.transmon import TransmonOps

    ops = TransmonOps(dev, dev.qubit.element, overrides={key: _override_at(dev, key, gain)})
    return getattr(ops, _OVERRIDE_KEYS[key][0])(**kw)


@pytest.mark.parametrize("key", sorted(_OVERRIDE_KEYS))
@pytest.mark.parametrize("gain", [0.95, -0.95, 0.5000001])
def test_override_above_the_drive_limit_is_refused_with_gain_none(dev, key, gain):
    from strumento.packs.cqed.safety import MAX_QUBIT_DRIVE_GAIN, GainOutOfRange

    assert dev.calib.qubit.max_drive_gain is None and abs(gain) > MAX_QUBIT_DRIVE_GAIN
    with pytest.raises(GainOutOfRange, match=key):
        _play(dev, key, gain)
    with pytest.raises(GainOutOfRange, match=key):
        _play(dev, key, gain, phase=90.0, freq=4001.0)  # other kwargs don't bypass it


@pytest.mark.parametrize("key", sorted(_OVERRIDE_KEYS))
def test_override_gain_sweep_beyond_the_limit_is_refused_with_gain_none(dev, key):
    from strumento.core.sweeps import Sweep
    from strumento.packs.cqed.safety import GainOutOfRange

    # An override's Sweep gain is in the IR's unit (fractions), so each endpoint is a fraction.
    for sw in (Sweep(start=0.0, stop=0.9, on="amp"), Sweep(start=-0.7, stop=0.1, on="amp")):
        with pytest.raises(GainOutOfRange):
            _play(dev, key, sw)


@pytest.mark.parametrize("key", sorted(_OVERRIDE_KEYS))
def test_override_honours_the_per_element_drive_limit(dev, key):
    from strumento.packs.cqed.safety import MAX_QUBIT_DRIVE_GAIN, GainOutOfRange

    dev.calib.qubit.max_drive_gain = 0.15
    assert 0.15 < 0.2 < MAX_QUBIT_DRIVE_GAIN  # inside the pack limit, over the element's
    with pytest.raises(GainOutOfRange, match=key):
        _play(dev, key, 0.2)
    assert _play(dev, key, 0.15).gain == 0.15  # at the element's limit is allowed


@pytest.mark.parametrize("key", sorted(_OVERRIDE_KEYS))
@pytest.mark.parametrize("gain", [0.0, 0.2, -0.3, 0.5, "sweep"])
@pytest.mark.parametrize("kw", [{}, {"phase": 90.0}, {"freq": 4001.0}])
def test_override_within_the_limit_plays_exactly_as_before(dev, key, gain, kw):
    """Characterized on the code before #187: the override, with the caller's freq/phase only."""
    from dataclasses import replace

    from strumento.core.sweeps import Sweep

    g = Sweep(start=0.0, stop=0.4, on="amp") if gain == "sweep" else gain
    ov = _override_at(dev, key, g)
    expected = replace(ov, freq_mhz=kw.get("freq", ov.freq_mhz), phase_deg=kw.get("phase", 0.0))
    got = _play(dev, key, g, **kw)
    assert got == expected
    assert got.gain is ov.gain  # the very same value, not a re-derived one


# --- An override plays at the cursor, like a calibrated gate (#210) ---------------------------
# `from_solution` emits each pulse with `start_us=0.0` — an offset *inside* its parallel segment.
# Installed as a gate override, that offset used to survive `_pulse`, so the compiler placed the
# gate absolutely (no cursor advance): both gates of a pair and the readout stacked at the pair's
# start. The override boundary drops the offset; the compiled-timing proof is in
# tests/packs/cqed/test_allxy_overrides.py.


@pytest.mark.parametrize("key", sorted(_OVERRIDE_KEYS))
@pytest.mark.parametrize("start_us", [0.0, 0.25])
@pytest.mark.parametrize("kw", [{}, {"phase": 90.0}, {"gain": 0.1}, {"freq": 4001.0}])
def test_an_override_drops_its_start_offset_and_plays_at_the_cursor(dev, key, start_us, kw):
    from dataclasses import replace

    from strumento.packs.cqed.library.transmon import TransmonOps

    ov = replace(_override_at(dev, key, 0.2), start_us=start_us)
    ops = TransmonOps(dev, dev.qubit.element, overrides={key: ov})
    got = getattr(ops, _OVERRIDE_KEYS[key][0])(**kw)
    assert got.start_us is None                  # sequential: the compiler advances past it
    assert got.envelope is ov.envelope           # ... and it is still the override's waveform


def test_an_oc_import_installed_as_an_override_plays_at_the_cursor(dev):
    """The real producer: a ``from_solution`` pulse keeps 0.0 in-segment, loses it as a gate."""
    import numpy as np

    from strumento.core.pulses import from_solution
    from strumento.packs.cqed.calibration import PiPulseReference
    from strumento.packs.cqed.library.transmon import TransmonOps

    t_us = np.linspace(0.0, 0.1, 51)
    sol = {"times": t_us, "controls": {"qubit": 5.0 * np.sin(np.pi * t_us / 0.1)}}
    seq = from_solution(sol, drive_map={"qubit": (("qubit", "drive"), dev.calib.qubit.f_ge)},
                        gain_calib=PiPulseReference(pi_gain_frac=0.25, pi_rabi_mhz=10.0))
    (oc,) = seq.ops[0].pulses
    assert oc.start_us == 0.0                    # the import's in-segment semantics are unchanged
    assert TransmonOps(dev, dev.qubit.element, overrides={"pi_ge": oc}).ge_pi().start_us is None


def test_calibrated_gates_are_unchanged_by_the_override_placement_fix(dev):
    for method, _ in _OVERRIDE_KEYS.values():
        assert getattr(dev.qubit, method)().start_us is None
