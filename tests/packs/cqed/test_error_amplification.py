"""Error-amplification experiment (issue #82).

Repeats a calibrated pulse ``n_pulses`` times, sweeping its frequency or gain across a narrow
window, then fits the amplified deviation (``fit_parity``'s cosine model) to recover the
zero-error parameter value — the mechanism behind sub-linewidth pulse fine-tuning (a small
per-pulse error accumulates N-fold).

Unrolled by construction (the same call ``active_reset`` made for its own multi-round composite,
see ``library/composite.py``'s ``hardware_loop`` docstring): the built sequence is ``n_pulses``
``Play`` ops followed by exactly ONE ``Measure``, so ``reads_per_shot`` is trivially correct
regardless of ``n_pulses`` and there is no question about a ``Sweep``-carrying pulse inside a
``Repeat`` body.

Pure at import: no ``qick`` (the Runner does the qick-facing acquire, lazily).
"""

from __future__ import annotations

import numpy as np
import pytest

from strumento.core.device import Device
from strumento.core.fitters import FitResult
from strumento.core.pulses import Measure, Pulse
from strumento.core.result import MeasurementResult
from strumento.core.sweeps import Loop, Sweep
from strumento.packs.cqed.experiments.error_amplification import ErrorAmplification
from tests._paths import DEVICES

LOOPBACK_YAML = DEVICES / "loopback_demo" / "device.yaml"

#: loopback_demo has no dedicated readout line (single-generator instance) — probe on the
#: qubit line instead, same as resonator_spectroscopy's own test override (issue #102).
LOOPBACK_PROBE_LINE = ("qubit", "drive")


@pytest.fixture
def dev(loopback_dev):
    return loopback_dev


def _amplified_pulses(seq):
    """The pulses ErrorAmplification is amplifying — excludes the readout probe pulse that
    ``dev.readout.dispersive_measure()`` now prepends to the Measure (issue #102): that probe
    is a `Pulse` too, but it is not one of the ``n_pulses`` played copies these tests pin."""
    return [op for op in seq.ops if isinstance(op, Pulse) and op.label != "ro_probe"]


# -- sequence shape (AC1, AC4) ----------------------------------------------------------


def test_sequence_plays_the_pulse_n_pulses_times_then_one_measure(dev):
    exp = ErrorAmplification(dev, n_pulses=5, line=LOOPBACK_PROBE_LINE)
    seq, sweeps = exp.sequence()
    kinds = [type(op).__name__ for op in seq.ops]
    # issue #102: a readout probe now precedes the Measure (dev.readout.dispersive_measure()).
    assert kinds == ["Pulse"] * 5 + ["Pulse", "Measure"]
    assert isinstance(seq.ops[-1], Measure)
    assert sweeps == ()  # the sweep travels in each pulse's freq=/gain=; the compiler discovers it


def test_every_played_pulse_carries_the_same_swept_window(dev):
    window = Sweep(start=3999.0, stop=4001.0, on="ax")
    exp = ErrorAmplification(dev, n_pulses=3, window=window, line=LOOPBACK_PROBE_LINE)
    seq, _ = exp.sequence()
    pulses = _amplified_pulses(seq)
    assert len(pulses) == 3
    for p in pulses:
        assert p.freq_mhz is window


def test_default_pulse_is_the_calibrated_ge_pi(dev):
    exp = ErrorAmplification(dev, n_pulses=2, line=LOOPBACK_PROBE_LINE)
    seq, _ = exp.sequence()
    pulses = _amplified_pulses(seq)
    assert all(p.label == "pi_ge" for p in pulses)


def test_n_pulses_defaults_to_seven(dev):
    exp = ErrorAmplification(dev)
    assert exp.n_pulses == 7


def test_zero_or_negative_n_pulses_is_rejected(dev):
    with pytest.raises(ValueError, match="n_pulses"):
        ErrorAmplification(dev, n_pulses=0)


def test_axes_declares_a_loop_when_the_window_is_bound(dev):
    window = Sweep(start=3999.0, stop=4001.0, on="detune")
    exp = ErrorAmplification(dev, window=window, points=17)
    assert exp.axes() == (Loop("detune", 17),)


# -- parameter -> factory kwarg mapping (frequency -> freq=, gain -> gain=) --------------


def test_parameter_gain_sweeps_the_gain_kwarg_not_frequency(dev):
    window = Sweep(start=8000.0, stop=8400.0, on="ax")
    exp = ErrorAmplification(dev, parameter="gain", window=window, n_pulses=2,
                              line=LOOPBACK_PROBE_LINE)
    seq, _ = exp.sequence()
    pulses = _amplified_pulses(seq)
    assert len(pulses) == 2
    for p in pulses:
        assert not isinstance(p.freq_mhz, Sweep)  # frequency stays at the calibrated scalar
        # the factory maps the gain sweep's lab-native int-code endpoints -> fraction (D23);
        # the inverse it attaches recovers exactly the window we passed in.
        assert isinstance(p.gain, Sweep)
        assert p.gain._lab_inverse(p.gain.start) == pytest.approx(window.start)
        assert p.gain._lab_inverse(p.gain.stop) == pytest.approx(window.stop)


def test_an_unknown_parameter_is_rejected(dev):
    with pytest.raises(ValueError, match="parameter"):
        ErrorAmplification(dev, parameter="phase")


# -- addressed by a bound factory method, not a string name (recommendation #3) ---------


def test_pulse_accepts_any_bound_factory_method_ef_pi(dev):
    """Genericity: the SAME experiment class amplifies an ef pulse with no special-casing."""
    exp = ErrorAmplification(dev, pulse=dev.qubit.ef_pi, n_pulses=3,
                              window=Sweep(start=3819.0, stop=3821.0, on="ax"),
                              line=LOOPBACK_PROBE_LINE)
    seq, _ = exp.sequence()
    pulses = _amplified_pulses(seq)
    assert len(pulses) == 3
    assert all(p.label == "pi_ef" for p in pulses)


def test_pulse_accepts_the_sideband_f0g1_factory_too(dev):
    """The third pulse family the Key Decision names explicitly (ge, ef, f0g1) -- routed
    through an instance-declared alias (``dev.sideband``), needs multimode_demo's multiphoton
    calibration (absent from loopback_demo)."""
    multimode = Device.load(DEVICES / "multimode_demo" / "device.yaml")
    exp = ErrorAmplification(multimode, pulse=multimode.sideband.f0g1, n_pulses=2,
                              window=Sweep(start=2099.0, stop=2101.0, on="ax"))
    seq, _ = exp.sequence()
    pulses = _amplified_pulses(seq)
    assert len(pulses) == 2
    assert all(p.label == "f0-g1_pi0" for p in pulses)


def test_gain_window_exceeding_the_safety_ceiling_is_rejected(dev):
    """Constraint: 'safety ceilings apply to every repeated pulse'. Satisfied BY CONSTRUCTION
    -- the same validated factory call played N times -- not by new safety code in
    ErrorAmplification itself; this pins that the propagation actually happens."""
    from strumento.packs.cqed.safety import GainOutOfRange

    # ~0.916 fraction on this instance's maxv -- the same historical over-ceiling AmplitudeRabi
    # used to default to (audit-fixed there; deliberately re-created here as the negative case).
    unsafe_window = Sweep(start=0.0, stop=30000.0, on="ax")
    exp = ErrorAmplification(dev, parameter="gain", window=unsafe_window, n_pulses=2)
    with pytest.raises(GainOutOfRange):
        exp.sequence()


# -- analyzer math (AC2): the zero-crossing recovery, on hand-built data -----------------


def _cosine_curve(x, *, freq, phase, offset=0.5, amp=0.4):
    return offset + amp * np.cos(2.0 * np.pi * freq * x + phase)


def test_analyze_recovers_the_corrected_parameter_and_n_pulses(dev):
    # one fringe across a 1 MHz window centred on 4000.0, true zero-error value at 4000.03
    x = np.linspace(3999.5, 4000.5, 61)
    x_ideal = 4000.03
    freq = 1.0  # cycles per MHz -> exactly one cycle across this 1 MHz span
    phase = -2.0 * np.pi * freq * x_ideal
    y = _cosine_curve(x, freq=freq, phase=phase)
    result = MeasurementResult(xpts=x, avgi=y, avgq=np.zeros_like(y))

    exp = ErrorAmplification(dev, n_pulses=4, window=Sweep(start=3999.5, stop=4000.5, on="ax"))
    fit = exp.analyze(result)

    assert fit.success
    assert fit.derived["n_pulses"] == pytest.approx(4.0)
    assert fit.derived["frequency_corrected"] == pytest.approx(x_ideal, abs=1e-3)


def test_zero_crossing_folds_to_the_representative_nearest_the_window_center(dev):
    """``_zero_crossing`` must be a deterministic fold, independent of which phase branch a
    real fit happens to settle on: ``phase`` and ``phase + 2*pi*k`` describe the IDENTICAL
    curve (cosine is 2*pi-periodic in its argument), so without this fold the reported
    correction would depend on optimizer happenstance rather than physics. Exercised directly
    against a hand-built ``FitResult`` -- ``fit_parity``'s own optimizer is a separate concern
    (covered by ``test_analyze_recovers_the_corrected_parameter_and_n_pulses`` above, at a
    frequency that converges reliably)."""
    window = Sweep(start=3999.5, stop=4000.5, on="ax")
    exp = ErrorAmplification(dev, n_pulses=1, window=window)
    freq = 3.0
    period = 1.0 / freq
    x_ideal = 3999.97  # inside the window, near its low edge

    for k in (-2, -1, 0, 1, 2):
        phase = -2.0 * np.pi * freq * (x_ideal + k * period)
        fit = FitResult(params={"freq": freq, "phase": phase}, stderr={}, derived={}, success=True)
        recovered = exp._zero_crossing(fit)
        assert recovered == pytest.approx(x_ideal, abs=1e-9), f"k={k} broke the fold"


def test_zero_crossing_canonicalizes_a_negative_amplitude_fit(dev):
    """``fit_parity``'s ``amp`` is an UNCONSTRAINED sign: ``(amp, phase)`` and
    ``(-amp, phase + pi)`` are the exact same curve (a lmfit-legal degeneracy, not a fitter
    bug), so an optimizer is free to converge to either. Without correcting for this, the
    reported correction would land on the fringe's ANTI-node (half a period off) whenever the
    optimizer happened to land on the negative-amplitude branch -- silently, since ``success``
    stays True and the residual is identical either way."""
    window = Sweep(start=3999.8, stop=4000.2, on="ax")
    exp = ErrorAmplification(dev, n_pulses=1, window=window)
    freq = 2.5
    x_ideal = 3999.99

    positive = FitResult(
        params={"freq": freq, "phase": -2.0 * np.pi * freq * x_ideal, "amp": 0.4},
        stderr={}, derived={}, success=True,
    )
    negative = FitResult(
        # the identical curve, expressed on the negative-amplitude branch (phase shifted by pi)
        params={"freq": freq, "phase": -2.0 * np.pi * freq * x_ideal + np.pi, "amp": -0.4},
        stderr={}, derived={}, success=True,
    )
    assert exp._zero_crossing(positive) == pytest.approx(x_ideal, abs=1e-9)
    assert exp._zero_crossing(negative) == pytest.approx(x_ideal, abs=1e-9)


def test_analyze_uses_fit_parity_reads_signal_via_readout(dev):
    from strumento.packs.cqed.readout import DispersiveIQ

    x = np.linspace(3999.5, 4000.5, 61)
    y = _cosine_curve(x, freq=1.0, phase=0.0)
    # avgi/avgq shaped so only the readout's projection recovers the plain cosine signal
    avgi = y * np.cos(0.3)
    avgq = y * np.sin(0.3)
    result = MeasurementResult(xpts=x, avgi=avgi, avgq=avgq)

    exp = ErrorAmplification(dev, readout=DispersiveIQ(threshold=0.5, angle=0.3))
    fit = exp.analyze(result)
    assert fit.success
    assert fit.derived["frequency_corrected"] == pytest.approx(4000.0, abs=1e-2)


# -- fixed-period fit (the physical fringe period is known) ---------------------------------
#
# A typical gain window spans a fraction of one amplified fringe (7 π pulses over ±200 codes around
# a π gain of 8192 ≈ 0.17 period), where a free-frequency cosine is degenerate: it can lock onto a
# period ~10× off the physical 2·8192/7 ≈ 2341 and still "pass". With the period fixed by
# physics the model is LINEAR in (a, b, c) — y = a + b·cos(kx) + c·sin(kx) — so it has a unique
# solution and closed-form error bars.


def _anchored_readout():
    from strumento.packs.cqed.readout import DispersiveIQ

    # g at I=0, e at I=1 along angle 0: population == avgi
    return DispersiveIQ(threshold=0.5, angle=0.0, g_center=(0.0, 0.0), e_center=(1.0, 0.0))


def _peaked_population(x, x0, period, contrast=0.9, offset=0.05):
    return offset + contrast * (0.5 + 0.5 * np.cos(2.0 * np.pi * (x - x0) / period))


@pytest.mark.parametrize("x0", [8192.0, 8254.0, 8130.0])
def test_fixed_period_fit_recovers_the_peak_on_a_sub_period_window(dev, x0):
    period = 2.0 * 8192.0 / 7.0          # 7 π pulses: P_e = ½ + ½cos(2π(g − g_ideal)/period)
    x = np.linspace(7992.0, 8392.0, 41)  # the runner's ±200-code window ≈ 0.2 period
    rng = np.random.default_rng(3)
    y = _peaked_population(x, x0, period) + rng.normal(0.0, 0.01, x.size)
    exp = ErrorAmplification(dev, parameter="gain", n_pulses=7, points=41,
                             window=Sweep(start=7992.0, stop=8392.0, on="ax"),
                             readout=_anchored_readout(), fringe_period=period)
    fit = exp.analyze(MeasurementResult(xpts=x, avgi=y, avgq=np.zeros_like(y)))
    assert fit.success
    assert fit.derived["fit_mode"] == "fixed_period"
    assert fit.derived["gain_corrected"] == pytest.approx(x0, abs=4.0 * fit.derived["gain_corrected_stderr"] + 1.0)
    assert 0.0 < fit.derived["gain_corrected_stderr"] < 20.0


def test_fixed_period_fit_finds_the_peak_not_the_trough(dev):
    # Population is anchored, so the zero-error point is the MAXIMUM — a trough at x0 must put the
    # correction half a period away, never at x0.
    period = 1000.0
    x = np.linspace(-200.0, 200.0, 41)
    y = 0.5 - 0.4 * np.cos(2.0 * np.pi * x / period)   # trough at 0 → peaks at ±500
    exp = ErrorAmplification(dev, parameter="gain", n_pulses=7, points=41,
                             window=Sweep(start=-200.0, stop=200.0, on="ax"),
                             readout=_anchored_readout(), fringe_period=period)
    fit = exp.analyze(MeasurementResult(xpts=x, avgi=y, avgq=np.zeros_like(y)))
    assert abs(abs(fit.derived["gain_corrected"]) - 500.0) < 1.0
    assert not fit.success  # the peak sits outside the window: refused, not reported as a correction


def test_fixed_period_fit_refuses_an_unanchored_readout(dev):
    from strumento.packs.cqed.readout import DispersiveIQ

    period = 2.0 * 8192.0 / 7.0
    x = np.linspace(7992.0, 8392.0, 41)
    y = _peaked_population(x, 8192.0, period)
    exp = ErrorAmplification(dev, parameter="gain", n_pulses=7, points=41,
                             window=Sweep(start=7992.0, stop=8392.0, on="ax"),
                             readout=DispersiveIQ(threshold=0.5), fringe_period=period)
    fit = exp.analyze(MeasurementResult(xpts=x, avgi=y, avgq=np.zeros_like(y)))
    # a projected signal's sign is arbitrary (peak vs trough), so the fixed-period path refuses it
    assert not fit.success


def test_fixed_period_fit_refuses_pure_noise(dev):
    period = 2.0 * 8192.0 / 7.0
    x = np.linspace(7992.0, 8392.0, 41)
    y = 0.5 + np.random.default_rng(0).normal(0.0, 0.05, x.size)
    exp = ErrorAmplification(dev, parameter="gain", n_pulses=7, points=41,
                             window=Sweep(start=7992.0, stop=8392.0, on="ax"),
                             readout=_anchored_readout(), fringe_period=period)
    assert not exp.analyze(MeasurementResult(xpts=x, avgi=y, avgq=np.zeros_like(y))).success


def test_fringe_period_must_be_positive(dev):
    with pytest.raises(ValueError):
        ErrorAmplification(dev, parameter="gain", fringe_period=0.0)


@pytest.mark.qick
@pytest.mark.parametrize("key,n_pulses,delta", [("pi_ge", 7, 120.0), ("hpi_ge", 10, -60.0)])
def test_fixed_period_recovers_an_injected_gain_error_on_the_simulator(key, n_pulses, delta):
    # The real path (compile → SimulatorSoc → blob-anchored population → fixed-period fit) must
    # recover x_ideal = belief − delta for BOTH gate keys, not echo the belief back.
    pytest.importorskip("qick")
    import strumento.packs.cqed.responses  # noqa: F401
    from strumento.core.simulator import SimulatorSoc
    from strumento.packs.cqed.readout import DispersiveIQ
    from strumento.packs.cqed.responses import truth_from_device

    multimode = DEVICES / "multimode_demo" / "device.yaml"
    soccfg = DEVICES / "_fixtures" / "soccfg_v2_cqed.json"
    device = Device.load(multimode)
    # record blob centres on the device so the response places shots there (the anchored path)
    device.calib.readout.g_center_iq = [0.0, 0.0]
    device.calib.readout.e_center_iq = [1.0, 0.0]
    ro = DispersiveIQ(threshold=0.5, angle=0.0, g_center=(0.0, 0.0), e_center=(1.0, 0.0))
    g0 = float(device.calib.qubit.pulses[key].gain)
    theta = np.pi if key == "pi_ge" else np.pi / 2.0
    period = 2.0 * np.pi * g0 / (n_pulses * theta)
    span = 0.4 * period
    pulse = device.qubit.ge_pi if key == "pi_ge" else device.qubit.ge_hpi
    exp = ErrorAmplification(device, pulse=pulse, parameter="gain", n_pulses=n_pulses,
                             window=Sweep(start=g0 - span / 2, stop=g0 + span / 2, on="err_amp"),
                             points=41, readout=ro, fringe_period=period)
    truth = truth_from_device(device, err_amp_gain_delta=delta)
    soc = SimulatorSoc.from_snapshot(soccfg, truth=truth, seed=11)
    out = exp.go(soc, reps=400, progress=False)
    d = out.fit.derived
    assert out.fit.success, d
    assert d["gain_corrected"] == pytest.approx(g0 - delta, abs=max(4.0 * d["gain_corrected_stderr"], 2.0))
    assert abs(d["gain_corrected"] - g0) > 0.5 * abs(delta)  # moved off the belief
    assert 0.8 < d["peak_population"] < 1.2
