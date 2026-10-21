"""Tests for DRAGCalibration.analyze()'s population-anchored zero-crossing.

Regression: the analyzer looked for the zero-crossing of
``readout.project(iq) - 0.5``, assuming the projected signal was a population in ``[0, 1]``
centered on the equator. But ``project`` returns rotated-I in RAW ADC/IQ units (order ±10 on
hardware), so ``- 0.5`` never brackets the true crossing — on real data it fell back to the grid
midpoint (a fake ``alpha_opt`` that merely equals the scan center) while still reporting
``success=True``. The sim path hid it because the response model emits a signal already built
around 0.5.

These tests feed the analyzer HARDWARE-SHAPED data (averaged IQ anchored to g/e blob centres, the
shape production emits) — the experiment-contract discipline: an analyzer must be verified against
the shape production actually produces, not a hand-built [0,1] curve.

Pure: no qick needed.
"""

from __future__ import annotations

import numpy as np
import pytest

from strumento.packs.cqed.experiments.drag_calibration import DRAGCalibration
from tests._synthetic_readout import dispersive_iq, result_from_populations

# Synthetic ge blob centres (raw IQ) live in tests/_synthetic_readout.py: g and e separate on a
# rotated axis, neither near 0.5 — the hardware-like shape the old analyzer mishandled. The angle
# (g->e direction) and threshold (projected midpoint) are derived there, so the readout is
# self-consistent.


@pytest.fixture
def readout():
    """A calibrated dispersive readout WITH blob centres — what the Runner threads in."""
    return dispersive_iq()


def _hardware_shaped_result(alphas, alpha_true, readout, *, slope=3.0):
    """Averaged IQ per alpha for a DRAG scan whose true optimum is ``alpha_true``.

    The DRAG X/2·Y/2 diagnostic leaves the qubit near the equator; leakage/phase error tilts the
    excited population linearly through 0.5 at the optimal alpha. Map that population back onto the
    g/e IQ axis so ``avgi``/``avgq`` are in the same RAW units ``project`` sees on hardware.
    """
    p_e = np.clip(0.5 + slope * (np.asarray(alphas) - alpha_true), 0.0, 1.0)
    return result_from_populations(p_e, xpts=alphas)


def test_recovers_optimum_from_hardware_shaped_iq(dev, readout):
    """The zero-crossing must land at the true optimum for RAW-IQ (off-equator) data."""
    alphas = np.linspace(-0.1, 0.1, 21)
    alpha_true = 0.035
    result = _hardware_shaped_result(alphas, alpha_true, readout)

    exp = DRAGCalibration(dev, alpha_start=-0.1, alpha_stop=0.1, points=21, readout=readout)
    fit = exp.analyze(result)

    assert fit.success
    # The bug reported 0.0 (grid midpoint) here; the population-anchored crossing must find 0.035.
    assert fit.derived["alpha_opt"] == pytest.approx(alpha_true, abs=0.005)
    assert fit.derived["drag_alpha"] == pytest.approx(alpha_true, abs=0.005)


def test_recovers_negative_optimum_from_hardware_shaped_iq(dev, readout):
    """A second, negative optimum — guards against a coincidental midpoint match at 0.0."""
    alphas = np.linspace(-0.1, 0.1, 21)
    alpha_true = -0.04
    result = _hardware_shaped_result(alphas, alpha_true, readout)

    exp = DRAGCalibration(dev, alpha_start=-0.1, alpha_stop=0.1, points=21, readout=readout)
    fit = exp.analyze(result)

    assert fit.success
    assert fit.derived["alpha_opt"] == pytest.approx(alpha_true, abs=0.005)


# --- The calibrated pulse must BE the gate pulse ------------------------------------------------
#
# Regression: the calibration built its DRAG Arb at a fixed 64 samples, which an Arb
# plays back-to-back at the DAC rate — 10 ns on a 6.4 GS/s DAC, against the 140 ns (4σ, σ=35 ns)
# pulse the transmon factory ships. Same normalized I/Q profile, 14x shorter: the hpi_ge gain then
# rotated only ~6° (so the X/2·Y/2 diagnostic sat at P_e ≈ 0.006 and could never cross the equator on
# hardware), and α — a Q/I peak ratio, whose optimum scales as 1/σ — would not have transferred to
# the gate even if it had. The "byte-identical shape" invariant held per-sample, not in time.


def _played_pulses(seq, label):
    return [op for op in seq.ops if getattr(op, "label", None) == label]


@pytest.mark.parametrize("alpha", [0.0, 0.015, -0.05])
def test_calibration_plays_the_factory_gate_pulse(dev, readout, alpha):
    """Each block's X/2 is the exact pulse ``ge_hpi()`` ships once ``drag_alpha = α`` is written."""
    from strumento.core.compiler import _GAUSS_LENGTH_SIGMAS

    exp = DRAGCalibration(dev, alpha_start=alpha, alpha_stop=alpha, points=1, readout=readout)
    seq, _ = exp.sequence()
    (cal_x,) = _played_pulses(seq, "drag_hpi_x")
    (cal_y,) = _played_pulses(seq, "drag_hpi_y")

    dev.calib.qubit.drag_alpha = alpha
    gate = dev.qubit.ge_hpi()
    assert cal_x.envelope == gate.envelope
    assert cal_y.envelope == gate.envelope
    assert cal_x.gain == pytest.approx(gate.gain)

    # And in TIME, not just per-sample: the window spans the Gaussian's 4σ at the drive DAC rate.
    sigma = dev.calib.qubit.pulses["hpi_ge"].sigma
    dac_rate = dev.qubit._dac_rate(dev.calib.qubit.f_ge)
    assert len(cal_x.envelope.idata) / dac_rate == pytest.approx(
        _GAUSS_LENGTH_SIGMAS * sigma, abs=1.0 / dac_rate
    )


def _first_order_drag_alpha(dev, lam):
    """Expected optimum of the Q/I peak ratio for first-order DRAG on this device's hpi pulse.

    Ω_Q = -λ Ω̇_I / Δ with a peak-normalized Gaussian I (|dI/dt|_max = e^{-1/2}/σ) gives
    Q_pk/I_pk = λ e^{-1/2} / (σ |Δ|), Δ = 2π (f_ge - f_ef) in rad/µs.
    """
    qc = dev.calib.qubit
    sigma = qc.pulses["hpi_ge"].sigma
    delta = 2.0 * np.pi * abs(qc.f_ge - qc.f_ef)
    return lam * np.exp(-0.5) / (sigma * delta)


def test_default_sweep_resolves_the_physical_optimum(dev):
    """The default scan brackets the first-order optimum (λ ∈ [0.5, 1]) with grid points inside it.

    The old ±2 default stepped 0.2 across an optimum of ~0.01–0.02: the whole physics lived
    between two grid points, and most of the scan sat in the clamped |α| > 1 regime.
    """
    exp = DRAGCalibration(dev)
    lo, hi = _first_order_drag_alpha(dev, 0.5), _first_order_drag_alpha(dev, 1.0)
    assert exp.alpha_start < 0.0 < lo < hi < exp.alpha_stop
    step = (exp.alpha_stop - exp.alpha_start) / (exp.points - 1)
    assert step <= hi  # at least one grid step resolves the optimum's scale
    assert max(abs(exp.alpha_start), abs(exp.alpha_stop)) <= 1.0


@pytest.mark.parametrize("bounds", [(-2.5, 2.5), (-0.1, 2.01), (-3.0, 0.1)])
def test_rejects_alpha_beyond_the_q_clamp(dev, readout, bounds):
    """Beyond ``ALPHA_LIMIT`` (2) the sweep is refused at construction, loudly."""
    with pytest.raises(ValueError, match="alpha"):
        DRAGCalibration(dev, alpha_start=bounds[0], alpha_stop=bounds[1], readout=readout)


@pytest.mark.parametrize("alpha", [1.5, -2.0])
def test_large_alpha_is_played_faithfully_via_gain_headroom(dev, readout, alpha):
    """|α| > 1 must play gain·(I + iQ) == hpi_gain·(G + iα·dĜ) exactly — no Q clipping.

    ``drag_arb`` clips Q at ±1, so at α = 2 ~80% of Q samples saturate (Q area 0.65 of nominal) and
    the swept axis would mislabel what was played. The calibration instead divides the envelope by
    |α| and multiplies the gain by |α|: the product the qubit sees is unchanged, nothing clips.
    """
    from strumento.packs.cqed.library._drag import drag_arb

    exp = DRAGCalibration(dev, alpha_start=alpha, alpha_stop=alpha, points=1, readout=readout)
    seq, _ = exp.sequence()
    (pulse,) = _played_pulses(seq, "drag_hpi_x")
    dev.calib.qubit.drag_alpha = 0.0
    gain0 = dev.qubit.ge_hpi().gain

    dac_rate = dev.qubit._dac_rate(dev.calib.qubit.f_ge)
    sigma = dev.calib.qubit.pulses["hpi_ge"].sigma
    gauss = np.asarray(drag_arb(sigma, 0.0, dac_rate=dac_rate).idata)
    dgauss_unit = np.asarray(drag_arb(sigma, 1.0, dac_rate=dac_rate).qdata)  # peak 1, unclipped

    played_i = pulse.gain * np.asarray(pulse.envelope.idata)
    played_q = pulse.gain * np.asarray(pulse.envelope.qdata)
    assert played_i == pytest.approx(gain0 * gauss, rel=1e-9, abs=1e-12)
    assert played_q == pytest.approx(gain0 * alpha * dgauss_unit, rel=1e-9, abs=1e-12)
    assert max(np.max(np.abs(pulse.envelope.idata)), np.max(np.abs(pulse.envelope.qdata))) <= 1.0
    assert pulse.gain == pytest.approx(gain0 * abs(alpha))


def test_headroom_gain_is_validated_against_the_drive_limit(dev, readout):
    """The |α|-scaled gain still faces the qubit-drive safety limit (it bypasses the factory)."""
    from strumento.packs.cqed.safety import GainOutOfRange

    dev.calib.qubit.max_drive_gain = 1.5 * dev.qubit.ge_hpi().gain  # below 2x the hpi gain
    exp = DRAGCalibration(dev, alpha_start=-2.0, alpha_stop=2.0, points=5, readout=readout)
    with pytest.raises(GainOutOfRange):
        exp.sequence()


def test_factory_refuses_to_ship_a_clipped_drag_pulse(dev):
    """The gate factory plays drag_arb directly (clipped beyond |α| = 1), so a calibrated
    |drag_alpha| > 1 would ship a different pulse than the one measured — refuse instead."""
    dev.calib.qubit.drag_alpha = 1.5
    with pytest.raises(ValueError, match="drag_alpha"):
        dev.qubit.ge_hpi()


def test_sim_default_optimum_lies_inside_the_default_sweep(dev):
    """The simulator's prior optimum must sit inside the default scan, or --sim proves nothing."""
    from strumento.packs.cqed.responses import DEFAULT_TRUTH

    exp = DRAGCalibration(dev)
    assert exp.alpha_start < DEFAULT_TRUTH["drag_alpha_opt"] < exp.alpha_stop


# --- Without g/e blob centres there is no population, so there is no equator ------------------
#
# Regression: with no recorded blob centres the analyzer fell back to the raw projected
# axis and still hunted for the 0.5 crossing — the bug described at the top of this file, alive
# on any device without blobs. The experiment contract exercised exactly that path and "passed"
# on a wrong-sign fit (alpha_opt = -0.50 against a truth of +0.50, success=True), because raw IQ
# at the simulator's output scale crosses 0.5 wherever the first sample happens to exceed it.


def test_refuses_without_blob_centres(dev):
    alphas = np.linspace(-0.1, 0.1, 21)
    bare = dispersive_iq(with_centres=False)  # calibrated axis, NO g/e centres
    exp = DRAGCalibration(dev, readout=bare)
    fit = exp.analyze(_hardware_shaped_result(alphas, 0.035, bare))

    assert not fit.success
    assert "alpha_opt" not in fit.derived  # no fabricated optimum, not even a placeholder
    assert "drag_alpha" not in fit.derived
    assert fit.derived["population_anchored"] == 0.0
    assert fit.derived["alphas"] == pytest.approx(list(alphas))  # still plottable against α


def test_population_anchored_flag_on_anchored_fits(dev, readout):
    alphas = np.linspace(-0.1, 0.1, 21)
    exp = DRAGCalibration(dev, readout=readout)
    assert exp.analyze(_hardware_shaped_result(alphas, 0.035, readout)).derived[
        "population_anchored"] == 1.0
    assert exp.analyze(_no_crossing_result(alphas)).derived["population_anchored"] == 1.0


def test_display_without_blobs_does_not_call_raw_iq_a_population(dev):
    pytest.importorskip("matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    from strumento.packs.cqed.plot import _display_drag

    alphas = np.linspace(-0.1, 0.1, 21)
    bare = dispersive_iq(with_centres=False)
    result = _hardware_shaped_result(alphas, 0.035, bare)
    fit = DRAGCalibration(dev, readout=bare).analyze(result)
    ax = _display_drag(result, fit).axes[0]

    assert r"\alpha" in ax.get_xlabel()
    assert "P_e" not in ax.get_ylabel() and "ADC" in ax.get_ylabel()
    assert "blob" in ax.get_title()
    assert not any("equator" in ln.get_label() for ln in ax.get_lines())  # no 0.5 line on raw IQ


# --- Every block must start from |g> --------------------------------------------------------------
#
# Finding (2026-10-07): a scan plateaued at P_e ≈ 0.5 for every α ≥ 0. Root cause: blocks were
# played back-to-back (X/2 · Y/2 · measure, next α) with no reset — a measurement is not a reset, so
# block k started from block k-1's post-measurement state: P_k − ½ = (p_k − ½)(1 − 2 P_{k−1}). Once
# the chain reaches ½ every later block reads ½ regardless of α. TransmonAllXY already carries the
# fix (passive reset before each pair); DRAG did not.


def test_every_block_starts_with_a_passive_reset(dev, readout):
    from strumento.core.pulses import Delay

    exp = DRAGCalibration(dev, alpha_start=-0.1, alpha_stop=0.1, points=5, readout=readout)
    seq, _ = exp.sequence()
    relax = float(dev.calib.readout.relax_delay)
    ops = seq.ops
    x_idx = [i for i, op in enumerate(ops) if getattr(op, "label", None) == "drag_hpi_x"]
    assert len(x_idx) == 5
    for i in x_idx:
        # The op immediately before each block's first drive pulse is a reset delay ≥ relax_delay —
        # including the first block (the previous shot's final readout precedes it).
        assert isinstance(ops[i - 1], Delay), f"block at op {i} has no reset before it"
        assert float(ops[i - 1].duration_us) >= relax


def test_reset_delay_is_overridable(dev, readout):
    from strumento.core.pulses import Delay

    exp = DRAGCalibration(dev, points=3, readout=readout, reset_delay_us=1234.0)
    seq, _ = exp.sequence()
    delays = [op.duration_us for op in seq.ops if isinstance(op, Delay)]
    assert delays == [1234.0] * 3


# --- Crossing estimator: a fitted line with an uncertainty, not the first noisy sign change -------
#
# Regression: the analyzer took the FIRST sign change of the noisy P_e − ½ scanning
# from the left, so on a shallow slope it landed on whichever early point happened to dip — biased
# by several hundredths in α, with success=True even on a pure-noise scan and no uncertainty.


def _first_sign_change(alphas, pops):
    """The OLD estimator: the first α (from the left) where P_e − ½ changes sign."""
    d = np.sign(np.asarray(pops) - 0.5)
    i = int(np.flatnonzero(d[1:] != d[:-1])[0])
    return float(alphas[i])


def test_flat_noise_scan_is_refused(dev, readout):
    """A flat scan (no α dependence) under binomial shot noise has no resolvable crossing."""
    rng = np.random.default_rng(1)
    alphas = np.linspace(-0.1, 0.1, 21)
    pops = rng.binomial(300, 0.48, size=alphas.size) / 300
    exp = DRAGCalibration(dev, alpha_start=-0.1, alpha_stop=0.1, points=21, readout=readout)
    fit = exp.analyze(result_from_populations(pops))
    assert not fit.success
    assert "alpha_opt" not in fit.derived and "drag_alpha" not in fit.derived
    assert abs(fit.derived["slope"] / fit.derived["slope_stderr"]) < 3.0


def test_shallow_noisy_scan_lands_on_the_fitted_crossing_not_the_first_dip(dev, readout):
    """Shallow slope + shot noise + one early point straddling ½: the old first-sign-change answer
    sits far left; the fitted line lands on the truth with an error bar that excludes the old one."""
    rng = np.random.default_rng(2)
    alphas = np.linspace(-0.25, 0.25, 61)
    a_true = -0.05
    pops = rng.binomial(100, 0.5 + 0.7 * (alphas - a_true)) / 100
    pops[5] = 0.55  # an early noisy point above the equator — where the old estimator stopped
    old = _first_sign_change(alphas, pops)
    exp = DRAGCalibration(dev, alpha_start=-0.25, alpha_stop=0.25, points=61, readout=readout)
    fit = exp.analyze(result_from_populations(pops))
    assert fit.success
    a0, se = fit.derived["alpha_opt"], fit.derived["alpha_opt_stderr"]
    assert 0.004 < se < 0.03
    assert abs(a0 - a_true) <= 3 * se  # the truth is inside the reported error bar
    assert abs(a0 - a_true) < 0.05
    assert abs(a0 - old) > 3 * se  # the old first-dip answer is excluded by the new error bar


def test_wide_scan_fits_the_central_linear_region(dev, readout):
    """On a ±1 S-curve (saturating at the ends) the line is fitted locally around the crossing."""
    rng = np.random.default_rng(3)
    alphas = np.linspace(-1.0, 1.0, 61)
    a_true = 0.02
    pops = rng.binomial(100, 0.5 + 0.42 * np.tanh(2.5 * (alphas - a_true))) / 100
    exp = DRAGCalibration(dev, alpha_start=-1.0, alpha_stop=1.0, points=61, readout=readout)
    fit = exp.analyze(result_from_populations(pops))
    assert fit.success
    a0, se = fit.derived["alpha_opt"], fit.derived["alpha_opt_stderr"]
    assert abs(a0 - a_true) <= 3 * se  # the truth is inside the reported error bar
    assert abs(a0 - a_true) < 0.05
    lo, hi = fit.derived["fit_alpha_min"], fit.derived["fit_alpha_max"]
    assert -0.35 < lo < fit.derived["alpha_opt"] < hi < 0.35


def test_non_monotonic_scan_picks_the_central_crossing(dev, readout):
    """The ideal ±2 X/2·Y/2 response re-crosses ½ near |α| ≈ 1.5; the fit must stay on the central one."""
    alphas = np.linspace(-2.0, 2.0, 81)
    a_true = 0.03
    pops = 0.5 + 0.4 * np.sin(2.0 * (alphas - a_true))  # crossings at a_true and a_true ± π/2
    exp = DRAGCalibration(dev, alpha_start=-2.0, alpha_stop=2.0, points=81, readout=readout)
    fit = exp.analyze(result_from_populations(pops))
    assert fit.success
    assert fit.derived["alpha_opt"] == pytest.approx(a_true, abs=0.01)


def test_crossing_outside_the_scan_is_refused_not_extrapolated(dev, readout):
    alphas = np.linspace(0.1, 0.3, 21)
    pops = 0.5 + 0.7 * alphas  # significant slope, crossing at α = 0 — outside [0.1, 0.3]
    exp = DRAGCalibration(dev, alpha_start=0.1, alpha_stop=0.3, points=21, readout=readout)
    fit = exp.analyze(result_from_populations(pops))
    assert not fit.success
    assert "alpha_opt" not in fit.derived


def test_crossing_is_unbiased_and_its_error_bar_covers(dev, readout):
    """Monte Carlo at hardware-like noise (binomial, 100 reps): mean within 0.003 of truth, and the
    reported ±1σ covers the truth ~68% of the time (accept 55–80% over 300 draws)."""
    rng = np.random.default_rng(20261007)
    alphas = np.linspace(-0.25, 0.25, 41)
    a_true, slope, reps = -0.04, 0.7, 100
    p = 0.5 + slope * (alphas - a_true)
    exp = DRAGCalibration(dev, alpha_start=-0.25, alpha_stop=0.25, points=41, readout=readout)
    est, cover = [], 0
    for _ in range(300):
        fit = exp.analyze(result_from_populations(rng.binomial(reps, p) / reps))
        assert fit.success
        a0, se = fit.derived["alpha_opt"], fit.derived["alpha_opt_stderr"]
        est.append(a0)
        cover += abs(a0 - a_true) <= se
    assert np.mean(est) == pytest.approx(a_true, abs=0.003)
    assert 0.55 <= cover / 300 <= 0.80


# --- Display: the plot's x-axis must be the swept DRAG alpha on BOTH fit outcomes ---------------
#
# Regression: a no-crossing fit returned only ``alpha_opt`` in ``derived``, so
# ``_display_drag`` fell to its legacy branch and plotted raw averaged-I against POINT INDEX — the
# failed scan (the one you most need to read, to know which way to shift the range) lost its alpha
# axis, and the "no equator crossing" title was unreachable.


def _no_crossing_result(alphas):
    """Hardware-shaped scan whose optimum lies far outside the range: P_e never reaches 0.5."""
    return _hardware_shaped_result(alphas, alpha_true=0.5, readout=None, slope=0.4)


def test_no_crossing_fit_keeps_alpha_axis_and_population(dev, readout):
    alphas = np.linspace(-0.1, 0.1, 21)
    exp = DRAGCalibration(dev, alpha_start=-0.1, alpha_stop=0.1, points=21, readout=readout)
    fit = exp.analyze(_no_crossing_result(alphas))

    assert not fit.success
    assert fit.derived["alphas"] == pytest.approx(list(alphas))
    signal = np.asarray(fit.derived["signal"])
    assert signal.shape == alphas.shape
    assert np.all(signal < 0.5)  # population units, entirely below the equator


@pytest.mark.parametrize("crossing", [True, False], ids=["success", "no-crossing"])
def test_display_plots_against_swept_alpha(dev, readout, crossing):
    pytest.importorskip("matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    from strumento.packs.cqed.plot import _display_drag

    alphas = np.linspace(-0.1, 0.1, 21)
    result = (_hardware_shaped_result(alphas, 0.035, readout) if crossing
              else _no_crossing_result(alphas))
    exp = DRAGCalibration(dev, alpha_start=-0.1, alpha_stop=0.1, points=21, readout=readout)
    fit = exp.analyze(result)
    assert fit.success is crossing

    fig = _display_drag(result, fit)
    ax = fig.axes[0]
    assert r"\alpha" in ax.get_xlabel()
    assert "P_e" in ax.get_ylabel()
    data_line = ax.get_lines()[0]
    assert np.asarray(data_line.get_xdata()) == pytest.approx(alphas)
    assert np.asarray(data_line.get_ydata()) == pytest.approx(fit.derived["signal"])
    vlines = [ln for ln in ax.get_lines() if ln.get_label().startswith(r"$\alpha_{opt}")]
    if crossing:
        assert len(vlines) == 1
        assert np.asarray(vlines[0].get_xdata())[0] == pytest.approx(fit.derived["alpha_opt"])
        assert "\\pm" in vlines[0].get_label()  # the crossing is shown WITH its 1σ
        fit_lines = [ln for ln in ax.get_lines() if ln.get_label().startswith("linear fit")]
        assert len(fit_lines) == 1
        fx = np.asarray(fit_lines[0].get_xdata())
        assert fx.min() >= fit.derived["fit_alpha_min"] - 1e-9
        assert fx.max() <= fit.derived["fit_alpha_max"] + 1e-9
    else:
        assert vlines == []  # no fabricated optimum drawn on a failed fit
        assert "no resolved equator crossing" in ax.get_title()
