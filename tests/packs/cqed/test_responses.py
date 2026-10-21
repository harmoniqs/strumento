"""cqed Tier-1 response models (D32) — pure layer, no qick.

These tests pin the *physics side* of the offline simulator without a program in sight: a model is
a function of an armed :class:`SimContext`, so it can be evaluated directly. What matters here is
the three-layer parameter resolution (pack defaults → device belief → armed truth), the shape and
range of what each model returns, and that each closed form is the one its matching fitter fits —
so the e2e fit-recovery gate is measuring the plumbing, not compensating for a mismatched model.
"""

from __future__ import annotations

import numpy as np
import pytest

from strumento.core.fitters import fit_decaysin_detuned, fit_exp, fit_parity, fit_rabi
from strumento.core.resonator_fitters import fit_lorentzian, fit_resonator_circle
from strumento.core.response import SimContext, SimResponse, resolve_response
from strumento.core.sweeps import Sweep
from strumento.packs.cqed.experiments.amplitude_rabi import AmplitudeRabi
from strumento.packs.cqed.experiments.ef_spectroscopy import EFSpectroscopy
from strumento.packs.cqed.experiments.error_amplification import ErrorAmplification
from strumento.packs.cqed.experiments.fock_cavity_t1 import FockCavityT1
from strumento.packs.cqed.experiments.fock_cavity_t2_ramsey import FockCavityT2Ramsey
from strumento.packs.cqed.experiments.parity_fringe import ParityFringe
from strumento.packs.cqed.experiments.resonator_spectroscopy import ResonatorSpectroscopy
from strumento.packs.cqed.experiments.sideband_rabi import SidebandRabi
from strumento.packs.cqed.experiments.t1 import T1
from strumento.packs.cqed.experiments.t2_echo import T2Echo
from strumento.packs.cqed.experiments.t2_ramsey import T2Ramsey
from strumento.packs.cqed.responses import (
    DEFAULT_TRUTH,
    params,
    truth_from_device,
)
from tests._paths import DEVICES

LOOPBACK_YAML = DEVICES / "loopback_demo" / "device.yaml"


@pytest.fixture
def dev(loopback_dev):
    return loopback_dev


def _ctx(dev, experiment, xpts, **truth) -> SimContext:
    return SimContext(experiment=experiment, device=dev, xpts=tuple(xpts), truth=truth)


def _pops(response: SimResponse) -> np.ndarray:
    assert response.iq is None
    return np.asarray(response.populations, dtype=float)


# -- registration + parameter resolution ---------------------------------------------------


@pytest.mark.parametrize(
    "cls", [AmplitudeRabi, T1, T2Ramsey, T2Echo, ResonatorSpectroscopy, ParityFringe, ErrorAmplification,
            FockCavityT1, FockCavityT2Ramsey]
)
def test_pack_import_registers_a_response(cls):
    assert callable(resolve_response(cls))  # registered by `import strumento.packs.cqed`


def test_truth_follows_the_device_belief_by_default(dev):
    truth = truth_from_device(dev)
    assert truth["pi_gain"] == dev.calib.qubit.pulses["pi_ge"].gain == 8192.0
    assert truth["T1"] == dev.calib.qubit.T1 == 100.0
    assert truth["resonator_f0"] == dev.calib.readout.frequency == 750.0
    # a quantity the instance holds no belief about falls back to the pack default
    assert truth["blob_sigma"] == DEFAULT_TRUTH["blob_sigma"]


def test_overrides_perturb_one_key_and_leave_the_rest(dev):
    truth = truth_from_device(dev, pi_gain=9000.0)
    assert truth["pi_gain"] == 9000.0
    assert truth["T1"] == 100.0


def test_armed_truth_beats_the_belief_key_by_key(dev):
    ctx = _ctx(dev, AmplitudeRabi(dev), (0.0,), pi_gain=1234.0)
    resolved = params(ctx)
    assert resolved["pi_gain"] == 1234.0     # armed truth wins
    assert resolved["T1"] == 100.0           # belief still fills in
    assert resolved["p_thermal"] == DEFAULT_TRUTH["p_thermal"]


def test_no_device_still_resolves_to_pack_defaults():
    assert params(SimContext(experiment=None)) == DEFAULT_TRUTH


# -- each model is the closed form its fitter fits -----------------------------------------


def test_rabi_response_is_a_pi_rotation_at_the_true_pi_gain(dev):
    g = np.linspace(0.0, 30000.0, 601)
    resp = resolve_response(AmplitudeRabi)(_ctx(dev, AmplitudeRabi(dev), g, pi_gain=9000.0))
    pops = _pops(resp)
    assert pops.shape == g.shape
    assert 0.0 <= pops.min() and pops.max() <= 1.0
    assert pops[0] == pytest.approx(DEFAULT_TRUTH["p_thermal"], abs=1e-9)  # no drive, no excitation
    # The arch peaks just *below* the true pi gain: the coherence envelope multiplying the cosine
    # pulls the maximum of the product in by ~1 % at the default 8-pi-gain decay length. The fit is
    # unbiased regardless — fit_rabi carries the same envelope, so it recovers the period exactly
    # (see test_rabi_response_is_recovered_by_its_own_fitter).
    assert g[int(np.argmax(pops))] == pytest.approx(9000.0, rel=0.02)
    # the readout model's blob geometry rides along (physics, not a core constant)
    assert resp.blob0 == DEFAULT_TRUTH["blob0"] and resp.sigma == DEFAULT_TRUTH["blob_sigma"]


def test_rabi_response_is_recovered_by_its_own_fitter(dev):
    g = np.linspace(0.0, 30000.0, 121)
    pops = _pops(resolve_response(AmplitudeRabi)(_ctx(dev, AmplitudeRabi(dev), g, pi_gain=9350.0)))
    assert fit_rabi(g, pops).derived["pi_gain"] == pytest.approx(9350.0, rel=1e-3)


def test_sideband_rabi_response_maps_residual_f_to_e_after_the_closing_ef_pi(dev):
    gains = np.array([0.0, 5500.0])
    response = resolve_response(SidebandRabi)(
        _ctx(dev, SidebandRabi(dev), gains, sideband_pi_gain=5500.0)
    )
    populations = _pops(response)

    # With the post-f0g1 ef map, zero sideband drive leaves f mapped bright (e), while a pi swap
    # leaves g,1 dark on the ge readout.
    assert populations[0] > populations[1]


def test_t1_response_is_recovered_by_its_own_fitter(dev):
    t = np.linspace(0.0, 250.0, 101)
    pops = _pops(resolve_response(T1)(_ctx(dev, T1(dev), t, T1=68.0)))
    assert pops[0] > pops[-1]
    assert fit_exp(t, pops).derived["T1"] == pytest.approx(68.0, rel=1e-6)


def test_fock_cavity_t1_response_is_recovered_by_its_own_fitter(dev):
    t = np.linspace(0.0, 1200.0, 81)
    pops = _pops(resolve_response(FockCavityT1)(_ctx(dev, FockCavityT1(dev), t, cavity_T1=210.0)))

    assert pops[0] > pops[-1]
    assert fit_exp(t, pops).derived["T1"] == pytest.approx(210.0, rel=1e-6)


def test_fock_cavity_t1_response_scales_recorded_blob_centers_for_the_simulator(dev):
    dev.calib.readout.g_center_iq = (250.0, -100.0)
    dev.calib.readout.e_center_iq = (-500.0, 400.0)
    response = resolve_response(FockCavityT1)(
        SimContext(
            experiment=FockCavityT1(dev),
            device=dev,
            xpts=(0.0,),
            readout_output_scale=10_000.0,
        )
    )

    assert response.blob0 == pytest.approx((0.025, -0.01))
    assert response.blob1 == pytest.approx((-0.05, 0.04))
    assert response.sigma == pytest.approx(DEFAULT_TRUTH["blob_sigma"] / 10_000.0)


def test_fock_cavity_t2_ramsey_response_recovers_cavity_t2_and_reference_fringe(dev):
    t = np.linspace(0.0, 300.0, 101)
    exp = FockCavityT2Ramsey(dev, reference_detuning_mhz=0.02)
    pops = _pops(
        resolve_response(FockCavityT2Ramsey)(
            _ctx(dev, exp, t, cavity_T2=75.0, cavity_detuning=0.006)
        )
    )
    fit = fit_decaysin_detuned(t, pops)

    assert fit.derived["T2"] == pytest.approx(75.0, rel=1.0e-5)
    assert fit.derived["detuning"] == pytest.approx(0.026, rel=1.0e-5)


def test_t2_response_carries_the_detuning_and_the_envelope(dev):
    t = np.linspace(0.0, 60.0, 241)
    ctx = _ctx(dev, T2Ramsey(dev), t, T2=22.0, ramsey_detuning=0.12)
    pops = _pops(resolve_response(T2Ramsey)(ctx))
    fit = fit_decaysin_detuned(t, pops)
    assert fit.derived["T2"] == pytest.approx(22.0, rel=1e-5)
    assert fit.derived["detuning"] == pytest.approx(0.12, rel=1e-5)


def test_t2_response_follows_the_experiment_phase_advance(dev):
    t = np.linspace(0.0, 60.0, 121)
    model = resolve_response(T2Ramsey)
    plain = _pops(model(_ctx(dev, T2Ramsey(dev), t)))
    flipped = _pops(model(_ctx(dev, T2Ramsey(dev, advance_phase_deg=180.0), t)))
    # a 180-degree advance on the second half-pi mirrors the fringe about its mean
    assert np.allclose(plain + flipped, plain[0] + flipped[0], atol=1e-9)


def test_t2_echo_response_is_recovered_over_total_evolution_time(dev):
    half_delay = np.linspace(0.0, 200.0, 201)
    truth_t2_echo = 80.0
    pops = _pops(
        resolve_response(T2Echo)(
            _ctx(dev, T2Echo(dev), half_delay, T2_echo=truth_t2_echo)
        )
    )
    assert fit_exp(half_delay, pops).derived["tau"] == pytest.approx(
        truth_t2_echo / 2.0, rel=1e-6
    )


def test_ef_spectroscopy_maps_fitted_center_to_ef_truth(dev):
    assert EFSpectroscopy(dev).recovery_map() == {"f0": "f_ef"}


def test_ef_spectroscopy_response_recovers_ef_truth(dev):
    freqs = np.linspace(3800.0, 3840.0, 201)
    truth_f_ef = 3822.0
    pops = _pops(
        resolve_response(EFSpectroscopy)(
            _ctx(dev, EFSpectroscopy(dev), freqs, f_ef=truth_f_ef)
        )
    )
    assert fit_lorentzian(freqs, pops).derived["f0"] == pytest.approx(truth_f_ef, rel=1e-6)


def test_resonator_response_is_a_complex_trace_recovered_by_the_circle_fit(dev):
    f = np.linspace(747.0, 753.0, 401)
    ctx = _ctx(dev, ResonatorSpectroscopy(dev), f, resonator_f0=750.08,
               resonator_Q_loaded=2450.0)
    resp = resolve_response(ResonatorSpectroscopy)(ctx)
    assert resp.populations is None
    s21 = np.asarray(resp.iq)
    assert np.iscomplexobj(s21) and s21.shape == f.shape
    # the dip sits on resonance to within a linewidth — exactly on it only for phi = 0, since the
    # impedance-mismatch angle skews the lineshape (which is why the circle fit exists)
    fwhm = 750.08 / 2450.0
    assert abs(f[int(np.abs(s21).argmin())] - 750.08) < fwhm
    assert np.abs(s21).min() < 0.5 * np.abs(s21).max()  # a real dip, not a ripple
    fit = fit_resonator_circle(f, s21)
    assert fit.success
    assert fit.derived["Q_loaded"] == pytest.approx(2450.0, rel=0.02)
    assert fit.derived["f0"] == pytest.approx(750.08, rel=1e-6)


def test_parity_response_is_recovered_by_its_own_fitter(dev):
    phases = np.linspace(0.0, 360.0, 145)
    ctx = _ctx(dev, ParityFringe(dev, mode="cavity"), phases, parity_phase=62.0)
    pops = _pops(resolve_response(ParityFringe)(ctx))
    fit = fit_parity(phases, pops)
    assert fit.derived["phase_deg"] % 360.0 == pytest.approx(62.0, rel=1e-3)


def test_populations_stay_in_range_under_an_extreme_truth(dev):
    # a truth a user could plausibly type: contrast + thermal > 1. Shots must stay drawable.
    ctx = _ctx(dev, T1(dev), np.linspace(0, 10, 11), p_thermal=0.4, readout_contrast=0.9)
    pops = _pops(resolve_response(T1)(ctx))
    assert pops.min() >= 0.0 and pops.max() <= 1.0


# -- error amplification: the fringe frequency scales with n_pulses, crossing zero-error at
# the TRUE (belief - delta) parameter value ------------------------------------------------


def test_error_amplification_response_recovers_the_true_frequency(dev):
    """Noiseless recovery, n_pulses=1: the closed form IS what ``fit_parity`` fits, at the
    experiment's own sensitivity default (one cycle across the default window)."""
    window = Sweep(start=3999.8, stop=4000.2, on="ax")
    exp = ErrorAmplification(dev, n_pulses=1, window=window)
    x = window.axis_values(161)
    ctx = _ctx(dev, exp, x, err_amp_freq_delta=0.02, err_amp_freq_sens=2.5)
    pops = _pops(resolve_response(ErrorAmplification)(ctx))
    assert 0.0 <= pops.min() and pops.max() <= 1.0
    fit = fit_parity(x, pops)
    assert fit.success
    # ideal = belief (f_ge=4000.0) - delta(0.02) = 3999.98
    assert exp._zero_crossing(fit) == pytest.approx(3999.98, abs=1e-3)


def test_error_amplification_response_recovers_the_true_gain(dev):
    """The same closed form, on the gain branch — genericity, no special-casing."""
    window = Sweep(start=7800.0, stop=8600.0, on="ax")
    exp = ErrorAmplification(dev, parameter="gain", n_pulses=1, window=window)
    x = window.axis_values(161)
    ctx = _ctx(dev, exp, x, err_amp_gain_delta=15.0, err_amp_gain_sens=0.00125)
    pops = _pops(resolve_response(ErrorAmplification)(ctx))
    fit = fit_parity(x, pops)
    assert fit.success
    # ideal = belief (pi_gain=8192) - delta(15) = 8177
    assert exp._zero_crossing(fit) == pytest.approx(8177.0, abs=1.0)


def test_error_amplification_response_fringe_tightens_with_n_pulses(dev):
    """The defining error-amplification signature, directly from the closed form (no fitting
    needed): the number of fringes across a FIXED window scales with n_pulses."""
    window = Sweep(start=3999.8, stop=4000.2, on="ax")
    x = window.axis_values(801)
    crossings = {}
    for n in (1, 3, 9):
        exp = ErrorAmplification(dev, n_pulses=n, window=window)
        ctx = _ctx(dev, exp, x, err_amp_freq_delta=0.0, err_amp_freq_sens=2.5)
        pops = _pops(resolve_response(ErrorAmplification)(ctx))
        centered = pops - pops.mean()
        crossings[n] = int(np.sum(np.diff(np.sign(centered)) != 0))
    assert crossings[9] > crossings[3] > crossings[1] > 0


# -- purity ---------------------------------------------------------------------------------


def test_module_imports_without_qick():
    import subprocess
    import sys

    code = (
        "import sys; sys.modules['qick']=None; "
        "import strumento.packs.cqed.responses; print('ok')"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0 and "ok" in r.stdout, r.stderr
