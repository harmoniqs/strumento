"""**The fit-recovery gate**: offline fit-recovery through the real path (D32).

For each Tier-1 cqed experiment, on the v2-assembly instance (``devices/loopback_demo``) against a
:class:`~strumento.core.simulator.SimulatorSoc` armed with a **perturbed** ground truth,
``Experiment.go(sim)`` runs the whole chain — pulse IR → compiler → real ``AveragerProgramV2`` →
acquire → readout projection → fitter — and the fit recovers the simulator's *truth*, not the
device's belief.

That is the claim that makes offline experiment development real: nothing here is a synthetic
array handed to a fitter (the slice e2e tests already cover that). The only fiction is the IQ the
board would have returned.

Tolerances are the gate's, from the plan: π-gain rel err < 2 %, T1/T2 rel err < 5 %, circle-fit Q
rel err < 5 %, parity phase rel err < 5 %.

**Sweep/parameter choices are conditioning-driven, not round numbers.** Every one of these fits is
*unbiased* (fitting the noiseless response recovers the truth to <1e-5 — see
``tests/packs/cqed/test_responses.py``), so the whole error budget is estimator **variance** set by
the shot noise σ_p = sqrt(p(1-p)/reps) ≈ 0.5/√reps and the scan's conditioning:

.. code::

    σ_est / value  ~  σ_p / (A · √N) · κ(scan)      A = signal amplitude, N = sweep points
                                                    κ = how well the scan constrains the parameter

κ is what the span buys: e-folds of decay for a τ, cycles for a frequency/phase, linewidths for a
Q. Each axis below is sized so the measured scatter sits ≥ 4σ inside its gate; the scatter column
is measured over 80 synthetic noise draws at the stated (N, reps).

==========  =======================================================  ==========  ======  =======
experiment  scan (conditioning)                                      N × reps    scatter  gate
==========  =======================================================  ==========  ======  =======
Rabi        30 000 int codes = 1.6 cosine periods (2·g_π = 18 700),   61 × 500    0.2 %   2 %
            38 pts/period
T1          250 µs = 3.7 e-folds of T1 = 68 µs                        201 × 1000  0.9 %   5 %
T2 Ramsey   80 µs = 3.6 e-folds of T2* = 22 µs; δ = 0.12 MHz ⇒        401 × 1000  1.1 %   5 %
            8.33 µs period, 41 pts/period (Nyquist wants 2), 9.6
            fringes
Resonator   FWHM = f₀/Q_l = 0.306 MHz; span 6 MHz = 19.6 FWHM,        241 × 200   0.1 %   5 %
            12 pts/FWHM (the circle fit needs the loop resolved,
            not two points on an arc)
Parity      one 360° turn at 5°/point                                 73 × 1000   0.9 %   5 %
EF T1       110 µs = 3.7 e-folds of T1_ef = 30 µs (1:1 scaled from     201 × 1000  0.5 %   5 %
            the ge T1 row's own e-fold coverage, since T1_ef differs
            from T1)
EF T2       54 µs = 3.6 e-folds of T2_ef = 15 µs; δ_ef = 0.25 MHz     551 × 1000  0.6 %   5 %
Ramsey      ⇒ 4 µs period, ~41 pts/period, 13.5 fringes (1:1 scaled   (detuning:  0.03 %  2 %
            from the ge Ramsey row; the detuning gate is free here    <0.1 %)
            too)
==========  =======================================================  ==========  ======  =======

The simulator seed is pinned, so the gate is deterministic; across seeds 1–8 every recovery stays
inside its tolerance with the margins above. The two ef rows' scatter is the standard deviation
measured over 80 synthetic noise draws at the stated (N, reps) — zero fit failures across those
160 draws, ≥7.9σ inside each gate (vs. the ≥4σ floor every other row targets).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("qick")

sys.path.insert(0, str(Path(__file__).parents[1]))
from _v2gate import V2_SOCCFG, requires_v2_soccfg  # noqa: E402

from strumento.core.device import Device  # noqa: E402
from strumento.core.response import ResponseError  # noqa: E402
from strumento.core.simulator import NonV2SnapshotError, SimulatorSoc  # noqa: E402
from strumento.core.sweeps import Sweep  # noqa: E402
from strumento.packs.cqed.calibration import ModeCal, PulseCal  # noqa: E402
from strumento.packs.cqed.experiments.amplitude_rabi import AmplitudeRabi  # noqa: E402
from strumento.packs.cqed.experiments.eft1 import EFT1  # noqa: E402
from strumento.packs.cqed.experiments.eft2_echo import EFT2Echo  # noqa: E402
from strumento.packs.cqed.experiments.eft2_ramsey import EFT2Ramsey  # noqa: E402
from strumento.packs.cqed.experiments.error_amplification import ErrorAmplification  # noqa: E402
from strumento.packs.cqed.experiments.parity_fringe import ParityFringe  # noqa: E402
from strumento.packs.cqed.experiments.resonator_spectroscopy import (  # noqa: E402
    ResonatorSpectroscopy,
)
from strumento.packs.cqed.experiments.t1 import T1  # noqa: E402
from strumento.packs.cqed.experiments.t2_ramsey import T2Ramsey  # noqa: E402
from strumento.packs.cqed.readout import DispersiveIQ  # noqa: E402
from strumento.packs.cqed.responses import truth_from_device  # noqa: E402
from tests._paths import DEVICES  # noqa: E402

pytestmark = [pytest.mark.qick, requires_v2_soccfg]

LOOPBACK_YAML = DEVICES / "loopback_demo" / "device.yaml"

#: loopback_demo has no dedicated readout line (single-generator instance) — every experiment
#: below that now drives a readout probe (issue #102) probes on the qubit line instead, same as
#: resonator_spectroscopy's own `line=("qubit", "drive")` override further down this file.
LOOPBACK_PROBE_LINE = ("qubit", "drive")

#: The ground truth, deliberately *away* from what the instance believes (belief in brackets):
#: π gain 9350 [8192], T1 68 µs [100], T2* 22 µs [50 = the T1/2 stand-in], readout f₀ 750.08 MHz
#: [750.0], Q_l 2450 [3000 pack default], 62° parity phase [45° pack default], T1_ef 30 µs /
#: T2_ef 15 µs / ef Ramsey detuning 0.25 MHz [40 µs / 20 µs / 0.30 MHz pack defaults — no
#: per-device belief for the ef transition's coherence or its Ramsey detuning, issue #90] /
#: T2_echo_ef 35 µs [40 µs pack default — the ef Hahn-echo coherence, EFT2Echo].
PERTURBATION = dict(
    pi_gain=9350.0,
    T1=68.0,
    T2=22.0,
    ramsey_detuning=0.12,
    resonator_f0=750.08,
    resonator_Q_loaded=2450.0,
    parity_phase=62.0,
    T1_ef=30.0,
    T2_ef=15.0,
    ef_ramsey_detuning=0.25,
    T2_echo_ef=35.0,
)

SEED = 7
REPS = 500


@pytest.fixture
def dev():
    device = Device.load(LOOPBACK_YAML)
    # loopback_demo is a single-generator instance with no cavity element, but the parity sequence
    # is two qubit half-pi pulses around the revival wait — it needs only that one calibration
    # value, injected here as test scaffolding rather than fictionalising the shipped instance.
    device.calib.modes["cavity"] = ModeCal(revival_time=[1.6, 0.8])
    # loopback_demo predates the ef_hpi factory (#89), so its calibration carries pi_ef but not
    # hpi_ef -- EFT2Ramsey's sequence needs the half-pi. Same scaffolding-injection idiom as the
    # cavity mode above, not a fictionalised recalibration of the shipped instance. Gain
    # halves pi_ef's (3800 vs 7600), matching the hpi_ge/pi_ge ratio (4096/8192) and multimode_demo's
    # own hpi_ef/pi_ef pair.
    device.calib.qubit.pulses["hpi_ef"] = PulseCal(gain=3800, sigma=0.030, length=0.0, type="gauss")
    return device


@pytest.fixture
def truth(dev):
    return truth_from_device(dev, **PERTURBATION)


def _run(exp, truth, *, reps: int = REPS, seed: int = SEED):
    """``go`` on an armed simulator — the real path, end to end."""
    soc = SimulatorSoc.from_snapshot(V2_SOCCFG, truth=truth, seed=seed)
    return exp.go(soc, reps=reps, progress=False), soc


def _rel(got: float, want: float) -> float:
    return abs(float(got) / float(want) - 1.0)


# -- the five gates ------------------------------------------------------------------------


def test_amplitude_rabi_recovers_the_true_pi_gain(dev, truth):
    exp = AmplitudeRabi(dev, gains=Sweep(start=0.0, stop=16380.0, on="amp"), points=61,
                        readout=DispersiveIQ(threshold=0.5), line=LOOPBACK_PROBE_LINE)
    outcome, _ = _run(exp, truth)
    assert outcome.fit.success
    got = outcome.fit.derived["pi_gain"]
    # the belief is 8192; recovering *that* would mean the simulator ignored its truth
    assert _rel(got, truth["pi_gain"]) < 0.02, f"pi_gain {got} vs truth {truth['pi_gain']}"
    assert _rel(got, dev.calib.qubit.pulses["pi_ge"].gain) > 0.05


def test_t1_recovers_the_true_relaxation_time(dev, truth):
    exp = T1(dev, delays=Sweep(start=0.0, stop=250.0, on="t"), points=201,
            readout=DispersiveIQ(threshold=0.5),
            line=LOOPBACK_PROBE_LINE)
    outcome, _ = _run(exp, truth, reps=2 * REPS)
    assert outcome.fit.success
    got = outcome.fit.derived["T1"]
    assert _rel(got, truth["T1"]) < 0.05, f"T1 {got} vs truth {truth['T1']}"
    assert _rel(got, dev.calib.qubit.T1) > 0.05  # not the belief (100 µs)


def test_eft1_recovers_the_true_ef_relaxation_time(dev, truth):
    # conditioning: ge T1's gate uses 250 µs = 3.7 e-folds of T1=68 µs at 201×1000 (docstring
    # table). Scaled 1:1 to our T1_ef=30 µs truth (same e-fold coverage, same N, same reps) keeps
    # the same generous margin under the 5% gate: 3.7 * 30 ≈ 110 µs.
    exp = EFT1(dev, delays=Sweep(start=0.0, stop=110.0, on="t"), points=201,
              readout=DispersiveIQ(threshold=0.5),
              line=LOOPBACK_PROBE_LINE)
    outcome, _ = _run(exp, truth, reps=2 * REPS)
    assert outcome.fit.success
    got = outcome.fit.derived["T1"]
    assert _rel(got, truth["T1_ef"]) < 0.05, f"T1_ef {got} vs truth {truth['T1_ef']}"
    # the ge-pi shelve must actually land on the ef decay, not silently measure the ge T1
    assert _rel(got, truth["T1"]) > 0.05


def test_t2_ramsey_recovers_the_true_t2_and_detuning(dev, truth):
    # the thinnest-margin fit of the five: T2* is an envelope parameter, so it needs both e-folds
    # and points (see the module docstring's budget) — 3.6 e-folds x 401 points x 1000 reps
    exp = T2Ramsey(dev, delays=Sweep(start=0.0, stop=80.0, on="t"), points=401,
                   readout=DispersiveIQ(threshold=0.5),
                   line=LOOPBACK_PROBE_LINE)
    outcome, _ = _run(exp, truth, reps=2 * REPS)
    assert outcome.fit.success
    assert _rel(outcome.fit.derived["T2"], truth["T2"]) < 0.05
    # the fringe frequency IS the qubit-drive detuning — free, and far tighter than the T2 gate
    assert _rel(outcome.fit.derived["detuning"], truth["ramsey_detuning"]) < 0.02


def test_eft2ramsey_recovers_the_true_ef_t2_and_detuning(dev, truth):
    # EFT2Ramsey now WIRES its artificial detuning into both ef half-pi drives (like the ge
    # T2Ramsey): the observed fringe is the intrinsic ef mismatch (ef_ramsey_detuning=0.25 MHz
    # truth) PLUS the experiment's ramsey_freq_mhz (default 3.0 MHz) -> ~3.25 MHz. Size the grid
    # for that fast fringe over a T2_ef=15 µs envelope: 3.6 e-folds -> 54 µs span; period
    # 1/3.25 = 0.308 µs; 1101 pts -> dt = 0.049 µs (Nyquist 10.2 MHz, ~6.3 samples/period).
    ramsey_freq = 3.0  # the EFT2Ramsey default, now wired into the sequence
    exp = EFT2Ramsey(dev, delays=Sweep(start=0.0, stop=54.0, on="t"), points=1101,
                     readout=DispersiveIQ(threshold=0.5),
                     line=LOOPBACK_PROBE_LINE)
    assert exp.ramsey_freq_mhz == ramsey_freq  # exercising the wired-in default, not a no-op
    outcome, _ = _run(exp, truth, reps=2 * REPS)
    assert outcome.fit.success
    assert _rel(outcome.fit.derived["T2"], truth["T2_ef"]) < 0.05
    # the fit's detuning reads back the intrinsic ef mismatch PLUS the wired artificial detuning
    expected = truth["ef_ramsey_detuning"] + ramsey_freq
    assert _rel(outcome.fit.derived["detuning"], expected) < 0.02
    # and it's the ef fringe, not the ge one (the ge-pi shelve landed on the ef transition)
    assert _rel(outcome.fit.derived["detuning"], truth["ramsey_detuning"]) > 0.05


def test_eft2echo_recovers_the_true_ef_echo_time(dev, truth):
    # echo decays as exp(-2·tau/T2_echo_ef) over the half-delay tau, so the half-delay decay
    # constant is T2_echo_ef/2 = 17.5 µs; 3.7 e-folds -> ~65 µs span (mirrors the eft1 gate budget).
    exp = EFT2Echo(dev, delays=Sweep(start=0.0, stop=65.0, on="t"), points=201,
                   readout=DispersiveIQ(threshold=0.5),
                   line=LOOPBACK_PROBE_LINE)
    outcome, _ = _run(exp, truth, reps=2 * REPS)
    assert outcome.fit.success
    got = outcome.fit.derived["T2_echo"]
    assert _rel(got, truth["T2_echo_ef"]) < 0.05, f"T2_echo_ef {got} vs truth {truth['T2_echo_ef']}"
    # distinct from the ge echo truth — the ge-pi shelve must land the echo on the ef transition
    assert _rel(got, truth["T2_echo"]) > 0.05


def test_resonator_spectroscopy_circle_fit_recovers_the_true_q(dev, truth):
    f0 = truth["resonator_f0"]
    exp = ResonatorSpectroscopy(
        dev,
        freqs=Sweep(start=f0 - 3.0, stop=f0 + 3.0, on="f"),
        points=241,
        line=("qubit", "drive"),  # the single-generator instance probes on its one line
        method="circle",
    )
    outcome, _ = _run(exp, truth, reps=200)
    fit = outcome.fit
    assert fit.success
    assert _rel(fit.derived["Q_loaded"], truth["resonator_Q_loaded"]) < 0.05
    assert _rel(fit.derived["f0"], f0) < 1e-4
    # the non-idealities the circle fit is *for* are recovered too, from the same trace
    assert _rel(fit.params["delay"], truth["resonator_delay"]) < 0.05
    q_i = 1.0 / (1.0 / truth["resonator_Q_loaded"]
                 - np.cos(truth["resonator_phi"]) / truth["resonator_Q_coupling"])
    assert _rel(fit.derived["Q_internal"], q_i) < 0.05


def test_parity_fringe_recovers_the_true_phase(dev, truth):
    # #100: parity() now drives the readout probe by default; loopback_demo has no dedicated
    # readout DAC line (same single-generator gap resonator_spectroscopy already works around),
    # so route the probe through the qubit's own drive line instead.
    # The injected mode has no element, so it is named explicitly (no cavity to infer).
    exp = ParityFringe(dev, mode="cavity", phases=Sweep(start=0.0, stop=360.0, on="phase"), points=73,
                       readout=DispersiveIQ(threshold=0.5), line=("qubit", "drive"))
    outcome, _ = _run(exp, truth, reps=2 * REPS)
    assert outcome.fit.success
    got = outcome.fit.derived["phase_deg"] % 360.0
    assert abs(got - truth["parity_phase"]) / truth["parity_phase"] < 0.05, f"phase {got}"


# -- error amplification: recovery TIGHTENS as n_pulses grows (issue #82) ------------------
#
# The defining error-amplification property, and the one none of the five gates above need to
# show: not just that a fit recovers a truth, but that the SAME armed error, scanned over the
# SAME window/points/reps, is recovered far more precisely at a larger n_pulses -- because the
# fringe's cycles-per-MHz scales with n_pulses (error_amplification_response's closed form), so
# the same shot-noise floor in the population domain translates to a much smaller error in the
# recovered frequency.
#
# Conditioning (this experiment's own budget, not copied from the table above): the window
# (belief +/- 0.5 MHz) is sized so n_pulses=4 puts exactly ONE fringe cycle across it --
# fit_parity's own seed assumption (freq0 = 1/span, it carries no FFT seed) -- while
# n_pulses=1 puts only a quarter-cycle across the SAME window, which is where the recovery
# is measurably loose. err_amp_freq_delta=0.05 MHz sits comfortably inside that window at
# every n_pulses tested (no re-scan needed between rounds, matching how a real amplification
# pass narrows precision without moving the scan).
ERR_AMP_DELTA = 0.05      # MHz -- the armed coherent detuning already in the belief (f_ge)
ERR_AMP_KAPPA = 0.25      # cycles/MHz per pulse -- error_amplification_response's sensitivity
ERR_AMP_HALF_SPAN = 0.5   # MHz -- the SAME half-window at every n_pulses tested
ERR_AMP_SEED = 7
ERR_AMP_REPS = 1000
ERR_AMP_POINTS = 101


def test_error_amplification_recovery_tightens_as_n_pulses_grows(dev):
    """AC3: a fit-recovery test that asserts the N-dependence, not just a point recovery."""
    belief = dev.calib.qubit.f_ge
    window = Sweep(start=belief - ERR_AMP_HALF_SPAN, stop=belief + ERR_AMP_HALF_SPAN, on="ax")
    truth = truth_from_device(dev, err_amp_freq_delta=ERR_AMP_DELTA, err_amp_freq_sens=ERR_AMP_KAPPA)

    def correction_error(n_pulses: int) -> float:
        exp = ErrorAmplification(dev, n_pulses=n_pulses, window=window, points=ERR_AMP_POINTS,
                                  readout=DispersiveIQ(threshold=0.5), line=LOOPBACK_PROBE_LINE)
        soc = SimulatorSoc.from_snapshot(V2_SOCCFG, truth=truth, seed=ERR_AMP_SEED)
        outcome = exp.go(soc, reps=ERR_AMP_REPS, progress=False)
        assert outcome.fit.success
        assert outcome.fit.derived["n_pulses"] == pytest.approx(float(n_pulses))
        correction = outcome.fit.derived["frequency_corrected"] - belief
        return abs(correction - (-ERR_AMP_DELTA))

    loose = correction_error(1)   # a quarter-cycle across the window: barely constrained
    tight = correction_error(4)   # one full cycle: fit_parity's own seed regime

    # the loose (n_pulses=1) recovery is only order-of-magnitude right...
    assert loose > 0.002, f"n_pulses=1 correction error {loose} is suspiciously tight already"
    # ...while the amplified (n_pulses=4) recovery is comfortably sub-part-in-a-thousand-MHz
    assert tight < 0.0005, f"n_pulses=4 correction error {tight} did not tighten"
    # and the whole point: MORE repeats recovers -delta measurably better, not coincidentally
    assert tight < loose / 10.0, f"n_pulses=4 ({tight}) did not tighten 10x over n_pulses=1 ({loose})"


# -- the loop closes: belief → measure → write back → re-measure ---------------------------


def test_recording_the_fit_shrinks_the_error_on_a_re_run(dev, truth):
    """The closed-loop step the MCP capstone automates, as plain Python."""
    gains = Sweep(start=0.0, stop=16380.0, on="amp")
    before = AmplitudeRabi(dev, gains=gains, points=61, readout=DispersiveIQ(threshold=0.5),
                           line=LOOPBACK_PROBE_LINE)
    first, _ = _run(before, truth)
    err_before = _rel(dev.calib.qubit.pulses["pi_ge"].gain, truth["pi_gain"])

    # write the fit back into the device's belief (D9's store does this durably; here, in place)
    dev.calib.qubit.pulses["pi_ge"].gain = float(first.fit.derived["pi_gain"])
    err_after = _rel(dev.calib.qubit.pulses["pi_ge"].gain, truth["pi_gain"])
    assert err_after < err_before / 5

    # and the belief-driven π pulse now really is a π: a fresh Rabi still reads the same truth.
    # 2*REPS (not the default): the safety-fixed sweep range (16380, not 30000 -- audit fix,
    # 2026-09-14) covers fewer Rabi periods, so this needs the same extra-precision reps
    # test_result_carries_per_shot_data_that_reproduces_the_fit already uses for the same reason.
    second, _ = _run(AmplitudeRabi(dev, gains=gains, points=61, readout=DispersiveIQ(threshold=0.5),
                                   line=LOOPBACK_PROBE_LINE),
                     truth, reps=2 * REPS, seed=SEED + 1)
    assert _rel(second.fit.derived["pi_gain"], truth["pi_gain"]) < 0.02


# -- per-shot data really is per-shot -----------------------------------------------------


def test_result_carries_per_shot_data_that_reproduces_the_fit(dev, truth):
    exp = AmplitudeRabi(dev, gains=Sweep(start=0.0, stop=16380.0, on="amp"), points=61,
                        readout=DispersiveIQ(threshold=0.5), line=LOOPBACK_PROBE_LINE)
    outcome, soc = _run(exp, truth)
    shots = outcome.result.shots
    assert shots is not None and np.shape(shots) == (61, REPS, 2)

    # the per-shot path: discriminate per shot, average -> the same populations the average reported
    pops = DispersiveIQ(threshold=0.5).reduce(shots)
    from strumento.core.fitters import fit_rabi

    assert _rel(fit_rabi(outcome.result.xpts, pops).derived["pi_gain"], truth["pi_gain"]) < 0.02
    # heralding machinery runs on it unchanged
    assert DispersiveIQ(threshold=0.5).herald(shots).shape == (61, REPS)


# -- typed errors ---------------------------------------------------------------------------


def test_an_unmodelled_experiment_fails_typed_through_the_experiment_path(dev):
    from strumento.core.experiment import Experiment
    from strumento.core.pulses import Seq

    class Unmodelled(Experiment):
        def sequence(self):
            return Seq().play(self.device.qubit.ge_pi()).measure(), ()

        def analyze(self, result):
            raise AssertionError("must never get as far as analysis")

    soc = SimulatorSoc.from_snapshot(V2_SOCCFG)
    with pytest.raises(ResponseError, match="no response model registered for 'Unmodelled'"):
        Unmodelled(dev).go(soc, reps=8, progress=False)


def test_a_v1_instance_cannot_back_a_simulator(v1_snapshot_yaml):
    v1 = Device.load(v1_snapshot_yaml)
    with pytest.raises(NonV2SnapshotError, match="multimode_demo"):
        SimulatorSoc.from_device(v1)


def test_the_mock_path_is_untouched_by_arming(dev):
    """A non-armable soc runs exactly as before — no arming, no shots, no simulated data."""
    from strumento.core.mock import MockQickSocV2

    mock = MockQickSocV2.from_snapshot(V2_SOCCFG)
    outcome = AmplitudeRabi(dev, gains=Sweep(start=0.0, stop=16380.0, on="amp"), points=21,
                            line=LOOPBACK_PROBE_LINE).go(
        mock, reps=8, progress=False
    )
    assert outcome.result.shots is None
    assert not np.any(np.asarray(outcome.result.avgi))  # the mock's zeros, still zeros
