"""The cqed bring-up graph's new thin Seq experiments, on the sim soc (issue #56).

Each new node is a thin Seq experiment with a vetted fitter, riding the existing experiment
test pattern (sim soc, armed truth): construct → ``go(SimulatorSoc, record=child)`` → the
vetted emitter writes the ``fit`` and — when a truth is armed — the ``fit-recovery`` gate
events into the child record, and the graph routes on the node's returned verdict map.

Covered here, per acceptance criterion:

- AC 2 — each new node (punchout, gain-ladder spectroscopy, readout state, TLS scan,
  validation) runs through the sim soc with armed truth, fits, and emits ``fit`` +
  ``fit-recovery`` gate events through the vetted emitter;
- AC 3 — the qubit-spec frame check gates on the fitted exponents (p_delta ≈ 1, p_gamma ≈ 0.5)
  plus premature-saturation detection: a genuine line passes, impostors do not;
- AC 6 — the validation gates: Ramsey detuning < 50 kHz and readout fidelity ≥ the pack
  threshold, both against the pack constants.
"""

from __future__ import annotations

import pytest

pytest.importorskip("qick")

import strumento.packs.cqed.responses  # noqa: F401,E402  (registers the pack's response models)
from strumento.core.device import Device
from strumento.core.record import read_events
from strumento.core.simulator import SimulatorSoc
from strumento.packs.cqed import EXPERIMENTS as CQED_EXPERIMENTS
from strumento.packs.cqed.bringup import (
    ASSIGNMENT_FIDELITY_MIN,
    KAPPA_MHZ,
    gain_ladder_frame_gates,
    punchout_gates,
    readout_length_gates,
    readout_state_gates,
    validation_gates,
)
from strumento.packs.cqed.responses import truth_from_device
from tests._paths import DEVICES

MULTIMODE_YAML = DEVICES / "multimode_demo" / "device.yaml"
CQED_SOCCFG = DEVICES / "_fixtures" / "soccfg_v2_cqed.json"

REPS = 400
SEED = 7

#: Shared truth perturbation for the node tests — values need only differ from the device's
#: belief so a fit has something to find.
NODE_TRUTH = dict(pi_gain=9100.0, T1=88.0, T2=41.0)

#: The new bring-up node experiments, by registry name — the graph's thin Seq additions.
NEW_NODES = (
    "punchout",
    "gain_ladder_spectroscopy",
    "readout_state",
    "tls_scan",
    "validation_check",
)

#: The smallest scans that still converge per new node (the contract-suite convention).
NODE_KWARGS: dict[str, dict] = {
    "punchout": {},
    "gain_ladder_spectroscopy": dict(points=81),
    "readout_state": {},
    "tls_scan": dict(points=81),
    "validation_check": dict(points=81),
}


def _device():
    return Device.load(MULTIMODE_YAML)


def _soc(dev, truth_overrides: dict | None = None, seed: int = SEED):
    truth = truth_from_device(dev, **(truth_overrides or {}))
    return SimulatorSoc.from_snapshot(CQED_SOCCFG, truth=truth, seed=seed)


def _child(tmp_path, name):
    from strumento.core.record import TaskRecord

    return TaskRecord.create(
        str(tmp_path / f"{name}-child"), title=name, kind="experiment", pid=0, heartbeat=False
    )


def _gates(child) -> dict:
    """The child ledger's last verdict per gate name (a reader takes the last event per name)."""
    out: dict[str, bool] = {}
    for event in read_events(child.dir):
        if event.get("ev") == "gate":
            out[str(event.get("name"))] = bool(event.get("pass"))
    return out


# -- AC 2: each new node rides the sim soc + armed truth, and the vetted emitter writes gates --


@pytest.mark.qick
@pytest.mark.parametrize("name", NEW_NODES)
def test_new_node_runs_through_the_sim_soc_and_emits_the_vetted_gates(tmp_path, name, closing):
    """AC: each new node is a thin Seq experiment with a vetted fitter: through the real
    Runner path on the sim soc with armed truth it returns a typed fit with derived
    quantities, and the child record carries the vetted emitter's ``fit`` AND
    ``fit-recovery`` gate events (the truth is armed, so recovery is checked)."""

    dev = _device()
    soc = _soc(dev)
    child = closing(_child(tmp_path, name))

    cls = CQED_EXPERIMENTS[name]
    outcome = cls(dev, **NODE_KWARGS[name]).go(soc, reps=REPS, record=child, progress=False)

    assert outcome.fit is not None and outcome.fit.success, f"{name}: the fit did not converge"
    assert outcome.fit.derived, f"{name}: no derived quantities"
    gates = _gates(child)
    assert gates.get("fit") is True, f"{name}: the vetted emitter wrote no passing fit gate"
    assert "fit-recovery" in gates, (
        f"{name}: the truth was armed but the vetted emitter emitted no fit-recovery verdict"
    )


# -- AC 4 (gate half): the punchout's kappa estimate is gate-only ------------------------------


def test_punchout_recovers_kappa_and_gates_it_against_the_pack_window():
    """AC: the punchout's kappa estimate is derived gate-only — the fit recovers the armed
    truth's κ through the vetted emitter's fit-recovery comparison, and the pack's
    kappa-window gate consumes the pack prior [0.1, 10] MHz."""
    dev = _device()
    outcome = CQED_EXPERIMENTS["punchout"](dev).go(
        _soc(dev, dict(resonator_kappa_mhz=0.25)), reps=REPS, progress=False
    )

    assert outcome.fit.success
    kappa = outcome.fit.derived["resonator_kappa_mhz"]
    assert KAPPA_MHZ[0] <= kappa <= KAPPA_MHZ[1]
    assert kappa == pytest.approx(0.25, rel=0.35)  # recovered from the depth-collapse knee
    assert punchout_gates(outcome, dev) == {"fit": True, "kappa-window": True}

    # an out-of-window estimate refuses the gate — the pack prior, checked by the gate emitter
    from strumento.core.experiment import ExperimentOutcome

    class _BadFit:
        success = True
        derived = {"resonator_kappa_mhz": 25.0}  # a 25 MHz linewidth: outside [0.1, 10] MHz

    bad = ExperimentOutcome(result=outcome.result, fit=_BadFit())
    assert punchout_gates(bad, dev)["kappa-window"] is False


# -- AC 3: the qubit-spec frame check ----------------------------------------------------------


@pytest.mark.qick
def test_gain_ladder_frame_check_passes_a_genuine_two_level_line(tmp_path, closing):
    """AC: the qubit-spec frame check passes a genuine qubit line — the fitted exponents land
    on the two-level model (p_delta ≈ 1, p_gamma ≈ 0.5), no premature saturation, and the
    lowest rung's centroid sits within the 0.5× cap of the scan's expected center."""
    dev = _device()
    soc = _soc(dev, dict(NODE_TRUTH))
    child = closing(_child(tmp_path, "gain-ladder"))
    exp = CQED_EXPERIMENTS["gain_ladder_spectroscopy"](dev)
    outcome = exp.go(soc, reps=REPS, record=child, progress=False)

    assert outcome.fit.success
    assert outcome.fit.derived["p_delta"] == pytest.approx(1.0, abs=0.35), outcome.fit.derived
    assert outcome.fit.derived["p_gamma"] == pytest.approx(0.5, abs=0.3)
    assert outcome.fit.derived["saturated"] == 0.0  # no premature saturation
    assert outcome.fit.derived["centroid_cap_ok"] == 1.0  # the lowest-rung cap holds
    verdicts = gain_ladder_frame_gates(outcome, dev)
    assert verdicts == {"fit": True, "gain-ladder-frame": True}


@pytest.mark.qick
def test_gain_ladder_frame_check_fails_an_off_model_exponent_impostor():
    """Impostor signature 1 — exponents off-model: a line whose Stark shift grows as s²
    (not s) fails the frame check: the fitted p_delta is off the two-level model."""
    dev = _device()
    soc = _soc(dev, dict(gain_ladder_impostor="shift", **NODE_TRUTH))
    exp = CQED_EXPERIMENTS["gain_ladder_spectroscopy"](dev)
    outcome = exp.go(soc, reps=REPS, progress=False)

    assert outcome.fit.success  # the per-rung fits converge — the LINE is what lies
    assert gain_ladder_frame_gates(outcome, dev)["gain-ladder-frame"] is False


@pytest.mark.qick
def test_gain_ladder_frame_check_detects_premature_saturation():
    """Impostor signature 2 — premature saturation: the line height collapses at rungs where
    the fitted two-level model predicts growth; the frame check refuses the line."""
    dev = _device()
    soc = _soc(dev, dict(gain_ladder_impostor="saturate", **NODE_TRUTH))
    exp = CQED_EXPERIMENTS["gain_ladder_spectroscopy"](dev)
    outcome = exp.go(soc, reps=REPS, progress=False)

    assert outcome.fit.success  # the per-rung fits converge — the LINE is what lies
    verdicts = gain_ladder_frame_gates(outcome, dev)
    assert verdicts["gain-ladder-frame"] is False


# -- AC 6: the validation terminal's measured gates --------------------------------------------


@pytest.mark.qick
def test_validation_gates_detuning_window_and_readout_fidelity(tmp_path):
    """AC: the validation terminal's measured gates — Ramsey detuning < 50 kHz and readout
    fidelity ≥ the pack threshold — pass on a healthy truth and fail the right gate when a
    prior is violated."""
    dev = _device()
    # a good truth: detuning 20 kHz (< 50 kHz), healthy readout blobs
    soc = _soc(dev, dict(ramsey_detuning=0.02, **NODE_TRUTH))
    exp = CQED_EXPERIMENTS["validation_check"](dev)
    verdicts = validation_gates(exp.go(soc, reps=REPS, progress=False), dev)
    assert verdicts["fit"] is True
    assert verdicts["detuning-window"] is True
    assert verdicts["readout-fidelity"] is True

    # a detuning beyond the 50 kHz window: the detuning gate fails (and only it)
    soc = _soc(dev, dict(ramsey_detuning=0.3, **NODE_TRUTH))
    outcome = exp.go(soc, reps=REPS, progress=False)
    verdicts = validation_gates(outcome, dev)
    assert verdicts["detuning-window"] is False
    assert verdicts["fit"] is True

    # a readout chain that cannot separate the blobs: the fidelity gate fails
    dev2 = _device()
    soc = _soc(dev2, dict(blob_sigma=1.2, **NODE_TRUTH))
    outcome = CQED_EXPERIMENTS["validation_check"](dev2).go(soc, reps=REPS, progress=False)
    verdicts = validation_gates(outcome, dev2)
    assert verdicts["readout-fidelity"] is False


@pytest.mark.qick
def test_readout_state_gates_the_shared_assignment_fidelity_threshold(tmp_path):
    """The readout-state node's assignment-fidelity gate consumes the SAME pack threshold the
    terminal validation references: a healthy readout passes, an overlapping-blob readout
    (assignment fidelity below the pack threshold) fails the gate."""
    dev = _device()
    soc = _soc(dev, dict(readout_error=0.02, **NODE_TRUTH))
    exp = CQED_EXPERIMENTS["readout_state"](dev)
    outcome = exp.go(soc, reps=REPS, progress=False)

    assert outcome.fit.success
    fidelity = outcome.fit.derived["assignment_fidelity"]
    assert fidelity >= ASSIGNMENT_FIDELITY_MIN
    assert readout_state_gates(outcome, dev) == {"fit": True, "assignment-fidelity": True}

    # a readout chain that cannot separate the blobs fails the SAME threshold object
    dev2 = _device()
    soc2 = _soc(dev2, dict(blob_sigma=1.0, **NODE_TRUTH))
    outcome2 = CQED_EXPERIMENTS["readout_state"](dev2).go(soc2, reps=REPS, progress=False)
    assert readout_state_gates(outcome2, dev2)["assignment-fidelity"] is False


@pytest.mark.qick
def test_tls_scan_banks_nothing_and_reports_the_line_count_as_evidence(tmp_path):
    """The TLS scan: a high-power sweep whose vetted fitter reports the TLS-line count as
    gate-only evidence (the derived quantities ride the child record's fit metrics); with no
    TLS lines armed it finds none, and with a TLS line armed it resolves it."""
    dev = _device()
    soc = _soc(dev, dict(NODE_TRUTH))  # the default truth: no TLS lines
    exp = CQED_EXPERIMENTS["tls_scan"](dev)
    outcome = exp.go(soc, reps=REPS, progress=False)

    assert outcome.fit.success
    assert outcome.fit.derived["tls_lines"] == 0.0  # a clean line: no TLS evidence

    # a TLS impostor line 3.2 MHz below the qubit: the scan's evidence reports it
    soc = _soc(dev, dict(tls_line_offsets_mhz=[-3.2], **NODE_TRUTH))
    outcome = CQED_EXPERIMENTS["tls_scan"](dev).go(soc, reps=REPS, progress=False)
    assert outcome.fit.derived["tls_lines"] >= 1.0
    assert outcome.fit.derived["tls_offset_mhz"] == pytest.approx(-3.2, abs=0.6)


@pytest.mark.qick
def test_rabi_node_gates_rabi_contrast_and_the_rabi_rate_prior(tmp_path):
    """The amplitude-Rabi node's gates: the fit plus the pack priors (rabi_contrast > 0.7 and
    the Rabi-rate window) — checked by the gate emitter against the fit and the calibrated
    π pulse."""
    from strumento.packs.cqed.bringup import rabi_gates

    dev = _device()
    soc = _soc(dev, dict(NODE_TRUTH))
    exp = CQED_EXPERIMENTS["amplitude_rabi"](dev)
    outcome = exp.go(soc, reps=REPS, progress=False)

    assert outcome.fit.success
    verdicts = rabi_gates(outcome, dev)
    assert verdicts["fit"] is True
    assert verdicts["rabi-contrast"] is True  # the fitted contrast > the pack's 0.7 prior
    assert verdicts["rabi-rate"] is True      # the implied π-pulse rate ∈ [0.1, 50] MHz


@pytest.mark.qick
def test_readout_length_gates_against_the_pack_kappa_window_prior(tmp_path):
    """The readout-length node's window gate consumes the PACK kappa prior ([1/κ] in µs) —
    the punchout's kappa wrote nothing, so no banked path feeds this gate. The sim run's
    fitted optimal integration sits inside the window and passes; an estimate outside the
    [0.1, 10] µs window fails the gate (the pack prior, checked by the gate emitter)."""
    from strumento.core.experiment import ExperimentOutcome

    dev = _device()
    soc = _soc(dev, dict(readout_optimal_length=3.0, **NODE_TRUTH))
    exp = CQED_EXPERIMENTS["readout_optimization"](dev)
    outcome = exp.go(soc, reps=REPS, progress=False)

    assert outcome.fit.success
    assert readout_length_gates(outcome, dev)["kappa-window"] is True

    # an optimal length outside the pack's [1/κ] window fails the gate
    class _SlowFit:
        success = True
        derived = {"optimal_length": 25.0}  # 25 µs: outside [1/10 MHz, 1/0.1 MHz] µs

    bad = ExperimentOutcome(result=outcome.result, fit=_SlowFit())
    assert readout_length_gates(bad, dev)["kappa-window"] is False
