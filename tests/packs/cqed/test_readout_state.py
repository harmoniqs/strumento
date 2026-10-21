"""Tests for ReadoutState (bring-up #56) -- had zero coverage before this audit fix
(2026-09-13): its documented mock/no-per-shot fallback path crashed with an
UnboundLocalError the moment it ran, unexercised by CI because SimulatorSoc's
registered response always provides per-shot data.

Pure: no qick needed.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.special import erfc

from strumento.core.result import MeasurementResult
from strumento.packs.cqed.experiments.readout_state import ReadoutState


def _shots_result(seed: int = 0, n: int = 200) -> MeasurementResult:
    """Per-shot data: two blobs (ground near (0,0), excited near (1,0))."""
    rng = np.random.default_rng(seed)
    g = rng.normal(loc=[0.0, 0.0], scale=0.1, size=(n, 2))
    e = rng.normal(loc=[1.0, 0.0], scale=0.1, size=(n, 2))
    shots = np.stack([g, e])  # (2 reads, n shots, 2 iq)
    avgi = shots[..., 0].mean(axis=1)
    avgq = shots[..., 1].mean(axis=1)
    return MeasurementResult(xpts=None, avgi=avgi, avgq=avgq, shots=shots)


def _shots_result_with_separation(
    seed: int, n: int, sigma: float, separation_sigma: float
) -> MeasurementResult:
    """Per-shot data (same two-Gaussian-blob shape as `_shots_result`): ground at (0, 0),
    excited at `separation_sigma * sigma` along I, each blob scattered by `sigma`.

    Because both blobs share `sigma` and sit `separation_sigma` sigma apart, discriminating
    at the exact midpoint has a KNOWN, closed-form misassignment probability -- the same
    tail integral `assignment_fidelity_from_separation` already uses for the fidelity figure:
    eps = P(Z > separation_sigma / 2) = 0.5 * erfc(separation_sigma / (2 * sqrt(2))).
    """
    rng = np.random.default_rng(seed)
    delta = separation_sigma * sigma
    g = rng.normal(loc=[0.0, 0.0], scale=sigma, size=(n, 2))
    e = rng.normal(loc=[delta, 0.0], scale=sigma, size=(n, 2))
    shots = np.stack([g, e])
    avgi = shots[..., 0].mean(axis=1)
    avgq = shots[..., 1].mean(axis=1)
    return MeasurementResult(xpts=None, avgi=avgi, avgq=avgq, shots=shots)


def _mock_result() -> MeasurementResult:
    """The documented mock/no-per-shot fallback: averaged reads only, shots=None."""
    return MeasurementResult(xpts=None, avgi=np.array([0.0, 1.0]), avgq=np.array([0.0, 0.0]),
                              shots=None)


class TestReadoutStateShotsPath:
    def test_analyze_recovers_the_blob_separation(self, dev):
        outcome = ReadoutState(dev).analyze(_shots_result())
        assert outcome.success
        assert outcome.derived["separation"] > 0
        assert outcome.derived["assignment_fidelity"] > 0.9  # well-separated blobs

    def test_discrimination_angle_points_from_ground_to_excited(self, dev):
        outcome = ReadoutState(dev).analyze(_shots_result())
        # ground at (0,0), excited at (1,0): the angle should be ~0 rad (along +I)
        assert outcome.derived["angle"] == pytest.approx(0.0, abs=0.2)

    def test_confusion_matrix_is_row_stochastic_and_near_identity_for_well_separated_blobs(
        self, dev
    ):
        # ground near (0,0), excited near (1,0), scale=0.1 -> blobs are 10 sigma apart, so
        # assignment at the fitted threshold is (for 200 shots/cloud) exactly clean: this is
        # the row-major [P(g|g), P(e|g), P(g|e), P(e|e)] layout check, not the ε-recovery case.
        outcome = ReadoutState(dev).analyze(_shots_result())
        p_gg, p_eg, p_ge, p_ee = outcome.derived["confusion_matrix"]
        assert len(outcome.derived["confusion_matrix"]) == 4
        assert p_gg + p_eg == pytest.approx(1.0)  # row 0: ground-prepped shots
        assert p_ge + p_ee == pytest.approx(1.0)  # row 1: excited-prepped shots
        assert p_gg == pytest.approx(1.0, abs=1e-9)
        assert p_ee == pytest.approx(1.0, abs=1e-9)

    def test_Ig_and_Ie_are_the_median_rotated_i_of_each_per_shot_cloud(self, dev):
        result = _shots_result()
        outcome = ReadoutState(dev).analyze(result)
        g_iq = np.asarray(result.shots)[0]
        e_iq = np.asarray(result.shots)[1]
        # self.readout.angle defaults to 0.0, so the rotated-I signal is exactly raw I here.
        assert outcome.derived["Ig"] == pytest.approx(float(np.median(g_iq[:, 0])))
        assert outcome.derived["Ie"] == pytest.approx(float(np.median(e_iq[:, 0])))

    def test_confusion_matrix_off_diagonal_recovers_a_constructed_assignment_error(self, dev):
        """A fit-recovery-style case (AC3): construct blobs a KNOWN `separation_sigma` apart
        so the assignment error at the midpoint threshold has a closed form (see
        `_shots_result_with_separation`'s docstring), then check the recovered off-diagonal
        confusion entries against it.

        Tolerance, sized from the error budget rather than a round number (the spirit of
        `tests/e2e/test_fit_recovery.py`'s module docstring): each off-diagonal entry is a
        Bernoulli fraction over `n` shots, so its sampling standard error is
        `sqrt(eps * (1 - eps) / n)`; 5x that (a comfortable, not-tight margin -- the fitted
        threshold carries its own small uncertainty on top of the sampling error, since the
        confusion matrix discriminates against the *fitted* threshold, not the true midpoint)
        is the tolerance actually asserted.
        """
        sigma = 0.1
        separation_sigma = 3.0  # moderate, non-trivial overlap -- not 10 sigma like the
        # near-identity case above, so eps is a real, measurable fraction rather than ~0
        n = 4000
        eps = 0.5 * float(erfc(separation_sigma / (2.0 * math.sqrt(2.0))))
        se = math.sqrt(eps * (1.0 - eps) / n)
        tol = 5.0 * se

        result = _shots_result_with_separation(
            seed=1, n=n, sigma=sigma, separation_sigma=separation_sigma
        )
        outcome = ReadoutState(dev).analyze(result)
        assert outcome.success
        p_gg, p_eg, p_ge, p_ee = outcome.derived["confusion_matrix"]

        assert p_eg == pytest.approx(eps, abs=tol), (
            f"P(e|g)={p_eg} vs closed-form eps={eps} (tol={tol})"
        )
        assert p_ge == pytest.approx(eps, abs=tol), (
            f"P(g|e)={p_ge} vs closed-form eps={eps} (tol={tol})"
        )
        # sanity: still row-stochastic on this harder (overlapping-blob) case too
        assert p_gg + p_eg == pytest.approx(1.0)
        assert p_ge + p_ee == pytest.approx(1.0)


class TestReadoutStateMockPath:
    """A result without per-shot data (the mock, an averaged board acquire).

    History: this path once crashed with UnboundLocalError (audit fix), then "fell back" to a
    double-Gaussian fit of the two averaged points — a histogram of two numbers, whose threshold,
    separation and assignment fidelity were shot quantities computed from averages. Since #199 it
    refuses: ``success=False`` with the reason, keeping only the discrimination angle (the blob
    centres ARE the averages, so the angle between them is honest) and naming the rest."""

    def test_mock_path_refuses_with_the_reason(self, dev):
        from strumento.core.experiment import SHOTS_REFUSAL

        outcome = ReadoutState(dev).analyze(_mock_result())
        assert outcome.success is False
        assert outcome.derived["refused"] == SHOTS_REFUSAL

    def test_mock_path_discrimination_angle_from_averaged_reads(self, dev):
        # ground read (0,0), excited read (1,0): same "points from ground to excited" as the
        # shots path, computed from the two averaged points instead of per-shot means.
        outcome = ReadoutState(dev).analyze(_mock_result())
        assert outcome.derived["angle"] == pytest.approx(0.0, abs=1e-6)

    def test_mock_path_reports_no_shot_quantity_and_names_every_one_it_withholds(self, dev):
        shot_only = {"threshold", "separation", "assignment_fidelity", "readout_contrast",
                     "confusion_matrix", "Ig", "Ie"}
        mock = ReadoutState(dev).analyze(_mock_result()).derived
        shots_keys = set(ReadoutState(dev).analyze(_shots_result()).derived)
        assert set(mock) == {"angle", "refused", "refused_parts"}
        assert set(mock["refused_parts"]) == shot_only
        assert shots_keys == shot_only | {"angle"}  # the shots path is unchanged
