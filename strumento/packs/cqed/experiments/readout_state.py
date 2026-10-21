"""Readout state — threshold/angle calibration with a vetted fitter (bring-up #56).

Two preparations in one shot (ground, then a ``ge_pi``-prepared excited state), one read
each: the per-shot projected signal forms a bimodal histogram whose vetted double-Gaussian
fit (:func:`fit_double_gaussian`) gives the discrimination threshold, the blob separation,
and — from the separation in σ units — the assignment fidelity the node's gate checks against
the pack threshold (the SAME threshold object the terminal validation references). The
discrimination angle comes from the two blob centers in the IQ plane.

Pure at import: no ``qick``.
"""

from __future__ import annotations

import numpy as np
from scipy.special import erf

from strumento.core.experiment import Experiment, shots_refusal
from strumento.core.fitters import DerivedValue, FitResult, fit_double_gaussian
from strumento.core.pulses import Seq
from strumento.core.result import MeasurementResult
from strumento.packs.cqed.readout import DispersiveIQ


def assignment_fidelity_from_separation(separation: float) -> float:
    """The two-Gaussian assignment fidelity for blobs separated by ``separation`` σ units
    (equal weights, the threshold at the midpoint): ``F = erf(separation / (2√2))``."""
    return float(erf(abs(separation) / (2.0 * np.sqrt(2.0))))


def _centre_angle(g_center: tuple[float, float], e_center: tuple[float, float]) -> float:
    """The discrimination angle from the blob centres (the IQ rotation maximizing separation)."""
    return float(np.arctan2(e_center[1] - g_center[1], e_center[0] - g_center[0]))


class ReadoutState(Experiment):
    """Two-preparation readout characterization — the readout-state bring-up node.

    The analyzer bins the per-shot projected signal of both preparations and fits the
    histogram with the vetted double-Gaussian fitter; the derived quantities are the
    discrimination ``threshold``, the blob ``angle`` (radians), the ``separation`` (σ units)
    and the ``assignment_fidelity``. On a soc **with** per-shot data it additionally reports
    the ``confusion_matrix`` (row-major ``[P(g|g), P(e|g), P(g|e), P(e|e)]``, each row summing
    to 1 — discriminating the per-shot clouds against the fitted threshold) and the ``Ig``/
    ``Ie`` blob-centre medians.

    It needs per-shot data (``needs_shots``, #199): every quantity above but the angle comes from
    the per-shot histogram, and a histogram of two averaged points is not one. A result without
    shots (the mock, an averaged board acquire) is refused — ``success=False``,
    ``derived["refused"]`` — keeping only ``angle`` (the blob centres *are* the averages, so the
    angle between them is honest) and naming every withheld quantity in
    ``derived["refused_parts"]``.
    """

    needs_shots = True

    #: what a result without shots cannot give: everything the per-shot histogram fit yields
    SHOT_QUANTITIES = ("threshold", "separation", "assignment_fidelity", "readout_contrast",
                       "confusion_matrix", "Ig", "Ie")

    def __init__(self, device, *, readout: DispersiveIQ | None = None, runner=None):
        super().__init__(device, runner=runner)
        self.readout = readout if readout is not None else DispersiveIQ.from_calib(device.calib.readout)

    def axes(self) -> tuple:
        return ()

    def sequence(self) -> tuple[Seq, tuple]:
        q = self.device.qubit
        seq = Seq().play(self.device.readout.dispersive_measure())  # read 0: the ground state
        seq.play(q.ge_pi())                                          # prepare |e>
        seq.play(self.device.readout.dispersive_measure())           # read 1: the excited state
        return seq, ()

    def analyze(self, result: MeasurementResult) -> FitResult:
        avgi = np.asarray(result.avgi, dtype=float).ravel()
        avgq = np.asarray(result.avgq, dtype=float).ravel()
        shots = result.shots

        if shots is None:
            # #199: an averaged acquire (the mock, a board without a shot path) has no per-shot
            # cloud. The averaged reads ARE the blob centres, so the angle between them is honest;
            # the histogram quantities are not, and are refused rather than fit to two points.
            angle = _centre_angle((float(avgi[0]), float(avgq[0])),
                                  (float(avgi[-1]), float(avgq[-1])))
            return shots_refusal(kept={"angle": angle}, withheld=self.SHOT_QUANTITIES)

        g_iq = np.asarray(shots)[0]  # (shots, 2): the ground preparation
        e_iq = np.asarray(shots)[1]  # (shots, 2): the excited preparation
        g_proj = self.readout.project(g_iq).ravel()
        e_proj = self.readout.project(e_iq).ravel()
        g_center = (float(g_iq[..., 0].mean()), float(g_iq[..., 1].mean()))
        e_center = (float(e_iq[..., 0].mean()), float(e_iq[..., 1].mean()))
        angle = _centre_angle(g_center, e_center)

        combined = np.concatenate([g_proj, e_proj])
        lo, hi = float(combined.min()), float(combined.max())
        pad = 1e-9 if hi <= lo else (hi - lo) * 1e-3
        bins = np.linspace(lo - pad, hi + pad, 41)
        counts, _edges = np.histogram(combined, bins=bins)
        fit = fit_double_gaussian(0.5 * (bins[:-1] + bins[1:]), counts.astype(float))

        separation = float(fit.derived["separation"])
        threshold = float(fit.derived["threshold"])
        blob_separation = abs(float(fit.derived["mu_high"]) - float(fit.derived["mu_low"]))
        fidelity = assignment_fidelity_from_separation(separation)
        derived: dict[str, DerivedValue] = {
            "threshold": threshold,
            "angle": angle,
            "separation": separation,
            "assignment_fidelity": fidelity,
            # the projected blob separation, in signal units — pairs with the armed
            # truth's readout contrast through the vetted emitter's recovery comparison
            "readout_contrast": blob_separation,
        }

        # Confusion matrix + blob centres from the per-shot clouds (#80).
        # Discriminate g_proj/e_proj (already projected at `self.readout.angle`, the SAME
        # frame `combined` was fit on) against the just-fitted `threshold` — mirroring
        # `DispersiveIQ.discriminate`'s own `rotated_i > threshold -> assigned e` rule,
        # applied to signal already in the right rotated frame. Building a fresh
        # DispersiveIQ(angle=angle) and calling .discriminate() on raw IQ here would rotate
        # by the *centres* angle instead — a different frame than the one `threshold` was
        # fit in — and silently decorrelate the confusion matrix from its own fit.
        p_e_given_g = float(np.mean(g_proj > threshold))
        p_e_given_e = float(np.mean(e_proj > threshold))
        derived["confusion_matrix"] = [
            1.0 - p_e_given_g,  # P(g|g)
            p_e_given_g,        # P(e|g)
            1.0 - p_e_given_e,  # P(g|e)
            p_e_given_e,        # P(e|e)
        ]
        derived["Ig"] = float(np.median(g_proj))
        derived["Ie"] = float(np.median(e_proj))

        return FitResult(
            params=dict(fit.params),
            stderr=dict(fit.stderr),
            derived=derived,
            success=bool(fit.success),
        )
