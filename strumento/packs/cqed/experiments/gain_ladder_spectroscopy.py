"""Gain-ladder qubit spectroscopy — the bring-up graph's qubit-spec node (#56).

One acquisition, one declared frequency axis, one **read per ladder rung**: the sequence
plays a spectroscopy probe at each rung's fixed gain (~6 dB apart) while the frequency
sweeps, so read *r* of the averaged acquisition is rung *r*'s spectrum. Per rung the vetted
:func:`fit_lorentzian` extracts the line's center, width and height; across rungs the
two-level frame fitter fits the shift and broadening **as power laws**:

- the AC-Stark shift ``Δ(s) = a·s^p_delta`` with ``s`` the relative drive beyond the lowest
  rung — a genuine qubit line has ``p_delta ≈ 1``;
- the power broadening ``Γ(s) = Γ0·(1+s)^p_gamma`` — a genuine line has ``p_gamma ≈ 0.5``.

Impostor signatures fail the frame: exponents off the two-level model, or **premature
saturation** (the line height collapsing at rungs where the fitted model predicts growth).
The lowest rung alone carries a **0.5× centroid-shift cap**: a weak-drive line sits within
half its fitted width of the scan's expected center — a shifted impostor does not. The
fitted zero-drive centroid is the node's ``f_spec`` (the set-writer of ``qubit.f_ge``).

Pure at import: no ``qick``.
"""

from __future__ import annotations

import numpy as np

from strumento.core.experiment import Experiment
from strumento.core.fitters import FitResult
from strumento.core.pulses import Const, Pulse, Seq
from strumento.core.resonator_fitters import fit_lorentzian
from strumento.core.result import MeasurementResult
from strumento.core.sweeps import Loop, Sweep
from strumento.core.wiring import LineRef
from strumento.packs.cqed.experiments._common import drive_gain_limit
from strumento.packs.cqed.library._gain import maxv_for
from strumento.packs.cqed.readout import DispersiveIQ
from strumento.packs.cqed.safety import validate_gain

#: The default gain ladder: 4-5 rungs, ~6 dB (a factor 2 in amplitude) apart.
DEFAULT_GAIN_LADDER: tuple[float, ...] = (1000.0, 2000.0, 4000.0, 8000.0, 16000.0)

#: A rung's fitted line height may not collapse below this fraction of the previous rung's:
#: the two-level model predicts growth (or at worst a plateau) — a collapse is the
#: premature-saturation impostor signature.
SATURATION_COLLAPSE_FRACTION: float = 0.5

#: The lowest rung's fitted centroid may not sit further than this fraction of its fitted
#: FWHM from the scan's expected center (the cap is on the lowest rung only — higher rungs
#: Stark-shift by design).
CENTROID_CAP_FRACTION: float = 0.5


def fit_gain_ladder_frame(
    gains, centers, widths, heights, *, expected_center: float
) -> FitResult:
    """Fit the two-level frame across the ladder's per-rung Lorentzian fits.

    ``s`` is the relative drive beyond the lowest rung (``g/g0 - 1``). The Stark shift is
    fit as a power law in ``s`` (log-log, rungs ≥ 2 — the lowest rung's shift is zero by
    construction) and the width as a power law in ``1+s`` (all rungs). Returns the fitted
    exponents, the zero-drive centroid (``f_spec``), the premature-saturation flag and the
    lowest-rung centroid-cap verdict — the frame check's inputs, as data.
    """
    gains = np.asarray(gains, dtype=float)
    centers = np.asarray(centers, dtype=float)
    widths = np.asarray(widths, dtype=float)
    heights = np.asarray(heights, dtype=float)
    s = gains / gains[0] - 1.0

    # the Stark exponent: log(Δ) = log(a) + p_delta·log(s) over the shifted rungs
    shifted = s > 0
    deltas = centers - centers[0]
    use = shifted & (deltas > 0)
    if int(use.sum()) >= 2:
        slope, _intercept = np.polyfit(np.log(s[use]), np.log(deltas[use]), 1)
        p_delta = float(slope)
        stark_slope = float(np.exp(_intercept))
    else:
        p_delta, stark_slope = float("nan"), float("nan")

    # the broadening exponent: log(Γ) = log(Γ0) + p_gamma·log(1+s) over every rung
    ok = widths > 0
    if int(ok.sum()) >= 2:
        slope_g, _i = np.polyfit(np.log(1.0 + s[ok]), np.log(widths[ok]), 1)
        p_gamma = float(slope_g)
    else:
        p_gamma = float("nan")

    # premature saturation: the height collapsing where the model predicts growth
    collapsed = any(
        h < SATURATION_COLLAPSE_FRACTION * h_prev for h_prev, h in zip(heights, heights[1:])
    )
    # the 0.5× centroid-shift cap, on the lowest rung only
    cap_ok = abs(float(centers[0]) - float(expected_center)) <= CENTROID_CAP_FRACTION * float(
        widths[0]
    )

    derived = {
        "f_spec": float(centers[0]),
        "p_delta": p_delta,
        "p_gamma": p_gamma,
        "stark_slope_mhz": stark_slope,
        "saturated": 1.0 if collapsed else 0.0,
        "centroid_cap_ok": 1.0 if cap_ok else 0.0,
    }
    for r, (c, w, h) in enumerate(zip(centers, widths, heights)):
        derived[f"rung{r}_center_mhz"] = float(c)
        derived[f"rung{r}_fwhm_mhz"] = float(w)
        derived[f"rung{r}_height"] = float(h)
    return FitResult(params={}, stderr={}, derived=derived, success=True)


class GainLadderSpectroscopy(Experiment):
    """Two-tone spectroscopy on a gain ladder — one read per rung, one frequency axis.

    ``gains`` is the ladder (lab-native int codes, ~6 dB per rung); ``center_mhz``/``span_mhz``
    tile the scanned band around the expected ``f_ge`` (resonator avoidance: the band lives on
    the qubit's GHz-scale line, far from the readout resonator). The analyzer fits each rung's
    line, then the two-level frame across the ladder (see :func:`fit_gain_ladder_frame`).
    """

    def __init__(self, device, *, center_mhz: float | None = None, span_mhz: float = 25.0,
                 gains=None, points: int = 81, readout: DispersiveIQ | None = None, runner=None):
        super().__init__(device, runner=runner)
        qc = device.calib.qubit
        self.center_mhz = center_mhz if center_mhz is not None else float(qc.f_ge)
        self.span_mhz = span_mhz
        self.gains = tuple(float(g) for g in (gains if gains is not None else DEFAULT_GAIN_LADDER))
        self.points = points
        self.readout = readout if readout is not None else DispersiveIQ.from_calib(device.calib.readout)

    def axes(self) -> tuple:
        return (Loop("freq", self.points),)

    def recovery_map(self) -> dict[str, str]:
        """The frame's fitted quantities pair with the armed truth's two-level model keys
        for the vetted emitter's fit-recovery comparison."""
        return {
            "f_spec": "f_ge",
            "p_delta": "gain_ladder_p_delta",
            "p_gamma": "gain_ladder_p_gamma",
            "stark_slope_mhz": "gain_ladder_stark_mhz",
        }

    def sequence(self) -> tuple[Seq, tuple]:
        qubit_name = self.device.qubit.element.name
        line = LineRef(qubit_name, "drive")
        freqs = Sweep(
            start=self.center_mhz - self.span_mhz, stop=self.center_mhz + self.span_mhz, on="freq"
        )
        maxv = maxv_for(self.device, line, self.center_mhz)
        seq = Seq()
        for gain_code in self.gains:
            gain_frac = gain_code / maxv
            validate_gain(gain_frac, limit=drive_gain_limit(self.device), role="qubit_drive",
                          label="ladder_probe")
            probe = Pulse(
                line=line, freq_mhz=freqs, gain=gain_frac,
                envelope=Const(length_us=10.0), label="ladder_probe",
            )
            # `probe` above drives the QUBIT line at this rung's gain, never the resonator
            # (#103) -- each rung still needs its own readout-line drive before its Measure,
            # one dispersive_measure() per rung, alongside the existing ladder tone.
            seq.play(probe)
            seq += self.device.readout.dispersive_measure()
        return seq, ()

    def analyze(self, result: MeasurementResult) -> FitResult:
        freqs = np.asarray(result.xpts, dtype=float)
        avgi = np.asarray(result.avgi, dtype=float)  # (n_rungs, n_points): read r = rung r
        avgq = np.asarray(result.avgq, dtype=float)

        centers, widths, heights, run_success = [], [], [], []
        for r in range(len(self.gains)):
            sig = self.readout.project(np.stack([avgi[r], avgq[r]], axis=-1))
            fit = fit_lorentzian(freqs, np.asarray(sig, dtype=float))
            run_success.append(bool(fit.success))
            centers.append(float(fit.derived["f0"]))
            widths.append(float(fit.derived["fwhm"]))
            heights.append(abs(float(fit.params["amp"])))

        frame = fit_gain_ladder_frame(
            self.gains, centers, widths, heights, expected_center=self.center_mhz
        )
        success = bool(frame.success) and all(run_success) and not np.isnan(
            frame.derived["p_delta"]
        )
        return FitResult(
            params=frame.params,
            stderr=frame.stderr,
            derived=frame.derived,
            success=success,
        )
