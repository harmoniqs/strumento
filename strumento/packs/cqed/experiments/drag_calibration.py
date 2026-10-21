"""DRAG calibration — optimize the DRAG coefficient to minimize leakage (cqed expansion).

Sweeps the DRAG alpha parameter by building a set of Arb envelopes with different derivative
scaling. At each alpha value, a diagnostic X/2-Y/2 sequence is played using pulses that
include the DRAG derivative component (Q channel = alpha * dI/dt). The leakage-sensitive
error signal crosses zero at the optimal alpha.

The implementation uses the multi-read pattern (one read per alpha value, no loop axis):
each block plays ge_hpi with a DRAG-modified Arb envelope, and the accumulated leakage
is read out. This is hardware-faithful — each alpha produces a physically different waveform.

Pure at import: no ``qick`` (the Runner does the qick-facing acquire, lazily).
"""

from __future__ import annotations

import numpy as np

from strumento.core.experiment import Experiment
from strumento.core.fitters import FitResult
from strumento.core.pulses import Arb, Pulse, Seq
from strumento.core.result import MeasurementResult
from strumento.core.wiring import LineRef
from strumento.packs.cqed.experiments._common import (
    PassiveReset,
    check_alpha_bounds,
    drive_gain_limit,
    is_population_anchored,
)
from strumento.packs.cqed.library._drag import drag_arb_with_headroom
from strumento.packs.cqed.readout import DispersiveIQ
from strumento.packs.cqed.safety import validate_gain


def _drag_envelope(
    sigma_us: float, alpha: float, *, dac_rate: float | None = None
) -> tuple[Arb, float]:
    """Build the Gaussian + α·(dGaussian/dt) DRAG ``Arb`` the calibration plays at ``alpha``.

    Thin shim over the shared :func:`~strumento.packs.cqed.library._drag.drag_arb` builder, called
    EXACTLY as the transmon factory calls it (``TransmonOps._pulse``): the 4σ window sampled at the
    drive DAC rate. That makes the calibrated pulse the shipped gate pulse in time as well as in
    shape — the only way ``alpha_opt`` and the ``hpi_ge`` gain mean the same thing on
    both sides.

    This shim used to pin a fixed 64 samples. An ``Arb`` plays its samples
    back-to-back at the DAC rate, so on a 6.4 GS/s DAC that was a 10 ns pulse against the 140 ns
    (σ = 35 ns) gate — the ``hpi_ge`` gain rotated ~6°, and α (a Q/I peak ratio, optimum ∝ 1/σ) did
    not transfer to the gate.

    Returns ``(arb, gain_scale)``: for ``|alpha| > 1`` the envelope is divided by ``|alpha|`` and the
    caller multiplies the pulse gain by ``gain_scale`` — gain headroom instead of ``drag_arb``'s Q
    clamp, so the qubit sees exactly ``gain · (G + iα·dĜ)``. For ``|alpha| <= 1`` the Arb is the
    factory's ``drag_arb`` shape and ``gain_scale == 1``.

    Parameters
    ----------
    sigma_us : float
        Gaussian σ in µs.
    alpha : float
        DRAG coefficient — the Q-channel peak as a fraction of the I peak. alpha=0 is a pure
        Gaussian.
    dac_rate : float | None
        Drive DAC sample rate (samples/µs). ``None`` (no soccfg snapshot) falls back to the
        builder's default sample count, as the factory does.
    """
    return drag_arb_with_headroom(sigma_us, alpha, dac_rate=dac_rate)


class DRAGCalibration(PassiveReset, Experiment):
    """DRAG coefficient sweep to minimize leakage to the |f⟩ state.

    Builds a multi-read program with one X/2-Y/2 block per alpha value. Each block uses
    Arb envelopes with the DRAG derivative scaled by alpha. The error signal crosses zero
    at the optimal alpha — the analyzer finds this zero-crossing.

    ``alpha_start`` and ``alpha_stop`` define the sweep range (default -0.1 to 0.1); ``points``
    sets the number of alpha values tested. The default brackets the first-order optimum
    ``λ e^{-1/2} / (σ |Δ|)`` (λ ∈ [0.5, 1]) — about 0.01–0.02 for σ ≈ 30–35 ns and
    |Δ|/2π ≈ 140–180 MHz — with a 0.01 grid step. On hardware the X/2·Y/2 response is only resolved
    at |α| ~ 1, so Runners sweep wider (up to ``ALPHA_LIMIT``). ``|alpha| > 1`` is played faithfully
    through gain headroom (envelope / |α|, gain × |α|, validated against the qubit-drive limit),
    never through ``drag_arb``'s Q clamp.
    """

    #: Largest |α| the calibration will sweep. Above 1 the pulse spends gain headroom (gain × |α|).
    ALPHA_LIMIT = 2.0

    #: Crossing estimator (see :func:`_fit_equator_crossing`): fewest points in the fit window, and
    #: the minimum slope significance |m|/σ_m below which the scan is refused as unresolved.
    MIN_FIT_POINTS = 5
    MIN_SLOPE_T = 3.0

    def __init__(self, device, *, alpha_start: float = -0.1, alpha_stop: float = 0.1,
                 points: int = 21,
                 readout: DispersiveIQ | None = None, reset_delay_us: float | None = None,
                 fit_halfwidth: float = 0.25, runner=None):
        check_alpha_bounds(alpha_start, alpha_stop, limit=self.ALPHA_LIMIT)
        super().__init__(device, runner=runner)
        self.alpha_start = alpha_start
        self.alpha_stop = alpha_stop
        self.points = points
        self.readout = readout if readout is not None else DispersiveIQ.from_calib(device.calib.readout)
        self._alphas = np.linspace(alpha_start, alpha_stop, points)
        # Passive reset before every block (µs). Defaults to the calibrated readout relax_delay.
        self.reset_delay_us = reset_delay_us
        # Half-width (in α) of the locally-linear window the crossing is fitted in. ±0.25 is where the
        # ideal X/2·Y/2 response stays linear (P_e ≈ 0.34–0.66); the hardware S-curve saturates
        # beyond ~±0.6.
        self.fit_halfwidth = float(fit_halfwidth)

    def axes(self) -> tuple:
        return ()

    def sequence(self) -> tuple[Seq, tuple]:
        q = self.device.qubit
        qc = self.device.calib.qubit
        pc = qc.pulses["hpi_ge"]
        sigma = pc.sigma

        # Resolve gain, line, and DAC rate through the factory's own lookups so the calibration
        # plays exactly the pulse ``ge_hpi()`` will ship once ``drag_alpha`` is written.
        from strumento.packs.cqed.calibration import gain_int_to_frac
        line = LineRef(q.element.name, "drive")
        gain_frac = gain_int_to_frac(pc.gain, q._maxv(qc.f_ge))
        dac_rate = q._dac_rate(qc.f_ge)

        limit = drive_gain_limit(self.device)
        reset = self._reset_delay()

        seq = Seq()
        for alpha in self._alphas:
            # One envelope per alpha, shared by X/2 and Y/2 (they differ only by phase). Above
            # |α| = 1 the envelope is scaled down and the gain up by the same factor — the qubit sees
            # gain·(G + iα·dĜ) exactly — and that scaled gain faces the same safety limit the
            # factory enforces (this path builds its Pulse directly, so it must check it itself).
            env, gain_scale = _drag_envelope(sigma, float(alpha), dac_rate=dac_rate)
            gain = validate_gain(gain_frac * gain_scale, limit=limit, role="qubit_drive",
                                 label=f"drag_hpi(alpha={float(alpha):.3g})")
            pulse_x = Pulse(
                line=line, freq_mhz=qc.f_ge, gain=gain,
                phase_deg=0.0, envelope=env, label="drag_hpi_x",
            )
            pulse_y = Pulse(
                line=line, freq_mhz=qc.f_ge, gain=gain,
                phase_deg=90.0, envelope=env, label="drag_hpi_y",
            )
            # Reset to |g> before every block. A measurement is not a reset: without this, block k
            # starts from block k-1's post-measurement state (P_k - 1/2 = (p_k - 1/2)(1 - 2 P_{k-1})),
            # the chain locks at P_e = 0.5 and every later α reads 0.5 — a plateau for α >= 0
            # (same failure TransmonAllXY guards against).
            seq.wait(reset)
            seq.play(pulse_x)
            seq.play(pulse_y)
            seq.play(self.device.readout.dispersive_measure())

        return seq, ()

    def analyze(self, result: MeasurementResult) -> FitResult:
        avgi = np.asarray(result.avgi, dtype=float).ravel()
        avgq = np.asarray(result.avgq, dtype=float).ravel()
        iq = np.stack([avgi, avgq], axis=-1)

        # The zero-crossing logic below is anchored to the equator (population 0.5): the DRAG
        # X/2·Y/2 diagnostic tilts P_e linearly through 0.5 at the optimal alpha. So the signal MUST
        # be a population in [0, 1], which only the g/e blob centres can provide (``to_population``,
        # g→0, e→1). The bare projection is rotated-I in arbitrary ADC units — comparing it to 0.5 is
        # meaningless (an earlier version did exactly that). Without centres we REFUSE (a
        # fallback here once let the experiment contract "pass" on a wrong-sign fit).
        anchored = is_population_anchored(self.readout)
        if anchored:
            signal = np.asarray(self.readout.to_population(iq), dtype=float).ravel()
        else:
            signal = np.asarray(self.readout.project(iq), dtype=float).ravel()

        xpts = self._alphas
        n = min(len(signal), len(xpts))
        # Display provenance: the α axis and the signal the crossing was (or would be) searched on, so
        # the plot shows the SAME quantity the fit consumed. Emitted on EVERY outcome — a failed scan
        # is the one you most need to read against α. ``population_anchored`` (1.0/0.0) tells the
        # display whether ``signal`` is P_e or raw projected IQ.
        provenance = {
            "alphas": [float(a) for a in xpts[:n]],
            "signal": [float(s) for s in signal[:n]],
            "population_anchored": 1.0 if anchored else 0.0,
        }

        if not anchored:
            # No population, hence no equator: no alpha_opt at all, not even a placeholder, so nothing
            # downstream (fit gate, summary, writeback) can mistake it for a measurement. Record
            # ``g_center_iq``/``e_center_iq`` (ge-readout-blobs) and pass a blob-anchored readout.
            return FitResult(params={}, stderr={}, derived=provenance, success=False)

        fit = _fit_equator_crossing(
            np.asarray(xpts[:n], dtype=float), signal[:n],
            halfwidth=self.fit_halfwidth, min_points=self.MIN_FIT_POINTS,
            min_slope_t=self.MIN_SLOPE_T,
        )
        derived = {**provenance, **fit["diagnostics"]}
        if fit["alpha_opt"] is None:
            # Refused: too few points, an insignificant slope (a flat/noise scan), or a crossing that
            # falls outside the fitted data (extrapolation). No alpha_opt at all — nothing downstream
            # can mistake a refusal for a measurement.
            return FitResult(params=fit["params"], stderr=fit["stderr"], derived=derived,
                             success=False)
        derived.update({
            "alpha_opt": fit["alpha_opt"],
            "alpha_opt_stderr": fit["alpha_opt_stderr"],
            "drag_alpha": fit["alpha_opt"],  # calibration-path alias: calib.qubit.drag_alpha
        })
        return FitResult(params=fit["params"], stderr=fit["stderr"], derived=derived, success=True)


def _fit_equator_crossing(x, y, *, halfwidth, min_points, min_slope_t, level=0.5, max_iter=8):
    """Where a locally-linear signal crosses ``level`` (½ for P_e), with a 1σ uncertainty — or a refusal.

    Shared by :class:`DRAGCalibration` (P_e through ½) and ``DRAGPairCalibration`` (the A − B
    difference through 0).

    Replaces "first sign change of the noisy signal, scanning from the left", which on a shallow
    noisy slope lands on whichever early point dips (biased by several hundredths in α,
    success=True even on pure noise, no error bar).

    1. **Seed** at the smoothed (5-point running mean) sign change nearest α = 0 — the DRAG crossing
       sits near zero, and the ideal X/2·Y/2 response re-crosses ½ near |α| ≈ 1.5 on wide scans.
       With no smoothed sign change, seed at the point closest to ½ (nearest 0 on ties).
    2. **Fit** an ordinary least-squares line to the points within ``halfwidth`` of the current
       centre; the crossing is ``α₀ = (½ − b)/m``. Re-centre on α₀ and refit until it settles.
    3. **Uncertainty**: residual-scaled covariance of (m, b) propagated to α₀ (delta method).
    4. **Refuse** when fewer than ``min_points`` fall in the window, when ``|m| / σ_m <
       min_slope_t`` (no resolved slope), or when α₀ lies outside the fitted points' α range.
    """
    out = {"alpha_opt": None, "alpha_opt_stderr": None, "params": {}, "stderr": {},
           "diagnostics": {}}
    if len(x) < min_points:
        out["diagnostics"] = {"n_fit": float(len(x))}
        return out

    k = min(5, len(y))
    smooth = np.convolve(y, np.ones(k) / k, mode="same")
    edge = k // 2  # 'same' mode under-weights the ends; don't seed on them
    s = smooth - level
    valid = np.arange(len(x))
    if len(x) > 2 * edge:
        valid = valid[edge:len(x) - edge]
    flips = [i for i in valid[:-1] if np.sign(s[i]) != np.sign(s[i + 1])]
    if flips:
        centre = float(min((0.5 * (x[i] + x[i + 1]) for i in flips), key=abs))
    else:
        i = min(valid, key=lambda j: (abs(s[j]), abs(x[j])))
        centre = float(x[i])

    m = b = None
    cov = None
    sel = None
    for _ in range(max_iter):
        sel = np.abs(x - centre) <= halfwidth + 1e-12
        if sel.sum() < min_points:
            break
        xx, yy = x[sel], y[sel]
        A = np.column_stack([xx, np.ones_like(xx)])
        coef, *_ = np.linalg.lstsq(A, yy, rcond=None)
        m, b = float(coef[0]), float(coef[1])
        resid = yy - A @ coef
        dof = max(len(xx) - 2, 1)
        cov = (float(resid @ resid) / dof) * np.linalg.inv(A.T @ A)
        if m == 0.0:
            break
        new_centre = (level - b) / m
        if not np.isfinite(new_centre) or abs(new_centre - centre) < 1e-6:
            break
        if not (x.min() - halfwidth <= new_centre <= x.max() + halfwidth):
            break  # running off the scan: let the bounds check below refuse it
        centre = float(new_centre)

    if m is None or sel is None or sel.sum() < min_points:
        out["diagnostics"] = {"n_fit": float(0 if sel is None else sel.sum())}
        return out

    sm, sb = float(np.sqrt(cov[0, 0])), float(np.sqrt(cov[1, 1]))
    t = abs(m) / sm if sm > 0 else (np.inf if m != 0 else 0.0)
    xs = x[sel]
    out["params"] = {"slope": m, "intercept": b}
    out["stderr"] = {"slope": sm, "intercept": sb}
    out["diagnostics"] = {
        "slope": m, "slope_stderr": sm, "intercept": b,
        "fit_alpha_min": float(xs.min()), "fit_alpha_max": float(xs.max()),
        "n_fit": float(sel.sum()),
    }
    if not t >= min_slope_t or m == 0.0:
        return out
    a0 = (level - b) / m
    if not (xs.min() <= a0 <= xs.max()):
        return out
    jac = np.array([-(level - b) / m**2, -1.0 / m])
    se = float(np.sqrt(max(jac @ cov @ jac, 0.0)))
    out["alpha_opt"], out["alpha_opt_stderr"] = float(a0), se
    return out
