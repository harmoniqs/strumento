"""DRAG pair calibration — the offset-cancelling DRAG diagnostic (cqed expansion).

For each DRAG coefficient α, two blocks are played back to back, each from a passive reset:

* **A** — X(π/2) then Y(π), then a dispersive measure;
* **B** — Y(π/2) then X(π), then a dispersive measure.

Ideally both leave the qubit on the equator (P_e = ½). The DRAG phase error enters the two with
**opposite** signs (3-level model, σ = 35 ns, Δ/2π = −143 MHz: dP/dα = +1.56 and −1.57), so the
difference ``D = P_A − P_B`` crosses zero at the optimum with ~4.6× the slope of the single X/2·Y/2
scan — and anything common to both blocks cancels: readout offset or scale, thermal population, T1
during the sequence, and π/2-amplitude error. The **pair mean** ``(P_A + P_B)/2 − ½`` measures that
common offset directly, so it is reported alongside ``alpha_opt``.

What it does NOT cancel: a qubit **detuning** still shifts the difference crossing (+0.013 per
+0.1 MHz in the model — about 6x smaller than its effect on X/2·Y/2). Calibrate f_ge first.

Why it exists: the single X/2·Y/2 crossing can sit ~0.05 from the first-order optimum when the
readout equator is offset; in the model a +0.03 equator offset shifts it by −0.044.

Reads: ``2 × points`` (A then B, interleaved per α — slow drift is common to the pair as well). The
pulses are the transmon factory's ``ge_hpi``/``ge_pi`` at ``drag_alpha = α`` (same builder, DAC-rate
sampled, 4σ); ``|α| > 1`` spends gain headroom (envelope / |α|, gain × |α|) and is validated against the
qubit-drive limit, exactly as :class:`~.drag_calibration.DRAGCalibration` does.

Pure at import: no ``qick``.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from strumento.core.experiment import Experiment
from strumento.core.fitters import FitResult
from strumento.core.pulses import Pulse, Seq
from strumento.core.result import MeasurementResult
from strumento.core.wiring import LineRef
from strumento.packs.cqed.experiments._common import (
    PassiveReset,
    check_alpha_bounds,
    drive_gain_limit,
    is_population_anchored,
)
from strumento.packs.cqed.experiments.drag_calibration import (
    DRAGCalibration,
    _drag_envelope,
    _fit_equator_crossing,
)
from strumento.packs.cqed.readout import DispersiveIQ
from strumento.packs.cqed.safety import validate_gain

#: (label, pulse key, phase °) for each block, in play order.
_BLOCK_A = (("drag_pair_x90", "hpi_ge", 0.0), ("drag_pair_y180", "pi_ge", 90.0))
_BLOCK_B = (("drag_pair_y90", "hpi_ge", 90.0), ("drag_pair_x180", "pi_ge", 0.0))


class DRAGPairCalibration(PassiveReset, Experiment):
    """DRAG coefficient from the X(π/2)·Y(π) vs Y(π/2)·X(π) difference — offsets cancel.

    ``alpha_start``/``alpha_stop``/``points`` define the α sweep (default ±0.25, 41 points — the
    difference is linear to ~5 % within |α| ≲ 0.12, saturates by ~0.3, and turns back beyond ~0.6).
    ``fit_halfwidth`` (default 0.12) is the half-width of the locally-linear window the zero crossing
    is fitted in. ``reset_delay_us`` overrides the passive reset (default: readout ``relax_delay``).
    """

    ALPHA_LIMIT = DRAGCalibration.ALPHA_LIMIT
    MIN_FIT_POINTS = DRAGCalibration.MIN_FIT_POINTS
    MIN_SLOPE_T = DRAGCalibration.MIN_SLOPE_T

    def __init__(self, device, *, alpha_start: float = -0.25, alpha_stop: float = 0.25,
                 points: int = 41, readout: DispersiveIQ | None = None,
                 reset_delay_us: float | None = None, fit_halfwidth: float = 0.12, runner=None):
        check_alpha_bounds(alpha_start, alpha_stop, limit=self.ALPHA_LIMIT)
        super().__init__(device, runner=runner)
        self.alpha_start = alpha_start
        self.alpha_stop = alpha_stop
        self.points = points
        self.readout = readout if readout is not None else DispersiveIQ.from_calib(device.calib.readout)
        self.reset_delay_us = reset_delay_us
        self.fit_halfwidth = float(fit_halfwidth)
        self._alphas = np.linspace(alpha_start, alpha_stop, points)

    def axes(self) -> tuple:
        return ()

    def sequence(self) -> tuple[Seq, tuple]:
        from strumento.packs.cqed.calibration import gain_int_to_frac

        q = self.device.qubit
        qc = self.device.calib.qubit
        line = LineRef(q.element.name, "drive")
        maxv = q._maxv(qc.f_ge)
        dac_rate = q._dac_rate(qc.f_ge)
        limit = drive_gain_limit(self.device)
        reset = self._reset_delay()
        base_gain = {key: gain_int_to_frac(qc.pulses[key].gain, maxv) for key in ("hpi_ge", "pi_ge")}
        sigma = {key: qc.pulses[key].sigma for key in ("hpi_ge", "pi_ge")}

        seq = Seq()
        for alpha in self._alphas:
            a = float(alpha)
            envs = {key: _drag_envelope(sigma[key], a, dac_rate=dac_rate) for key in sigma}
            for block in (_BLOCK_A, _BLOCK_B):
                # Every block from |g>: a measurement is not a reset (chaining once broke this).
                seq.wait(reset)
                for label, key, phase in block:
                    env, gain_scale = envs[key]
                    gain = validate_gain(base_gain[key] * gain_scale, limit=limit,
                                         role="qubit_drive", label=f"{label}(alpha={a:.3g})")
                    seq.play(Pulse(line=line, freq_mhz=qc.f_ge, gain=gain, phase_deg=phase,
                                   envelope=env, label=label))
                seq.play(self.device.readout.dispersive_measure())
        return seq, ()

    def analyze(self, result: MeasurementResult) -> FitResult:
        avgi = np.asarray(result.avgi, dtype=float).ravel()
        avgq = np.asarray(result.avgq, dtype=float).ravel()
        iq = np.stack([avgi, avgq], axis=-1)

        anchored = is_population_anchored(self.readout)
        if anchored:
            signal = np.asarray(self.readout.to_population(iq), dtype=float).ravel()
        else:
            signal = np.asarray(self.readout.project(iq), dtype=float).ravel()

        n = min(len(signal) // 2, len(self._alphas))
        x = np.asarray(self._alphas[:n], dtype=float)
        pa, pb = signal[0:2 * n:2], signal[1:2 * n:2]
        diff = pa - pb
        provenance = {
            "alphas": [float(v) for v in x],
            "signal_a": [float(v) for v in pa],
            "signal_b": [float(v) for v in pb],
            "difference": [float(v) for v in diff],
            "population_anchored": 1.0 if anchored else 0.0,
        }
        if not anchored:
            # The difference itself would survive an affine readout, but its error scale, the
            # equator offset, and the plot's P_e axis would not — refuse, as DRAGCalibration does.
            return FitResult(params={}, stderr={}, derived=provenance, success=False)

        fit = _fit_equator_crossing(x, diff, halfwidth=self.fit_halfwidth,
                                    min_points=self.MIN_FIT_POINTS,
                                    min_slope_t=self.MIN_SLOPE_T, level=0.0)
        derived: dict[str, Any] = {**provenance, **fit["diagnostics"]}

        # Common-mode offset over the same window the crossing used (all points if no window).
        hi: Any  # the crossing fit reports fit_alpha_max whenever it reports fit_alpha_min
        lo, hi = derived.get("fit_alpha_min"), derived.get("fit_alpha_max")
        sel = np.ones(n, dtype=bool) if lo is None else (x >= lo - 1e-12) & (x <= hi + 1e-12)
        mean = 0.5 * (pa[sel] + pb[sel]) - 0.5
        derived["equator_offset"] = float(np.mean(mean))
        derived["equator_offset_stderr"] = (
            float(np.std(mean, ddof=1) / np.sqrt(sel.sum())) if sel.sum() > 1 else float("nan")
        )

        if fit["alpha_opt"] is None:
            return FitResult(params=fit["params"], stderr=fit["stderr"], derived=derived,
                             success=False)
        derived.update({
            "alpha_opt": fit["alpha_opt"],
            "alpha_opt_stderr": fit["alpha_opt_stderr"],
            "drag_alpha": fit["alpha_opt"],  # calibration-path alias: calib.qubit.drag_alpha
        })
        return FitResult(params=fit["params"], stderr=fit["stderr"], derived=derived, success=True)
