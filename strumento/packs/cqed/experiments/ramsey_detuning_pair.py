"""Ramsey detuning pair — the SIGNED qubit-drive detuning, from two artificial detunings.

A single Ramsey reads only the fringe frequency ``|Δa − δ|`` (``Δa`` the artificial detuning, ``δ``
the qubit's offset from the believed ``f_ge``), so the sign of the correction is ambiguous: a
70 kHz fringe at Δa = 50 kHz is either δ = −20 kHz or δ = +120 kHz. Running the same Ramsey
at **+Δa and −Δa** resolves it:

    f₊ = |Δa − δ|        f₋ = |Δa + δ|

* ``|δ| < Δa`` (the operating branch): ``δ = (f₋ − f₊)/2`` and ``(f₊ + f₋)/2 = Δa`` — the second
  identity is a free **consistency check** against the known Δa;
* ``|δ| > Δa``: ``(f₊ + f₋)/2 = |δ|`` and ``|f₋ − f₊|/2 = Δa``, sign from which fringe is faster.

The analyzer tries every branch and accepts the one whose consistency residual is within tolerance;
none consistent → ``success=False`` (never a guessed sign).

**Sign convention:** ``qubit_detuning_mhz = f_q − f_ge(belief)``, so the corrected frequency is
``f_ge + qubit_detuning_mhz`` (derived ``f_ge_corrected``). Drives are at ``f_ge ± Δa``.

**Sequence:** for each delay ``t`` two blocks, each from a passive reset (a measurement is not a
reset — chained blocks once broke this):
``ge_hpi(f_ge+Δa) · wait(t) · ge_hpi(f_ge+Δa) · measure``, then
the same at ``f_ge−Δa``. Interleaved per delay, so slow drift is common to the pair. Delays are
unrolled (no loop axis): ``2 × points`` reads per shot, like :class:`DRAGPairCalibration`.

The analyzer fits the projected signal (frequencies and T2* are invariant under an affine readout),
so no g/e blob centres are required.

Pure at import: no ``qick``.
"""

from __future__ import annotations

import numpy as np

from strumento.core.experiment import Experiment
from strumento.core.fitters import FitResult, fit_decaysin_detuned
from strumento.core.pulses import Seq
from strumento.core.result import MeasurementResult
from strumento.packs.cqed.experiments._common import PassiveReset
from strumento.packs.cqed.readout import DispersiveIQ
from strumento.packs.cqed.safety import MAX_TIME_SWEEP, MIN_TIME_SWEEP


def resolve_detuning(f_plus: float, f_minus: float, delta_a: float, tol: float) -> dict:
    """Signed ``δ = f_q − f_ge`` from the two fringe frequencies (all MHz).

    Returns ``{"delta", "branch", "residual"}`` for the branch whose consistency residual is smallest
    and within ``tol``, else ``{"delta": None, ...}``. Branches (``Δa > 0``):

    * ``inner``  (|δ| < Δa): δ = (f₋ − f₊)/2, check (f₊ + f₋)/2 = Δa;
    * ``outer+`` (δ > Δa):   δ = (f₊ + f₋)/2, check (f₋ − f₊)/2 = Δa;
    * ``outer-`` (δ < −Δa):  δ = −(f₊ + f₋)/2, check (f₊ − f₋)/2 = Δa.
    """
    fp, fm, da = abs(float(f_plus)), abs(float(f_minus)), abs(float(delta_a))
    s, d = 0.5 * (fp + fm), 0.5 * (fm - fp)
    candidates = (
        ("inner", d, abs(s - da)),
        ("outer+", s, abs(d - da)),
        ("outer-", -s, abs(-d - da)),
    )
    branch, delta, residual = min(candidates, key=lambda c: c[2])
    if residual > tol:
        return {"delta": None, "branch": None, "residual": float(residual)}
    return {"delta": float(delta), "branch": branch, "residual": float(residual)}


class RamseyDetuningPair(PassiveReset, Experiment):
    """Signed qubit detuning (and T2*) from Ramsey fringes at ``f_ge + Δa`` and ``f_ge − Δa``.

    ``stop_us``/``points`` define the delay grid ``linspace(0, stop_us, points)`` (the fringe at
    ``Δa + |δ|`` must sit below its Nyquist limit ``(points−1)/(2·stop_us)`` — validated).
    ``ramsey_freq_mhz`` is Δa (> 0). ``reset_delay_us`` overrides the passive reset (default: the
    readout ``relax_delay``). ``consistency_tol_mhz`` bounds the branch-consistency residual
    (default: ``max(0.1·Δa, 3σ)`` with σ from the two fit stderrs).
    """

    def __init__(self, device, *, stop_us: float = 300.0, points: int = 76,
                 ramsey_freq_mhz: float = 0.05, readout: DispersiveIQ | None = None,
                 reset_delay_us: float | None = None, consistency_tol_mhz: float | None = None,
                 runner=None):
        if not (0.0 < float(ramsey_freq_mhz)):
            raise ValueError(f"ramsey_freq_mhz={ramsey_freq_mhz}: the artificial detuning must be > 0")
        if not (max(MIN_TIME_SWEEP, 0.0) < float(stop_us) <= MAX_TIME_SWEEP):
            raise ValueError(f"stop_us={stop_us}: outside the delay limits (0, {MAX_TIME_SWEEP}]")
        if int(points) < 8:
            raise ValueError(f"points={points}: a fringe fit needs at least 8 delays per sign")
        nyquist = (int(points) - 1) / (2.0 * float(stop_us))
        if float(ramsey_freq_mhz) >= nyquist:
            raise ValueError(
                f"ramsey_freq_mhz={ramsey_freq_mhz} is at/above the delay-grid Nyquist {nyquist:.4g} MHz"
            )
        super().__init__(device, runner=runner)
        self.stop_us = float(stop_us)
        self.points = int(points)
        self.ramsey_freq_mhz = float(ramsey_freq_mhz)
        self.readout = readout if readout is not None else DispersiveIQ.from_calib(device.calib.readout)
        self.reset_delay_us = reset_delay_us
        self.consistency_tol_mhz = consistency_tol_mhz
        self._delays = np.linspace(0.0, self.stop_us, self.points)

    @property
    def nyquist_mhz(self) -> float:
        return (self.points - 1) / (2.0 * self.stop_us)

    def axes(self) -> tuple:
        return ()

    def sequence(self) -> tuple[Seq, tuple]:
        q = self.device.qubit
        f_ge = float(self.device.calib.qubit.f_ge)
        reset = self._reset_delay()
        seq = Seq()
        for t in self._delays:
            for sign in (+1.0, -1.0):
                drive = f_ge + sign * self.ramsey_freq_mhz
                seq.wait(reset)                       # every block from |g>
                seq.play(q.ge_hpi(freq=drive))
                if t > 0.0:
                    seq.wait(float(t))
                seq.play(q.ge_hpi(freq=drive))
                seq.play(self.device.readout.dispersive_measure())
        return seq, ()

    def analyze(self, result: MeasurementResult) -> FitResult:
        avgi = np.asarray(result.avgi, dtype=float).ravel()
        avgq = np.asarray(result.avgq, dtype=float).ravel()
        signal = np.asarray(self.readout.project(np.stack([avgi, avgq], axis=-1)), dtype=float).ravel()
        n = min(len(signal) // 2, self.points)
        t = self._delays[:n]
        s_plus, s_minus = signal[0:2 * n:2], signal[1:2 * n:2]
        fit_p = fit_decaysin_detuned(t, s_plus)
        fit_m = fit_decaysin_detuned(t, s_minus)
        f_p, f_m = float(fit_p.derived["freq"]), float(fit_m.derived["freq"])
        sig_p = float(fit_p.stderr.get("freq", np.nan))
        sig_m = float(fit_m.stderr.get("freq", np.nan))
        sigma = 0.5 * float(np.hypot(sig_p, sig_m)) if np.isfinite(sig_p) and np.isfinite(sig_m) else np.nan
        tol = self.consistency_tol_mhz
        if tol is None:
            tol = max(0.1 * self.ramsey_freq_mhz, 3.0 * sigma if np.isfinite(sigma) else 0.0)
        branch = resolve_detuning(f_p, f_m, self.ramsey_freq_mhz, tol)

        derived = {
            "delays": [float(v) for v in t],
            "signal_plus": [float(v) for v in s_plus],
            "signal_minus": [float(v) for v in s_minus],
            "fringe_plus_mhz": f_p,
            "fringe_minus_mhz": f_m,
            "fringe_plus_stderr": sig_p,
            "fringe_minus_stderr": sig_m,
            "ramsey_freq_mhz": self.ramsey_freq_mhz,
            "consistency_residual_mhz": branch["residual"],
            "consistency_tol_mhz": float(tol),
            "T2_plus": float(fit_p.derived["T2"]),
            "T2_minus": float(fit_m.derived["T2"]),
            "T2": 0.5 * (float(fit_p.derived["T2"]) + float(fit_m.derived["T2"])),
            "nyquist_mhz": self.nyquist_mhz,
        }
        ok = bool(fit_p.success and fit_m.success and branch["delta"] is not None
                  and max(f_p, f_m) < self.nyquist_mhz)
        if branch["delta"] is not None:
            delta = branch["delta"]
            derived.update({
                "branch": branch["branch"],
                "qubit_detuning_mhz": delta,            # f_q − f_ge(belief)
                "qubit_detuning_stderr": sigma,
                "f_ge_corrected": float(self.device.calib.qubit.f_ge) + delta,
            })
        params = {"fringe_plus": f_p, "fringe_minus": f_m}
        stderr = {"fringe_plus": sig_p, "fringe_minus": sig_m}
        return FitResult(params=params, stderr=stderr, derived=derived, success=ok)
