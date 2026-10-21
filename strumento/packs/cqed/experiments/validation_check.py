"""Terminal bring-up validation — one thin Seq run, the campaign's last gate (#56).

A single acquisition carries both measured checks the terminal requires:

- reads 0/1 are the two readout preparations (ground, then a ``ge_pi``), whose per-shot
  projected signal yields the assignment fidelity the ``readout-fidelity`` gate checks
  against the pack threshold (per-shot data is required: a result without it is refused,
  #199);
- read 2 is a short Ramsey fringe (half-pi, swept idle, half-pi) whose vetted
  :func:`fit_decaysin_detuned` gives the detuning the ``detuning-window`` gate checks
  against the 50 kHz window — and T2 from the envelope.

The "T1 AND T2 recorded" half of the terminal is the graph's structure, not this
experiment's: the validation step *requires* ``qubit.T1``/``qubit.T2``, so with a store wired
the freshness engine detours to the providers until both are recorded.

Pure at import: no ``qick``.
"""

from __future__ import annotations

import numpy as np

from strumento.core.experiment import Experiment, shots_refusal
from strumento.core.fitters import FitResult, fit_decaysin_detuned, fit_double_gaussian
from strumento.core.pulses import Seq
from strumento.core.result import MeasurementResult
from strumento.core.sweeps import Loop, Sweep
from strumento.packs.cqed.experiments.readout_state import assignment_fidelity_from_separation
from strumento.packs.cqed.readout import DispersiveIQ
from strumento.packs.cqed.safety import MAX_TIME_SWEEP, MIN_TIME_SWEEP, validate_sweep_range


class ValidationCheck(Experiment):
    """The validation terminal's measured half: a short Ramsey + the two-preparation readout.

    Three reads per shot on one declared delay axis: read 0 (ground), read 1 (after a π) and
    read 2 (the Ramsey fringe at that delay). The analyzer fits the fringe for the detuning
    and T2, and the two preparation rows for the assignment fidelity.

    The readout-fidelity half needs per-shot data (``needs_shots``, #199); the Ramsey half does
    not. A result without shots (the mock, an averaged board acquire) is refused —
    ``success=False``, ``derived["refused"]`` — keeping the averaged Ramsey quantities (detuning,
    T2, …) and naming the withheld readout quantities in ``derived["refused_parts"]``. The
    terminal needs both checks, so a refused run cannot pass as a validation.
    """

    needs_shots = True

    #: what a result without shots cannot give: the per-shot readout-fidelity rows
    SHOT_QUANTITIES = ("assignment_fidelity", "readout_separation", "readout_threshold")

    def __init__(self, device, *, delays: Sweep | None = None, points: int = 81,
                 readout: DispersiveIQ | None = None, runner=None):
        super().__init__(device, runner=runner)
        ro = device.calib.readout
        self.delays = delays if delays is not None else Sweep(start=0.0, stop=80.0, on="t")
        validate_sweep_range(self.delays, min_val=MIN_TIME_SWEEP,
                             max_val=MAX_TIME_SWEEP, label="delay")
        self.points = points
        self.readout = readout if readout is not None else DispersiveIQ.from_calib(ro)

    def axes(self) -> tuple:
        if isinstance(self.delays, Sweep) and self.delays.on is not None:
            return (Loop(self.delays.on, self.points),)
        return ()

    def sequence(self) -> tuple[Seq, tuple]:
        q = self.device.qubit
        seq = Seq()
        seq.play(self.device.readout.dispersive_measure())  # read 0: the ground state
        seq.play(q.ge_pi())                                  # prepare |e>
        seq.play(self.device.readout.dispersive_measure())   # read 1: the excited state
        seq.play(q.ge_hpi())                                 # the Ramsey half-pi
        seq.wait(self.delays)                                # the swept idle
        seq.play(q.ge_hpi())
        seq.play(self.device.readout.dispersive_measure())   # read 2: the Ramsey fringe
        return seq, ()

    def analyze(self, result: MeasurementResult) -> FitResult:
        delays = np.asarray(result.xpts, dtype=float)

        # -- the Ramsey row (the data read: the last read in the shot) ---------------------
        iq = np.stack(
            [np.asarray(result.avgi, dtype=float), np.asarray(result.avgq, dtype=float)], axis=-1
        )
        ramsey_sig = self.readout.project(iq)
        while np.asarray(ramsey_sig).ndim > 1:
            ramsey_sig = np.asarray(ramsey_sig)[-1]
        ramsey = fit_decaysin_detuned(delays, np.asarray(ramsey_sig, dtype=float))

        # -- the readout-fidelity rows (per-shot, both preparations) ----------------------
        shots = result.shots
        if shots is None:
            # #199: two averaged rows are not a per-shot histogram — keep the averaged Ramsey,
            # refuse the readout-fidelity rows rather than fit a double Gaussian to two numbers.
            return shots_refusal(kept=dict(ramsey.derived), withheld=self.SHOT_QUANTITIES)
        arr = np.asarray(shots, dtype=float)          # (n_pts, 3, shots, 2)
        g_iq = arr[:, 0].reshape(-1, 2)               # every delay point's ground shots
        e_iq = arr[:, 1].reshape(-1, 2)               # every delay point's excited shots
        g_proj = self.readout.project(g_iq).ravel()
        e_proj = self.readout.project(e_iq).ravel()

        combined = np.concatenate([g_proj, e_proj])
        lo, hi = float(combined.min()), float(combined.max())
        pad = 1e-9 if hi <= lo else (hi - lo) * 1e-3
        bins = np.linspace(lo - pad, hi + pad, 41)
        counts, _edges = np.histogram(combined, bins=bins)
        blob_fit = fit_double_gaussian(0.5 * (bins[:-1] + bins[1:]), counts.astype(float))
        separation = float(blob_fit.derived["separation"])
        fidelity = assignment_fidelity_from_separation(separation)

        derived = dict(ramsey.derived)
        derived["assignment_fidelity"] = fidelity
        derived["readout_separation"] = separation
        derived["readout_threshold"] = float(blob_fit.derived["threshold"])
        return FitResult(
            params=dict(ramsey.params),
            stderr=dict(ramsey.stderr),
            derived=derived,
            success=bool(ramsey.success),
        )
