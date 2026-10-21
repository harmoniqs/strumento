"""Photon-parity fringe experiment.

Scans the second half-pi's phase through a full turn after the parity revival wait; the fringe
contrast is the parity signal. Analyzer fits a plain cosine (``fit_parity``).

Pure at import: no ``qick``.
"""

from __future__ import annotations

import numpy as np

from strumento.core.experiment import Experiment
from strumento.core.fitters import FitResult, fit_parity
from strumento.core.pulses import Seq
from strumento.core.result import MeasurementResult
from strumento.core.sweeps import Loop, Sweep
from strumento.packs.cqed.experiments._signal import readout_population, readout_signal
from strumento.packs.cqed.library.composite import PARITY_CLOSING_PHASE_DEG, parity
from strumento.packs.cqed.library.targets import resolve_mode
from strumento.packs.cqed.readout import DispersiveIQ


class ParityFringe(Experiment):
    """Measure the photon-number parity of a cavity mode via a Ramsey-style fringe on the qubit.

    Sequence: ``ge_hpi`` -- wait the parity revival time (pi/chi) -- ``ge_hpi(phase)``
    -- measure, with the second half-pi phase swept 0-360 degrees.  The fringe
    contrast encodes the parity expectation value.
    Fitter: ``fit_parity`` (plain cosine fit of the fringe vs. phase).

    Derived quantities:
        contrast  : peak-to-peak fringe amplitude (2 * |amp|), proportional to parity signal
        phase_deg : fringe phase offset in degrees

    When the readout carries g/e blob centres, ``analyze`` fits on the **population** path
    (``readout_population``) instead of the raw signal, and additionally derives the Wigner-parity
    normalization the reconstruction consumes:

        alpha_scale : population contrast (2*|amp|, now in [0, 1]) — the parity-contrast divisor
        parity_sign : sign(1 - 2 P_e) of the fitted vacuum fringe at the parity operating phase
                      (``PARITY_CLOSING_PHASE_DEG``) — the excited<->parity mapping
        pe_midpoint : fitted offset (the single-phase offset guard; a centered fringe sits at 0.5)

    Without blob centres it behaves exactly as before (signal-space contrast only).

    Parameters:
        mode      : device mode name used for the parity composite (default: the device's cavity)
        phases    : phase sweep for the second half-pi; defaults to 0-360 degrees
        points    : number of phase points (default 61)
        fast      : if ``True``, use the fast parity composite variant
        readout   : ``DispersiveIQ`` readout config; defaults to the recorded calibration
                    (``DispersiveIQ.from_calib``)
        line      : wiring line override for the readout probe (``None`` = the normal
                    ``("readout", "probe")`` line); see
                    :func:`~strumento.packs.cqed.library.composite.parity`.

    Usage::

        exp = ParityFringe(dev, mode="cavity")
        outcome = exp.go(soc, reps=300)
        outcome.fit.derived["contrast"]
    """

    def __init__(self, device, *, mode: str | None = None, phases: Sweep | None = None,
                 points: int = 61, fast: bool = False, readout: DispersiveIQ | None = None,
                 line=None, runner=None):
        super().__init__(device, runner=runner)
        mode = resolve_mode(device, mode, purpose=type(self).__name__, arg="mode")
        self.mode = mode
        self.fast = fast
        self.phases = phases if phases is not None else Sweep(start=0.0, stop=360.0, on="phase")
        self.points = points
        self.readout = readout if readout is not None else DispersiveIQ.from_calib(device.calib.readout)
        self.line = line

    def axes(self) -> tuple:
        if isinstance(self.phases, Sweep) and self.phases.on is not None:
            return (Loop(self.phases.on, self.points),)
        return ()

    def sequence(self) -> tuple[Seq, tuple]:
        # #100: parity() now drives the readout and measures internally by default -- chaining
        # our own .measure() here would double-measure.
        seq = parity(self.device, self.mode, second_phase=self.phases, fast=self.fast,
                     line=self.line)
        return seq, ()

    def analyze(self, result: MeasurementResult) -> FitResult:
        # Without blob centres, keep the historical signal-space fit (contrast/phase_deg only).
        if self.readout.g_center is None or self.readout.e_center is None:
            return fit_parity(result.xpts, readout_signal(result, self.readout))

        # Population path: fit P_e and derive the Wigner-parity normalization the reconstruction
        # reads. The Wigner samples parity at the composite's fixed closing phase, so evaluate the
        # fitted vacuum fringe THERE to fix the excited<->parity sign (not a naive phase heuristic).
        fit = fit_parity(result.xpts, readout_population(result, self.readout))
        p = fit.params
        pe_op = float(
            p["offset"]
            + p["amp"] * np.cos(2 * np.pi * p["freq"] * PARITY_CLOSING_PHASE_DEG + p["phase"])
        )
        vacuum_parity = 1.0 - 2.0 * pe_op
        fit.derived["alpha_scale"] = fit.derived["contrast"]  # 2*|amp|, population units in [0, 1]
        fit.derived["parity_sign"] = 1.0 if vacuum_parity >= 0.0 else -1.0
        fit.derived["pe_midpoint"] = float(p["offset"])
        return fit
