"""Active-reset experiment — how well does the conditional π prepare ``|g⟩``?

Sequence: the :func:`~strumento.packs.cqed.library.composite.active_reset` composite (``measure ·
settle · if(|e⟩) π``, ``rounds`` times) followed by a **verification read**. Per the pack's read
convention the verification read is the last read (the data read) and every reset round's read is
a herald lane, so the answer and the post-selection material come out of one acquisition.

There is no curve to fit here: the reported quantity is a mean over shots. :meth:`analyze` still
returns a :class:`~strumento.core.fitters.FitResult` (hard rule 7 — experiments report typed
results, never stdout) with the populations under ``derived`` and ``stderr`` carrying the binomial
shot-noise error. It needs **per-shot** data (``needs_shots``): a result that carries only averages
(the mock's zeros, an averaged hardware acquire) is refused — ``success=False`` with
``derived["refused"]`` saying so (#199) — rather than a fabricated number.

Pure at import: no ``qick``.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from strumento.core.experiment import Experiment, shots_refusal
from strumento.core.fitters import FitResult
from strumento.core.pulses import Seq
from strumento.core.result import MeasurementResult
from strumento.packs.cqed.library.composite import (
    DEFAULT_RESET_THRESHOLD,
    active_reset,
)
from strumento.packs.cqed.readout import DispersiveIQ

#: the herald lane naming: the **last** reset round certifies the state the shot ends in, so it
#: carries the bare name and is the only lane post-selected by default; earlier rounds are
#: ``reset_1 … reset_{rounds-1}``.
FINAL_LANE = "reset"


class ActiveReset(Experiment):
    """Measure the ground-state preparation fidelity of ``rounds`` of active reset.

    Sequence: the ``active_reset`` composite (measure, settle, conditional pi --
    repeated ``rounds`` times) followed by a verification readout. Each reset
    round's measurement is a herald lane; the final read is the data read.
    Fitter: none (shot statistics). Per-shot data is required (``needs_shots``); a result
    without shots is refused: ``success=False``, ``derived["refused"]`` gives the reason.

    Derived quantities:
        ground_population    : fraction of verification shots in |g>.
        excited_population   : fraction of verification shots in |e>.
        corrected_round_{i}  : fraction of shots each round fired its
                               conditional pi (one key per round).

    Parameters:
        rounds         : number of measure-and-correct cycles.
        threshold      : discrimination threshold for the host DispersiveIQ (projected units),
                         and the on-board branch too when ``branch_threshold`` is None.
        branch_threshold: on-board branch threshold in raw ADC-accumulator units (``--arm``);
                         ``None`` = fall back to ``threshold`` (the ``--sim`` default). See trap #1.
        adc            : ADC channel override (``None`` = device default).
        test           : branch polarity, ``">="`` (|e> reads high, default) or ``"<"`` (trap #2).
        settle_us      : settle time between reset rounds (``None`` = default).
        hardware_loop  : use the hardware loop for the reset rounds.
        readout_phase_deg: DDC demod phase (deg) folded into every reset round AND the verify read,
                         so the branch's raw quadrature carries the g/e separation (trap #1). ``0.0``
                         (default) = the software-angle path (``--sim``); ``--arm`` sets it to the
                         calibrated discrimination angle and reads with ``DispersiveIQ(angle=0.0)``.
        readout        : ``DispersiveIQ`` classifier (default: ``threshold``, angle 0 and the
                         recorded blob centres).
        line           : wiring line override for the readout probe (``None`` = the normal
                         ``("readout", "probe")`` line); see
                         :func:`~strumento.packs.cqed.library.composite.active_reset`.

    Usage::

        exp = ActiveReset(dev, rounds=3, threshold=0.5)
        outcome = exp.go(soc, reps=1000)
        outcome.fit.derived["ground_population"]
    """

    needs_shots = True

    def __init__(
        self,
        device,
        *,
        rounds: int = 1,
        threshold: float = DEFAULT_RESET_THRESHOLD,
        branch_threshold: float | None = None,
        adc: int | None = None,
        test: str = ">=",
        settle_us: float | None = None,
        hardware_loop: bool = False,
        readout_phase_deg: float = 0.0,
        readout: DispersiveIQ | None = None,
        line=None,
        runner=None,
    ) -> None:
        super().__init__(device, runner=runner)
        self.rounds = int(rounds)
        self.threshold = float(threshold)
        self.branch_threshold = None if branch_threshold is None else float(branch_threshold)
        self.adc = adc
        self.test = test
        self.settle_us = settle_us
        self.hardware_loop = bool(hardware_loop)
        self.readout_phase_deg = float(readout_phase_deg)
        # the discriminator keeps the projection threshold; the branch gets branch_threshold (trap #1).
        # Its threshold and angle stay tied to the in-program reset arguments (``threshold``, and
        # ``readout_phase_deg`` — an --arm run rotates in the DDC and reads with angle 0), so only
        # the analysis-side blob centres come from the recorded calibration (#196).
        if readout is None:
            recorded = DispersiveIQ.from_calib(device.calib.readout)
            readout = replace(recorded, threshold=float(threshold), angle=0.0)
        self.readout = readout
        self.line = line

    def sequence(self) -> tuple[Seq, tuple]:
        reset = active_reset(
            self.device,
            rounds=self.rounds,
            threshold=self.threshold,
            branch_threshold=self.branch_threshold,
            adc=self.adc,
            test=self.test,
            settle_us=self.settle_us,
            hardware_loop=self.hardware_loop,
            readout_phase_deg=self.readout_phase_deg,
            line=self.line,
        )
        # No pre-verification wait: the compiler's cursor already sits past the last round's
        # readout window, so the verification read cannot land inside it.
        # #100: the verification read drives the readout probe too -- a bare .measure() read an
        # undriven resonator.
        reset += self.device.readout.dispersive_measure(
            line=self.line, readout_phase_deg=self.readout_phase_deg
        )
        return reset, ()

    def analyze(self, result: MeasurementResult) -> FitResult:
        """Ground/excited population from the verification read, with the reset rounds as heralds.

        Heralds are attached to ``result`` here because the lane structure is a property of the
        *sequence* — which read was a reset round is knowable only from what was run. The IQ-shaped
        ``shots`` lane is deliberately left as-is (its trailing axis is ``(I, Q)``, not shots);
        ``keep_mask()`` is the consumable, and ``postselect()`` compacts the lanes.
        """
        if result.shots is None:
            return shots_refusal()  # #199: say why, rather than an empty fit

        outcomes = np.asarray(self.readout.discriminate(result.shots))
        if outcomes.ndim == 1:  # a single read: no reset round to herald on
            outcomes = outcomes[None, :]
        *rounds, verify = outcomes

        for index, round_outcomes in enumerate(rounds, start=1):
            lane = FINAL_LANE if index == len(rounds) else f"{FINAL_LANE}_{index}"
            result.heralds[lane] = round_outcomes == 0  # keep the shots that read |g>
        if rounds:
            result.herald_lanes = (FINAL_LANE,)

        shots = int(verify.size)
        excited = float(verify.mean()) if shots else float("nan")
        stderr = float(np.sqrt(max(excited * (1.0 - excited), 0.0) / shots)) if shots else float("nan")
        fired = [float(np.mean(r)) for r in rounds]
        return FitResult(
            params={"shots": float(shots), "rounds": float(len(rounds))},
            stderr={"ground_population": stderr, "excited_population": stderr},
            derived={
                "ground_population": 1.0 - excited,
                "excited_population": excited,
                # fraction of shots each round corrected — the thermal population the first round
                # sees, and how little is left for the later ones
                **{f"corrected_round_{i}": frac for i, frac in enumerate(fired, start=1)},
            },
            success=True,
        )
