"""Heralded initialization — active-reset preparation fidelity measurement (Layer 2).

Sequence: N rounds of ``measure · settle · if(excited) π``, followed by a verification
readout. Per the pack's read convention, the verification read is the last read (data read)
and every reset round's read is a herald lane. The analyzer reports the ground-state
preparation fidelity from per-shot verification data.

There is no curve to fit: the reported quantity is a mean over shots. :meth:`analyze` returns
a typed :class:`~strumento.core.fitters.FitResult` (hard rule 7) with the populations under
``derived`` and ``stderr`` carrying the binomial shot-noise error. It needs **per-shot** data
(``needs_shots``): a result that carries only averages is refused — ``success=False`` with
``derived["refused"]`` saying so (#199).

Applicable to: LD, Ge hole (any encoding with a calibrated pi pulse and charge-sensor readout).

Pure at import: no ``qick``.
"""

from __future__ import annotations

import numpy as np

from strumento.core.experiment import Experiment, shots_refusal
from strumento.core.fitters import FitResult
from strumento.core.pulses import Seq
from strumento.core.result import MeasurementResult
from strumento.packs.spin.library.composite import (
    active_reset,
)
from strumento.packs.spin.readout import PSBReadout

#: The herald lane naming: the **last** reset round certifies the state,
#: earlier rounds are ``reset_1 … reset_{rounds-1}``.
FINAL_LANE = "reset"


class HeraldedInit(Experiment):
    """Measure the ground-state preparation fidelity after N rounds of active reset.

    The experiment builds the active-reset composite (measure → settle → branch → π)
    repeated ``rounds`` times, followed by a verification readout. The analyzer
    extracts ground-state and excited-state populations from per-shot data.
    """

    needs_shots = True

    def __init__(
        self,
        device,
        *,
        dot_name: str = "q1",
        rounds: int = 1,
        threshold: float = 0.5,
        adc: int = 0,
        settle_us: float | None = None,
        hardware_loop: bool = False,
        readout: PSBReadout | None = None,
        runner=None,
    ) -> None:
        super().__init__(device, runner=runner)
        self._dot = getattr(self.device, dot_name)
        self._dot_name = dot_name
        self.rounds = int(rounds)
        self.threshold = float(threshold)
        self.adc = adc
        self.settle_us = settle_us
        self.hardware_loop = bool(hardware_loop)
        self.readout = readout if readout is not None else PSBReadout(threshold=threshold)

    def axes(self) -> tuple:
        return ()

    def sequence(self) -> tuple[Seq, tuple]:
        reset = active_reset(
            self._dot,
            rounds=self.rounds,
            threshold=self.threshold,
            adc=self.adc,
            settle_us=self.settle_us,
            hardware_loop=self.hardware_loop,
        )
        return reset.measure(), ()

    def analyze(self, result: MeasurementResult) -> FitResult:
        """Ground/excited population from the verification read (last read).

        Needs per-shot data (``result.shots``). Without it the analysis is refused (#199): the
        averaged signal is not a population, and reporting it as one (with a zero error bar) was
        the silent fallback this replaced.
        """
        if result.shots is None:
            return shots_refusal()
        # Per-shot path: discriminate each shot's verification read
        shots = np.asarray(result.shots, dtype=float)
        # The verification read is the last read; extract its I-channel
        if shots.ndim >= 2:
            verify = shots[-1] if shots.ndim == 2 else shots[..., -1, :, 0]
        else:
            verify = shots
        verify_flat = np.asarray(verify, dtype=float).ravel()
        excited = self.readout.discriminate(verify_flat)
        n_shots = len(excited)
        p_excited = float(np.mean(excited))

        p_ground = 1.0 - p_excited

        # Binomial shot-noise error
        if n_shots > 1:
            se = float(np.sqrt(p_excited * p_ground / n_shots))
        else:
            se = 0.0

        return FitResult(
            params={"p_excited": p_excited},
            stderr={"p_excited": se},
            derived={
                "p_ground": p_ground,
                "p_excited": p_excited,
                "rounds": self.rounds,
            },
            success=True,
        )
