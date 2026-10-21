"""Direct single-photon Ramsey measurement of cavity T2*."""

from __future__ import annotations

import math

import numpy as np

from strumento.core.experiment import Experiment
from strumento.core.fitters import FitResult, fit_decaysin_detuned
from strumento.core.pulses import Seq
from strumento.core.result import MeasurementResult
from strumento.core.sweeps import Loop, Sweep
from strumento.packs.cqed.experiments._signal import readout_iq, readout_signal
from strumento.packs.cqed.readout import DispersiveIQ
from strumento.packs.cqed.safety import MAX_TIME_SWEEP, MIN_TIME_SWEEP, validate_sweep_range


class FockCavityT2Ramsey(Experiment):
    """Measure T2* of the buffer addressed by calibrated ``f0-g1(0)``.

    ``ge_hpi -> ef_pi -> f0g1_pi`` prepares a cavity single-rail superposition while returning
    the qubit to ``|g>``. After the swept idle, a phase-advanced return ``f0g1_pi`` retrieves the
    photon, ``ef_pi`` maps ``|f,0>`` to ``|e,0>``, and a final ``ge_hpi`` converts the cavity phase
    into qubit population. The return-sideband phase advances at ``reference_detuning_mhz`` so a
    damped fringe constrains the T2* envelope without detuning the calibrated sideband carrier.
    """

    def __init__(
        self,
        device,
        *,
        delays: Sweep | None = None,
        points: int = 101,
        reference_detuning_mhz: float = 0.02,
        return_phase_offset_deg: float = 0.0,
        readout: DispersiveIQ | None = None,
        runner=None,
    ):
        super().__init__(device, runner=runner)
        self.delays = delays if delays is not None else Sweep(start=0.0, stop=300.0, on="t")
        validate_sweep_range(self.delays, min_val=MIN_TIME_SWEEP, max_val=MAX_TIME_SWEEP, label="delay")
        if not math.isfinite(reference_detuning_mhz) or reference_detuning_mhz == 0.0:
            raise ValueError("reference_detuning_mhz must be finite and non-zero")
        if not math.isfinite(return_phase_offset_deg):
            raise ValueError("return_phase_offset_deg must be finite")
        self.points = int(points)
        self.reference_detuning_mhz = float(reference_detuning_mhz)
        self.return_phase_offset_deg = float(return_phase_offset_deg)
        self.readout = readout if readout is not None else DispersiveIQ.from_calib(device.calib.readout)

    def axes(self) -> tuple:
        if isinstance(self.delays, Sweep) and self.delays.on is not None:
            return (Loop(self.delays.on, self.points),)
        return ()

    def sequence(self) -> tuple[Seq, tuple]:
        dev = self.device
        return_phase = self.delays.mapped(
            lambda delay_us: self.return_phase_offset_deg
            + 360.0 * self.reference_detuning_mhz * delay_us,
            inverse=lambda phase_deg: (phase_deg - self.return_phase_offset_deg)
            / (360.0 * self.reference_detuning_mhz),
        )
        seq = (
            Seq()
            .play(dev.qubit.ge_hpi())
            .play(dev.qubit.ef_pi())
            .play(dev.sideband.f0g1(photon=0))
            .wait(self.delays)
            .play(dev.sideband.f0g1(photon=0, phase=return_phase))
            .play(dev.qubit.ef_pi())
            .play(dev.qubit.ge_hpi())
        )
        seq += dev.readout.dispersive_measure()
        return seq, ()

    def analyze(self, result: MeasurementResult) -> FitResult:
        fit = fit_decaysin_detuned(result.xpts, readout_signal(result, self.readout))
        if self.readout.f_center is None:
            return fit

        raw_i, raw_q = readout_iq(result)
        populations = self.readout.to_level_populations(np.stack((raw_i, raw_q), axis=-1))
        fit.derived.update(
            {
                "ramsey_population_contrast": float(np.ptp(populations[:, 1])),
                "max_f_population": float(np.max(populations[:, 2])),
            }
        )
        return fit

    def recovery_map(self) -> dict[str, str]:
        return {"T2": "cavity_T2"}
