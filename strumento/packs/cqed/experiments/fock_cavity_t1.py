"""Direct single-photon lifetime measurement through the f0-g1 sideband."""

from __future__ import annotations

import numpy as np

from strumento.core.experiment import Experiment
from strumento.core.fitters import FitResult, fit_exp
from strumento.core.pulses import Seq
from strumento.core.result import MeasurementResult
from strumento.core.sweeps import Loop, Sweep
from strumento.packs.cqed.experiments._signal import readout_iq, readout_signal
from strumento.packs.cqed.library.composite import Fock, prepare
from strumento.packs.cqed.readout import DispersiveIQ
from strumento.packs.cqed.safety import MAX_TIME_SWEEP, MIN_TIME_SWEEP, validate_sweep_range


class FockCavityT1(Experiment):
    """Measure the lifetime of the buffer addressed by the calibrated ``f0-g1(0)`` sideband.

    A shot prepares ``|g,1>`` through ``ge_pi -> ef_pi -> f0g1``, idles for a swept delay,
    retrieves a surviving photon through a second ``f0g1``, maps ``|f,0>`` to bright ``|e,0>``
    with ``ef_pi``, then performs ordinary dispersive qubit readout. Photon survival therefore
    follows ``P_1(t) = P_1(0) exp(-t / T1)`` directly; unlike coherent-state parity decay, this
    is a single exponential.

    The sideband calibration identifies the physical buffer. The experiment deliberately accepts
    no ``mode_name`` parameter because the current calibration schema does not bind individual
    f0-g1 rungs to named modes.
    """

    def __init__(
        self,
        device,
        *,
        delays: Sweep | None = None,
        points: int = 81,
        readout: DispersiveIQ | None = None,
        runner=None,
    ):
        super().__init__(device, runner=runner)
        self.delays = delays if delays is not None else Sweep(start=0.0, stop=1200.0, on="t")
        validate_sweep_range(self.delays, min_val=MIN_TIME_SWEEP, max_val=MAX_TIME_SWEEP, label="delay")
        self.points = int(points)
        self.readout = readout if readout is not None else DispersiveIQ.from_calib(device.calib.readout)

    def axes(self) -> tuple:
        if isinstance(self.delays, Sweep) and self.delays.on is not None:
            return (Loop(self.delays.on, self.points),)
        return ()

    def sequence(self) -> tuple[Seq, tuple]:
        dev = self.device
        seq = prepare(dev, Fock(1))
        seq = seq.wait(self.delays)
        seq = seq.play(dev.sideband.f0g1(photon=0))
        seq = seq.play(dev.qubit.ef_pi())
        seq += dev.readout.dispersive_measure()
        return seq, ()

    def analyze(self, result: MeasurementResult) -> FitResult:
        fit = fit_exp(result.xpts, readout_signal(result, self.readout))
        if self.readout.f_center is None:
            return fit

        raw_i, raw_q = readout_iq(result)
        populations = self.readout.to_level_populations(np.stack((raw_i, raw_q), axis=-1))
        fit.derived.update(
            {
                "initial_mapped_e_population": float(populations[0, 1]),
                "final_mapped_e_population": float(populations[-1, 1]),
                "max_f_population": float(np.max(populations[:, 2])),
            }
        )
        return fit
