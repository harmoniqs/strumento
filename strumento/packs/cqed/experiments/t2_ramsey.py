"""T2 Ramsey experiment.

Half-pi, swept idle, second half-pi, measure. Run slightly detuned (set the drive off-resonance
or advance the second half-pi's phase) so the decay envelope carries visible fringes; the fit's
oscillation frequency IS the qubit-drive detuning, and ``tau`` is T2*.

Pure at import: no ``qick``.
"""

from __future__ import annotations

from strumento.core.experiment import Experiment
from strumento.core.fitters import FitResult, fit_decaysin_detuned
from strumento.core.pulses import Seq
from strumento.core.result import MeasurementResult
from strumento.core.sweeps import Loop, Sweep
from strumento.core.wiring import LineRef
from strumento.packs.cqed.experiments._signal import readout_signal
from strumento.packs.cqed.readout import DispersiveIQ
from strumento.packs.cqed.safety import MAX_TIME_SWEEP, MIN_TIME_SWEEP, validate_sweep_range


class T2Ramsey(Experiment):
    """Ramsey fringe measurement: ``ge_hpi · wait(t) · ge_hpi(phase) · measure``.

    ``ramsey_freq_mhz`` is an **artificial detuning**: both half-pi pulses are driven
    ``ramsey_freq_mhz`` off the calibrated ``f_ge``, so the Ramsey fringe oscillates at that known
    frequency and the decay envelope carries visible fringes. On resonance (``ramsey_freq_mhz=0``)
    there is no fringe — only the ``T2*`` envelope — which is why a nonzero value is the usual
    choice; the fit's oscillation frequency then reads back ``ramsey_freq_mhz`` plus any residual
    qubit-drive mismatch. The second half-pi also carries a static ``advance_phase_deg`` (useful
    for echo-style phase flips), applied on top of the detuned drive.
    """

    def __init__(
        self,
        device,
        *,
        delays: Sweep | None = None,
        points: int = 101,
        advance_phase_deg: float = 0.0,
        ramsey_freq_mhz: float = 0.0,
        readout: DispersiveIQ | None = None,
        line=("readout", "probe"),
        runner=None,
    ):
        super().__init__(device, runner=runner)
        self.delays = delays if delays is not None else Sweep(start=0.0, stop=100.0, on="t")
        validate_sweep_range(self.delays, min_val=MIN_TIME_SWEEP,
                             max_val=MAX_TIME_SWEEP, label="delay")
        self.points = points
        self.advance_phase_deg = advance_phase_deg
        self.ramsey_freq_mhz = ramsey_freq_mhz
        self.readout = readout if readout is not None else DispersiveIQ.from_calib(device.calib.readout)
        self.line = LineRef(*line) if not isinstance(line, LineRef) else line

    def axes(self) -> tuple:
        if isinstance(self.delays, Sweep) and self.delays.on is not None:
            return (Loop(self.delays.on, self.points),)
        return ()

    def sequence(self) -> tuple[Seq, tuple]:
        q = self.device.qubit
        # Artificial detuning: drive both half-pi pulses ramsey_freq_mhz off the calibrated f_ge so
        # the fringe oscillates at that frequency (ramsey_freq_mhz=0 -> on resonance -> no fringe,
        # just the T2* envelope). Backward compatible: 0.0 reproduces the on-resonance ge_hpi().
        drive_freq = float(self.device.calib.qubit.f_ge) + self.ramsey_freq_mhz
        seq = (
            Seq()
            .play(q.ge_hpi(freq=drive_freq))
            .wait(self.delays)
            .play(q.ge_hpi(freq=drive_freq, phase=self.advance_phase_deg))
            .play(self.device.readout.dispersive_measure(line=self.line))
        )
        return seq, ()

    def analyze(self, result: MeasurementResult) -> FitResult:
        return fit_decaysin_detuned(result.xpts, readout_signal(result, self.readout))
