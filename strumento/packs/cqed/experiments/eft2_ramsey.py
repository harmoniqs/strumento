"""E-F T2 Ramsey — Ramsey fringe on the e-f transition, after a ge-pi shelve (cqed expansion).

Shelves the qubit in |e> with a ge pi pulse (the pack's ge-shelving idiom -- see ``EFRabi``/
``EFSpectroscopy``), then runs the same half-pi -- wait -- half-pi Ramsey fringe the ge
``T2Ramsey`` experiment uses, but on the e-f transition, with an artificial detuning wired into
both ef half-pi drives and a closing ge pi for readout discrimination: ge_pi,
ef_hpi(freq=f_ef+ramsey_freq_mhz), wait(t), ef_hpi(freq=f_ef+ramsey_freq_mhz, phase), ge_pi,
measure. The fit's ``detuning`` reads back ramsey_freq_mhz plus any residual ef-drive mismatch,
and ``tau`` is the ef T2*.

Pure at import: no ``qick`` (the Runner does the qick-facing acquire, lazily).
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


class EFT2Ramsey(Experiment):
    """E-F Ramsey fringe: ``ge_pi · ef_hpi · wait(t) · ef_hpi(phase) · ge_pi · measure``.

    Shelves the qubit in |e> first (the pack's ge-shelving idiom, matching ``EFRabi`` and
    ``EFSpectroscopy``), runs the ge ``T2Ramsey`` experiment's fringe/decay pattern on the e-f
    transition, then maps |e> -> |g> with a closing ge pi so the ge discriminator reads the
    ef-Ramsey result. Fitter and fit model are reused
    verbatim from ``T2Ramsey`` -- only the state preparation and readout mapping differ.

    ``ramsey_freq_mhz`` is an **artificial detuning** wired into both ef half-pi drives
    (``ef_hpi(freq=f_ef + ramsey_freq_mhz)``), exactly as the ge ``T2Ramsey`` does it: the fringe
    then oscillates at that known frequency and the fit's ``detuning`` reads it back plus any
    residual ef-drive mismatch. A per-delay virtual-Z phase ramp on the second half-pi produces
    the identical fringe; the frequency offset here is the strumento-native equivalent. Defaulted to 3.0 MHz rather than the ge experiment's 0.0 MHz -- an e-f Ramsey
    conventionally runs at a larger detuning than a g-e one. The second half-pi also carries a
    static ``advance_phase_deg`` (useful for echo-style phase flips), applied on top of the
    detuned drive.
    """

    def __init__(
        self,
        device,
        *,
        delays: Sweep | None = None,
        points: int = 101,
        advance_phase_deg: float = 0.0,
        ramsey_freq_mhz: float = 3.0,
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
        # Artificial detuning: drive both ef half-pi pulses ramsey_freq_mhz off the calibrated
        # f_ef so the fringe oscillates at that frequency -- the ge ``T2Ramsey`` idiom (equivalent
        # to a per-delay virtual-Z phase ramp). ``ramsey_freq_mhz=0`` reproduces the on-resonance
        # (intrinsic-detuning-only) fringe. A closing ge pi maps |e> -> |g> so the ge
        # discriminator reads the ef-Ramsey result.
        drive_freq = float(self.device.calib.qubit.f_ef) + self.ramsey_freq_mhz
        seq = (
            Seq()
            .play(q.ge_pi())
            .play(q.ef_hpi(freq=drive_freq))
            .wait(self.delays)
            .play(q.ef_hpi(freq=drive_freq, phase=self.advance_phase_deg))
            .play(q.ge_pi())
            .play(self.device.readout.dispersive_measure(line=self.line))
        )
        return seq, ()

    def analyze(self, result: MeasurementResult) -> FitResult:
        return fit_decaysin_detuned(result.xpts, readout_signal(result, self.readout))
