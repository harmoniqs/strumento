"""Exchange gate calibration — two-qubit CZ/CPHASE gate (Layer 3).

Calibrates the exchange-mediated two-qubit gate. Applies an exchange pulse of swept duration
at the exchange-on operating point, with one qubit prepared in different states, and measures
the conditional phase accumulation. The gate time t_CZ is where the conditional phase equals pi.

Applicable to: LD (two-qubit), Ge hole (two-qubit), EO (the exchange axes ARE the gates).

Pure at import: no ``qick`` (the Runner does the qick-facing acquire, lazily).
"""

from __future__ import annotations

from strumento.core.experiment import Experiment
from strumento.core.fitters import FitResult, fit_rabi
from strumento.core.pulses import FlatTop, Pulse, Seq
from strumento.core.result import MeasurementResult
from strumento.core.sweeps import Loop, Sweep
from strumento.core.wiring import LineRef
from strumento.packs.spin.experiments._signal import readout_signal
from strumento.packs.spin.readout import PSBReadout
from strumento.packs.spin.safety import validate_gain


class ExchangeGateCalibration(Experiment):
    """Calibrate the exchange-mediated two-qubit gate (CZ/CPHASE).

    Swept: exchange pulse duration t (µs) at the exchange-on operating point.
    Measured: conditional phase or Ramsey fringe of the target qubit (conditioned on
        control qubit state). Oscillates at J_eff.
    Extracts: t_cz (gate time for pi conditional phase), J_eff (effective coupling).

    Protocol: prepare control qubit in |1⟩ (pi pulse on control dot), apply Ramsey on
    the target dot with an exchange pulse of swept duration in between the two hpi's.
    The Ramsey fringe oscillates at J_eff; t_CZ = 1/(2·J_eff).

    ``control_dot`` and ``target_dot`` select the two qubit dots.
    ``barrier_name`` selects the barrier carrying the exchange pulse.
    """

    def __init__(self, device, *, control_dot: str = "q1", target_dot: str = "q2",
                 barrier_name: str = "b12", durations: Sweep | None = None,
                 points: int = 101, exchange_gain: float | None = None,
                 readout: PSBReadout | None = None, runner=None):
        super().__init__(device, runner=runner)
        self._control = getattr(self.device, control_dot)
        self._target = getattr(self.device, target_dot)
        self._barrier_name = barrier_name
        self._exchange_gain = exchange_gain
        self.durations = durations if durations is not None else Sweep(start=0.006, stop=0.500, on="t")
        self.points = points
        self.readout = readout if readout is not None else PSBReadout(threshold=0.5)

    def axes(self) -> tuple:
        if isinstance(self.durations, Sweep) and self.durations.on is not None:
            return (Loop(self.durations.on, self.points),)
        return ()

    def sequence(self) -> tuple[Seq, tuple]:
        # Determine exchange gain
        gain = self._exchange_gain
        if gain is None:
            barrier = getattr(self.device, self._barrier_name, None)
            if barrier is not None and hasattr(barrier, "pulse"):
                gain = barrier.pulse().gain
            else:
                gain = 0.5
        validate_gain(gain, label=f"{self._barrier_name}_exchange_cal")

        exchange_pulse = Pulse(
            line=LineRef(self._barrier_name, "fast"),
            freq_mhz=0.0,
            gain=gain,
            phase_deg=0.0,
            envelope=FlatTop(ramp_sigma_us=0.006, length_us=self.durations),
        )

        # Protocol: pi on control, then Ramsey on target with exchange in between
        # pi(control) → wait → hpi(target) → exchange(t) → hpi(target) → measure
        seq = (
            Seq()
            .play(self._control.pi())
            .wait(0.050)
            .play(self._target.hpi())
            .play(exchange_pulse)
            .play(self._target.hpi())
            .measure()
        )
        return seq, ()

    def analyze(self, result: MeasurementResult) -> FitResult:
        fit = fit_rabi(result.xpts, readout_signal(result, self.readout))
        # Derive exchange gate parameters
        period = fit.derived.get("period", 0.0)
        j_eff_mhz = (1.0 / period) if period > 0 else 0.0
        t_cz_us = period / 2.0 if period > 0 else 0.0
        fit.derived["J_eff_MHz"] = j_eff_mhz
        fit.derived["t_cz_us"] = t_cz_us
        fit.derived["t_cz_ns"] = t_cz_us * 1000.0
        return fit
