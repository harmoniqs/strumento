"""Exchange oscillation — J(epsilon) mapping and exchange pulse calibration (Layer 2).

Sweeps the exchange pulse duration (at a fixed detuning/barrier voltage) and measures the
resulting oscillation between spin states. The oscillation frequency is J/h and calibrates
the pi-pulse time for exchange-mediated two-qubit gates.

Applicable to: S-T (X-rotation via J), EO (both J_12 and J_23 axes), LD (two-qubit gate).
Frequent recalibration (J fluctuates with charge noise).

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


class ExchangeOscillation(Experiment):
    """Exchange oscillation: prepare → exchange_pulse(t) → measure.

    Swept: exchange pulse duration t (µs) at fixed barrier voltage.
    Measured: P_singlet or P_up after the exchange pulse — oscillates at J/h.
    Extracts: J_MHz (exchange coupling), t_pi_exchange (half-period).

    ``barrier_name`` selects which barrier gate carries the exchange pulse. The exchange
    pulse is a flat-top baseband pulse on the barrier's ``fast`` line, with swept length.
    The dot's calibrated exchange gain is used (from ``device.calib.gates[barrier]``).
    """

    def __init__(self, device, *, dot_name: str = "q1", barrier_name: str = "b12",
                 durations: Sweep | None = None, points: int = 101,
                 exchange_gain: float | None = None,
                 readout: PSBReadout | None = None, runner=None):
        super().__init__(device, runner=runner)
        self._dot = getattr(self.device, dot_name)
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
        # Build exchange pulse: a flat-top baseband pulse on the barrier's fast line
        # with swept duration. If we have a calibrated barrier element, use its gain;
        # otherwise fall back to the provided exchange_gain or a default.
        gain = self._exchange_gain
        if gain is None:
            # Try to get calibrated gain from the barrier element ops
            barrier = getattr(self.device, self._barrier_name, None)
            if barrier is not None and hasattr(barrier, "pulse"):
                ref = barrier.pulse()
                gain = ref.gain
            else:
                gain = 0.5  # default exchange amplitude (fraction)
        validate_gain(gain, label=f"{self._barrier_name}_exchange")

        exchange_pulse = Pulse(
            line=LineRef(self._barrier_name, "fast"),
            freq_mhz=0.0,  # baseband
            gain=gain,
            phase_deg=0.0,
            envelope=FlatTop(ramp_sigma_us=0.006, length_us=self.durations),
        )
        # Prepare spin-down (ground state), apply exchange of swept duration, measure
        seq = Seq().play(exchange_pulse).measure()
        return seq, ()

    def analyze(self, result: MeasurementResult) -> FitResult:
        fit = fit_rabi(result.xpts, readout_signal(result, self.readout))
        # Re-derive with exchange-relevant names
        period = fit.derived.get("period", 0.0)
        j_mhz = (1.0 / period) if period > 0 else 0.0
        t_pi_exchange = period / 2.0 if period > 0 else 0.0
        fit.derived["J_MHz"] = j_mhz
        fit.derived["t_pi_exchange"] = t_pi_exchange
        return fit
