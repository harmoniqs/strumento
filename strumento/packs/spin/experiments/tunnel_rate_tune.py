"""Tunnel rate tuning — dot-reservoir tunnel rate calibration (Layer 1).

Measures the tunnel rate Gamma between a quantum dot and its reservoir by monitoring
real-time telegraph noise at the charge transition. For Elzerman readout, Gamma must
be in a window: fast enough for readout but slow enough to resolve the blip.

Target: Gamma ~ 1-100 kHz (tunneling time tau = 1/Gamma ~ 10-1000 us).
Applicable to: LD encoding with Elzerman readout. Routine recalibration (daily).

This is a registered ``Experiment`` subclass but uses ``lmfit`` directly for its
exponential tunnel-rate fit — the same design choice as the DC-sweep standalone classes
(see ``rf_reflectometry_tune.py`` for the rationale).

Pure at import: no ``qick``.
"""

from __future__ import annotations

import lmfit
import numpy as np

from strumento.core.experiment import Experiment
from strumento.core.fitters import FitResult
from strumento.core.pulses import Const, Pulse, Seq
from strumento.core.result import MeasurementResult
from strumento.core.sweeps import Loop, Sweep
from strumento.core.wiring import LineRef
from strumento.packs.spin.experiments._signal import readout_signal
from strumento.packs.spin.readout import ElzermanReadout
from strumento.packs.spin.safety import validate_gain


def _exponential(x, amp, gamma, offset):
    """Single exponential decay: amp * exp(-gamma * x) + offset."""
    return amp * np.exp(-gamma * x) + offset


class TunnelRateTune(Experiment):
    """Measure and tune the dot-reservoir tunnel rate Gamma.

    Swept: readout delay time after pulsing to the charge degeneracy point.
    Measured: time-resolved sensor signal — exponential decay as electron tunnels.
    Extracts: Gamma_in, Gamma_out from exponential fits to tunnel event timing.
    Prerequisites: last-electron isolation (at the 0<->1 transition).

    The sequence applies a detuning pulse (moving the dot level past the reservoir Fermi
    level) and then measures the sensor signal at a swept delay time. The exponential
    decay of the tunnel-in/out probability gives the tunnel rates.
    """

    def __init__(self, device, *, dot_name: str = "q1", plunger_name: str = "p0",
                 delays: Sweep | None = None,
                 points: int = 101, readout: ElzermanReadout | None = None, runner=None):
        super().__init__(device, runner=runner)
        self._dot_name = dot_name
        self._plunger_name = plunger_name
        self.delays = delays if delays is not None else Sweep(start=0.0, stop=1.0, on="t_delay")
        self.points = points
        self.readout = readout if readout is not None else ElzermanReadout(threshold=0.5)

    def axes(self) -> tuple:
        if isinstance(self.delays, Sweep) and self.delays.on is not None:
            return (Loop(self.delays.on, self.points),)
        return ()

    def sequence(self) -> tuple[Seq, tuple]:
        # Detuning pulse on plunger to move dot level past Fermi level, then time-resolved measurement
        detuning_gain = 0.5
        validate_gain(detuning_gain, label=f"{self._plunger_name}_detuning")
        detuning_pulse = Pulse(
            line=LineRef(self._plunger_name, "dc"),
            freq_mhz=0.0,
            gain=detuning_gain,
            phase_deg=0.0,
            envelope=Const(length_us=0.01),
        )
        seq = Seq().play(detuning_pulse).wait(self.delays).measure()
        return seq, ()

    def analyze(self, result: MeasurementResult) -> FitResult:
        x = np.asarray(result.xpts, dtype=float)
        signal = readout_signal(result, self.readout)

        # Fit exponential decay to extract tunnel rate
        # P(occupied, t) = 1 - exp(-Gamma_in * t) for tunnel-in
        # P(empty, t) = exp(-Gamma_out * t) for tunnel-out
        span = float(x.max() - x.min()) or 1.0

        model = lmfit.Model(_exponential)
        pars = model.make_params(
            amp=float(signal[0] - signal[-1]),
            gamma=3.0 / span,  # seed: ~3 time constants in the window
            offset=float(signal[-1]),
        )
        pars["gamma"].set(min=1e-6)

        out = model.fit(signal, pars, x=x)

        params = {k: float(v.value) for k, v in out.params.items()}
        stderr = {k: (float(v.stderr) if v.stderr is not None else float("nan"))
                  for k, v in out.params.items()}

        # Convert rate from 1/us to kHz (1/us = 1000 kHz)
        gamma = abs(params["gamma"])
        gamma_in_khz = gamma * 1e3
        gamma_out_khz = gamma * 1e3  # symmetric approximation from single trace

        derived = {"gamma_in_khz": gamma_in_khz, "gamma_out_khz": gamma_out_khz}
        return FitResult(params=params, stderr=stderr, derived=derived, success=bool(out.success))
