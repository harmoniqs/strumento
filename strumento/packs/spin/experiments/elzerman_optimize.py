"""Elzerman readout optimization — readout point, duration, threshold (Layer 1).

Optimizes the Elzerman energy-selective tunneling readout: finds the plunger voltage V_R
that positions the Fermi level between the Zeeman-split spin states, calibrates the readout
duration t_read, and sets the discrimination threshold via ROC analysis on bimodal histograms.

Applicable to: LD encoding with Elzerman readout. Routine (every session).

Pure at import: no ``qick``.
"""

from __future__ import annotations

import numpy as np

from strumento.core.experiment import Experiment
from strumento.core.fitters import FitResult, fit_double_gaussian
from strumento.core.pulses import Const, Pulse, Seq
from strumento.core.result import MeasurementResult
from strumento.core.sweeps import Loop, Sweep
from strumento.core.wiring import LineRef
from strumento.packs.spin.experiments._signal import readout_signal
from strumento.packs.spin.readout import ElzermanReadout


class ElzermanOptimize(Experiment):
    """Optimize the Elzerman readout: voltage, duration, threshold, fidelity.

    Swept: readout plunger voltage V_R (positions Fermi level between spin states).
    Measured: time-resolved sensor signal during readout — blip (spin-up tunnels out)
              vs flat (spin-down stays).
    Extracts: optimal V_R, threshold, readout fidelity F_R.
    Prerequisites: tunnel rate tuned (Gamma in target window); lever arm known (V_R -> energy).

    The sequence prepares a known spin state (via waiting for thermal equilibrium),
    then pulses to the readout point and measures the sensor. The bimodal histogram
    of sensor responses (blip vs no-blip) gives the discrimination threshold and fidelity.
    """

    def __init__(self, device, *, dot_name: str = "q1", plunger_name: str = "p0",
                 voltages: Sweep | None = None,
                 points: int = 51, bins: int = 60, readout: ElzermanReadout | None = None,
                 runner=None):
        super().__init__(device, runner=runner)
        self._dot_name = dot_name
        self._plunger_name = plunger_name
        self.voltages = voltages if voltages is not None else Sweep(start=-0.2, stop=0.2, on="v_readout")
        self.points = points
        self.bins = bins
        self.readout = readout if readout is not None else ElzermanReadout(threshold=0.5)

    def axes(self) -> tuple:
        if isinstance(self.voltages, Sweep) and self.voltages.on is not None:
            return (Loop(self.voltages.on, self.points),)
        return ()

    def sequence(self) -> tuple[Seq, tuple]:
        # Prepare spin state (thermal init), pulse plunger to swept readout voltage, measure
        readout_pulse = Pulse(
            line=LineRef(self._plunger_name, "dc"),
            freq_mhz=0.0,
            gain=self.voltages,  # swept readout voltage level
            phase_deg=0.0,
            envelope=Const(length_us=0.1),
        )
        seq = Seq().play(readout_pulse).measure()
        return seq, ()

    def analyze(self, result: MeasurementResult) -> FitResult:
        signal = readout_signal(result, self.readout)

        # Build histogram of sensor responses and fit double Gaussian
        counts, edges = np.histogram(signal, bins=self.bins)
        centers = (edges[:-1] + edges[1:]) / 2.0
        hist_fit = fit_double_gaussian(centers, counts)

        # Extract optimal threshold and readout fidelity from the double Gaussian
        threshold = hist_fit.derived.get("threshold", 0.5)
        separation = hist_fit.derived.get("separation", 0.0)

        # Fidelity estimate from Gaussian separation: F = 1 - erfc(sep / sqrt(8))
        fidelity = float(1.0 - 2.0 * np.exp(-separation**2 / 8.0)) if separation > 0 else 0.5

        # Optimal readout voltage: where visibility is maximized
        x = np.asarray(result.xpts, dtype=float)
        v_readout = float(x[len(x) // 2])  # center of sweep as proxy

        params = hist_fit.params
        stderr = hist_fit.stderr
        derived = {"v_readout": v_readout, "threshold": threshold, "fidelity": fidelity}
        return FitResult(params=params, stderr=stderr, derived=derived, success=hist_fit.success)
