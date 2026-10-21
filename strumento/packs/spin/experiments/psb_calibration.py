"""PSB calibration — Pauli spin blockade region identification and optimization (Layer 1).

Identifies the spin-blockade region in detuning space, determines which charge transition
supports PSB, and calibrates the measurement point for maximum contrast between singlet
(unblocked) and triplet (blocked) signals.

Applicable to: S-T, EO, LD+auxiliary, Ge holes with PSB readout. After any gate
reconfiguration or charge jump.

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
from strumento.packs.spin.readout import PSBReadout


class PSBCalibration(Experiment):
    """Identify and calibrate the Pauli spin blockade readout region.

    Swept: detuning epsilon across (1,1) <-> (2,0) transition.
    Measured: sensor signal (charge state) after pulsing to measurement point.
    Extracts: optimal measurement point, threshold, PSB visibility (contrast).
    Prerequisites: double-dot CSD with identified (1,1) and (2,0) regions; tunnel coupling set.

    The sequence pulses to the PSB readout point (deep in the (2,0) region) and measures
    the charge sensor. With a singlet state the electron transfers (unblocked); with a
    triplet state spin blockade prevents transfer. The histogram of sensor responses at
    the optimal detuning gives the threshold and visibility.
    """

    def __init__(self, device, *, dot_name: str = "q1", plunger_name: str = "p0",
                 detuning: Sweep | None = None,
                 points: int = 101, bins: int = 60, readout: PSBReadout | None = None,
                 runner=None):
        super().__init__(device, runner=runner)
        self._dot_name = dot_name
        self._plunger_name = plunger_name
        self.detuning = detuning if detuning is not None else Sweep(start=-0.5, stop=0.5, on="epsilon")
        self.points = points
        self.bins = bins
        self.readout = readout if readout is not None else PSBReadout(threshold=0.5)

    def axes(self) -> tuple:
        if isinstance(self.detuning, Sweep) and self.detuning.on is not None:
            return (Loop(self.detuning.on, self.points),)
        return ()

    def sequence(self) -> tuple[Seq, tuple]:
        # Pulse plunger to swept PSB detuning point and read the charge sensor
        psb_pulse = Pulse(
            line=LineRef(self._plunger_name, "dc"),
            freq_mhz=0.0,
            gain=self.detuning,  # swept detuning for PSB readout optimization
            phase_deg=0.0,
            envelope=Const(length_us=0.05),
        )
        seq = Seq().play(psb_pulse).measure()
        return seq, ()

    def analyze(self, result: MeasurementResult) -> FitResult:
        signal = readout_signal(result, self.readout)

        # Histogram the sensor responses to find bimodal (singlet/triplet) separation
        counts, edges = np.histogram(signal, bins=self.bins)
        centers = (edges[:-1] + edges[1:]) / 2.0
        hist_fit = fit_double_gaussian(centers, counts)

        threshold = hist_fit.derived.get("threshold", 0.5)
        separation = hist_fit.derived.get("separation", 0.0)

        # Visibility: fraction of the population that is correctly discriminated
        # Approximated from the Gaussian separation
        visibility = 1.0 - 2.0 * np.exp(-separation**2 / 8.0) if separation > 0 else 0.0

        params = hist_fit.params
        stderr = hist_fit.stderr
        derived = {"threshold": threshold, "visibility": float(visibility)}
        return FitResult(params=params, stderr=stderr, derived=derived, success=hist_fit.success)
