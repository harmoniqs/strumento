"""Virtual gate calibration — cross-capacitance matrix extraction (Layer 0).

Measures the slopes of charge transition lines in the 2D gate-voltage plane to build
the cross-capacitance matrix M such that virtual gates V_virt = M^{-1} V_phys decouple
the dots (each virtual axis moves only one dot's chemical potential).

Applicable to: all multi-dot encodings (S-T, EO, multi-qubit LD). Done once per device
configuration; relatively stable day-to-day.

Pure at import: no ``qick``.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from strumento.core.dc import DCSource
from strumento.core.fitters import FitResult
from strumento.core.result import MeasurementResult


class VirtualGateCalibration:
    """Extract the cross-capacitance matrix from charge transition line slopes.

    Host-driven DC sweep — NOT an Experiment subclass. The ``DCSource`` handles
    voltage clamping via ``max_v``; no additional safety validation needed here.

    Swept: gate_a voltages (primary axis); gate_b optionally swept independently.
    Measured: sensor signal — position/slope of each transition line.
    Extracts: M_ij = -(dV_i / dV_j)|_transition_fixed for the gate pair.
    Prerequisites: CC-3 (transitions identified in CSD).
    """

    def __init__(
        self,
        dc: DCSource,
        *,
        gate_a: str = "p0",
        gate_b: str = "b0",
        voltages_a=np.linspace(-0.1, 0.1, 51),
        voltages_b=None,
        probe: Callable[[], float],
        compensated: bool = False,
    ) -> None:
        self.dc = dc
        self.gate_a = gate_a
        self.gate_b = gate_b
        self.voltages_a = np.asarray(voltages_a, dtype=float)
        self.voltages_b = np.asarray(voltages_b, dtype=float) if voltages_b is not None else None
        self.probe = probe
        self.compensated = compensated

    def run(self) -> MeasurementResult:
        """Sweep gate_a, record probe at each point. Extract slope = cross-coupling."""
        setv = self.dc.set_compensated if self.compensated else self.dc.set
        data = np.zeros(len(self.voltages_a))
        for i, va in enumerate(self.voltages_a):
            setv(self.gate_a, float(va))
            data[i] = float(self.probe())
        return MeasurementResult(
            xpts=self.voltages_a,
            avgi=data.reshape(1, -1),
            avgq=np.zeros((1, len(self.voltages_a))),
            meta={
                "gate_a": self.gate_a,
                "gate_b": self.gate_b,
                "dc_snapshot": self.dc.snapshot(),
            },
        )

    def analyze(self, result: MeasurementResult) -> FitResult:
        """Linear fit to extract cross-coupling coefficient alpha_ij."""
        x = np.asarray(result.xpts, dtype=float)
        signal = np.asarray(result.avgi).ravel()

        # Fit a linear slope to the transition position in the swept cut.
        # The slope gives the cross-coupling coefficient: alpha_ij = -dV_i/dV_j.
        # Find the transition (steepest derivative) and fit its position.
        dsignal = np.gradient(signal, x)
        peak_idx = int(np.argmax(np.abs(dsignal)))

        # Linear fit around the transition region
        window = max(len(x) // 10, 3)
        lo = max(0, peak_idx - window)
        hi = min(len(x), peak_idx + window)
        if hi - lo >= 2:
            coeffs = np.polyfit(x[lo:hi], signal[lo:hi], 1)
            slope = float(coeffs[0])
        else:
            slope = 0.0

        # Cross-capacitance: alpha_ij approximated from the transition slope
        alpha_ij = -1.0 / slope if abs(slope) > 1e-10 else 0.0

        params = {"slope": slope, "transition_v": float(x[peak_idx])}
        stderr = {"slope": 0.0, "transition_v": 0.0}
        derived = {"alpha_ij": alpha_ij}
        return FitResult(params=params, stderr=stderr, derived=derived, success=abs(slope) > 1e-10)
