"""Pinch-off curve — gate threshold voltage extraction (Layer 0: electrostatic bring-up).

Sweeps a single gate voltage from accumulation toward depletion and measures the channel
conductance (or sensor current). Extracts the threshold voltage V_th where the channel
pinches off (G < G_min). Used to verify gate functionality after cooldown and identify
dead gates.

Applicable to: all encodings, all platforms. Done once per cooldown.

Pure at import: no ``qick``.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from strumento.core.dc import DCSource
from strumento.core.fitters import FitResult
from strumento.core.result import MeasurementResult


class PinchOff:
    """Sweep a gate voltage and extract the pinch-off threshold.

    Host-driven DC sweep — NOT an Experiment subclass. The ``DCSource`` handles
    voltage clamping via ``max_v``; no additional safety validation needed here.

    Swept: single gate voltage (plunger or barrier) from open to pinched.
    Measured: channel conductance G = I_SD / V_SD (or sensor current via probe).
    Extracts: threshold voltage V_th where G drops below ``cutoff_fraction`` of peak.
    Prerequisites: cooldown & wiring check (Layer 0.1).
    """

    def __init__(
        self,
        dc: DCSource,
        *,
        gate_name: str = "p0",
        voltages=np.linspace(0, -1.5, 51),
        probe: Callable[[], float],
        compensated: bool = False,
        cutoff_fraction: float = 0.1,
    ) -> None:
        self.dc = dc
        self.gate_name = gate_name
        self.voltages = np.asarray(voltages, dtype=float)
        self.probe = probe
        self.compensated = compensated
        self.cutoff_fraction = cutoff_fraction

    def run(self) -> MeasurementResult:
        """Sweep the gate and record probe signal at each voltage step."""
        setv = self.dc.set_compensated if self.compensated else self.dc.set
        data = np.zeros(len(self.voltages))
        for i, v in enumerate(self.voltages):
            setv(self.gate_name, float(v))
            data[i] = float(self.probe())
        return MeasurementResult(
            xpts=self.voltages,
            avgi=data.reshape(1, -1),
            avgq=np.zeros((1, len(self.voltages))),
            meta={"axis": self.gate_name, "dc_snapshot": self.dc.snapshot()},
        )

    def analyze(self, result: MeasurementResult) -> FitResult:
        """Find threshold where signal drops below cutoff_fraction of max."""
        x = np.asarray(result.xpts, dtype=float)
        signal = np.asarray(result.avgi).ravel()

        # Find threshold: first point where signal drops below cutoff_fraction of max
        g_max = float(np.max(np.abs(signal)))
        cutoff = self.cutoff_fraction * g_max
        below = np.where(np.abs(signal) < cutoff)[0]
        v_threshold = float(x[below[0]]) if below.size > 0 else float(x[-1])

        params = {"g_max": g_max, "cutoff": cutoff}
        stderr = {"g_max": 0.0, "cutoff": 0.0}
        derived = {"v_threshold": v_threshold}
        return FitResult(params=params, stderr=stderr, derived=derived, success=below.size > 0)
