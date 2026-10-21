"""Charge sensor tuning — QPC/SET operating point optimization (Layer 1).

Sweeps the sensor gate voltage to find the steepest transconductance (maximum sensitivity
to nearby charge transitions). For QPC: finds the half-plateau at G ~ 0.5 e^2/h. For SET:
finds the steepest slope of a Coulomb oscillation peak.

Applicable to: all encodings using a dedicated charge sensor. Frequent recalibration
(every few hours; after charge jumps).

Pure at import: no ``qick``.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from strumento.core.dc import DCSource
from strumento.core.fitters import FitResult
from strumento.core.result import MeasurementResult


class SensorTune:
    """Find the optimal charge sensor operating point (max transconductance).

    Host-driven DC sweep — NOT an Experiment subclass. The ``DCSource`` handles
    voltage clamping via ``max_v``; no additional safety validation needed here.

    Swept: sensor gate voltage V_QPC or V_SET_plunger.
    Measured: conductance G (QPC) or current I_SET (SET) via probe.
    Extracts: V* = argmax |dG/dV| (the point of maximum sensitivity).
    Prerequisites: pinch-off confirms sensor gates functional.
    """

    def __init__(
        self,
        dc: DCSource,
        *,
        sensor_name: str = "sensor",
        voltages=np.linspace(-1.0, 0.0, 51),
        probe: Callable[[], float],
        compensated: bool = False,
    ) -> None:
        self.dc = dc
        self.sensor_name = sensor_name
        self.voltages = np.asarray(voltages, dtype=float)
        self.probe = probe
        self.compensated = compensated

    def run(self) -> MeasurementResult:
        """Sweep sensor gate voltage, record probe signal at each point."""
        setv = self.dc.set_compensated if self.compensated else self.dc.set
        data = np.zeros(len(self.voltages))
        for i, v in enumerate(self.voltages):
            setv(self.sensor_name, float(v))
            data[i] = float(self.probe())
        return MeasurementResult(
            xpts=self.voltages,
            avgi=data.reshape(1, -1),
            avgq=np.zeros((1, len(self.voltages))),
            meta={"axis": self.sensor_name, "dc_snapshot": self.dc.snapshot()},
        )

    def analyze(self, result: MeasurementResult) -> FitResult:
        """Find peak transconductance dG/dV — the optimal sensor operating point."""
        x = np.asarray(result.xpts, dtype=float)
        signal = np.asarray(result.avgi).ravel()

        # Numerical derivative: transconductance dG/dV
        dg_dv = np.gradient(signal, x)
        abs_dg = np.abs(dg_dv)
        peak_idx = int(np.argmax(abs_dg))

        v_optimal = float(x[peak_idx])
        transconductance = float(abs_dg[peak_idx])

        params = {"peak_idx": float(peak_idx), "g_at_optimal": float(signal[peak_idx])}
        stderr = {"peak_idx": 0.0, "g_at_optimal": 0.0}
        derived = {"v_optimal": v_optimal, "transconductance": transconductance}
        return FitResult(params=params, stderr=stderr, derived=derived, success=transconductance > 0)
