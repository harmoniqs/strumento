"""Dispersive gate sensing calibration — spin-dependent C_Q shift (Layer 1).

Calibrates the dispersive gate sensing readout: maps the spin-state-dependent quantum
capacitance shift to the RF phase/amplitude signal, and optimizes the operating point
for maximum SNR.

Applicable to: Ge hole qubits, Si-MOS with gate reflectometry. After any gate
reconfiguration.

Pure at import: no ``qick``.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from strumento.core.dc import DCSource
from strumento.core.fitters import FitResult
from strumento.core.result import MeasurementResult


class DispersiveGateCalibration:
    """Calibrate spin-dependent dispersive shift for gate-based readout.

    Host-driven DC sweep — NOT an Experiment subclass. The ``DCSource`` handles
    voltage clamping via ``max_v``; no additional safety validation needed here.

    Swept: gate voltage (detuning) across charge transition to find max C_Q shift.
    Measured: RF phase shift Delta_phi conditioned on prepared spin state (via probe).
    Extracts: optimal operating point epsilon*, dispersive shift Delta_C_Q.
    Prerequisites: RF reflectometry tuned (CC-2); charge transitions identified.
    """

    def __init__(
        self,
        dc: DCSource,
        *,
        gate_name: str = "p0",
        voltages=np.linspace(-0.3, 0.3, 51),
        probe: Callable[[], float],
        compensated: bool = False,
    ) -> None:
        self.dc = dc
        self.gate_name = gate_name
        self.voltages = np.asarray(voltages, dtype=float)
        self.probe = probe
        self.compensated = compensated

    def run(self) -> MeasurementResult:
        """Sweep gate detuning, record dispersive signal at each point."""
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
        """Find the detuning where the dispersive shift (peak derivative) is maximized."""
        x = np.asarray(result.xpts, dtype=float)
        signal = np.asarray(result.avgi).ravel()

        # The dispersive shift manifests as a steep slope in the RF signal vs detuning.
        # Find the point of maximum derivative (steepest signal change).
        dsignal = np.gradient(signal, x)
        abs_ds = np.abs(dsignal)
        peak_idx = int(np.argmax(abs_ds))

        epsilon_optimal = float(x[peak_idx])
        # Delta_C_Q is proportional to the peak derivative (in aF units, approximate)
        delta_cq_af = float(abs_ds[peak_idx])

        params = {"peak_idx": float(peak_idx), "signal_at_optimal": float(signal[peak_idx])}
        stderr = {"peak_idx": 0.0, "signal_at_optimal": 0.0}
        derived = {"epsilon_optimal": epsilon_optimal, "delta_cq_af": delta_cq_af}
        return FitResult(params=params, stderr=stderr, derived=derived, success=delta_cq_af > 0)
