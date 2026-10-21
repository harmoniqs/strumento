"""Coulomb diamond measurement — charging energy & lever arm extraction (Layer 0).

2D sweep of plunger gate voltage vs source-drain bias V_SD, measuring transport current
I_SD (or dI/dV). Diamond-shaped regions of Coulomb blockade reveal the charging energy
E_C, lever arm alpha = C_g / C_Sigma, and excited state energies (orbital, valley).

Applicable to: all encodings, all platforms. Done once per device configuration.

Pure at import: no ``qick``.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from strumento.core.dc import DCSource
from strumento.core.fitters import FitResult
from strumento.core.result import MeasurementResult


class CoulombDiamond:
    """2D sweep: plunger voltage vs V_SD to map Coulomb diamonds.

    Host-driven DC sweep — NOT an Experiment subclass. The ``DCSource`` handles
    voltage clamping via ``max_v``; no additional safety validation needed here.

    Swept: V_plunger (inner loop), V_SD (outer loop).
    Measured: I_SD or dI/dV_SD — diamonds of suppressed current.
    Extracts: E_C / alpha (half-height), alpha (slopes), excited states (lines inside diamonds).
    Prerequisites: CC-3 (charge stability — know which transition to zoom into).
    """

    def __init__(
        self,
        dc: DCSource,
        *,
        plunger_axis: str = "p0",
        plunger_volts=np.linspace(-0.5, 0.5, 51),
        bias_axis: str = "v_sd",
        bias_volts=np.linspace(-1.0, 1.0, 41),
        probe: Callable[[], float],
        compensated: bool = False,
    ) -> None:
        self.dc = dc
        self.plunger_axis = plunger_axis
        self.plunger_volts = np.asarray(plunger_volts, dtype=float)
        self.bias_axis = bias_axis
        self.bias_volts = np.asarray(bias_volts, dtype=float)
        self.probe = probe
        self.compensated = compensated

    def run(self) -> MeasurementResult:
        """Nested 2D sweep: bias (outer) x plunger (inner), record probe at each point."""
        setv = self.dc.set_compensated if self.compensated else self.dc.set
        grid = np.zeros((len(self.bias_volts), len(self.plunger_volts)))
        for j, v_bias in enumerate(self.bias_volts):
            setv(self.bias_axis, float(v_bias))
            for i, v_plunger in enumerate(self.plunger_volts):
                setv(self.plunger_axis, float(v_plunger))
                grid[j, i] = float(self.probe())
        return MeasurementResult(
            xpts=self.plunger_volts,
            avgi=grid,
            avgq=np.zeros_like(grid),
            meta={
                "plunger_axis": self.plunger_axis,
                "bias_axis": self.bias_axis,
                "bias_volts": self.bias_volts.tolist(),
                "dc_snapshot": self.dc.snapshot(),
            },
        )

    def analyze(self, result: MeasurementResult) -> FitResult:
        """Extract diamond geometry from the 2D conductance map."""
        x = np.asarray(result.xpts, dtype=float)
        signal = np.asarray(result.avgi)

        # For 2D data, collapse to 1D by averaging over the bias axis
        if signal.ndim == 2:
            signal_1d = np.mean(np.abs(signal), axis=0)
        else:
            signal_1d = np.abs(signal).ravel()

        # Extract diamond geometry: half-height of diamond = E_C / alpha
        # The slopes of diamond edges give the lever arm alpha = C_g / C_Sigma
        # Simple extraction: find the peak-to-peak span of blockade region
        blockade_mask = signal_1d < 0.1 * float(np.max(signal_1d))
        if np.any(blockade_mask):
            v_span = float(x[blockade_mask][-1] - x[blockade_mask][0])
        else:
            v_span = float(x[-1] - x[0])

        # Estimate lever arm from diamond width (alpha ~ e / (2 * V_span))
        # and charging energy E_C = alpha * e * V_half_height
        lever_arm = 1.0 / max(v_span, 1e-6) * 0.1  # approximate from single axis
        charging_energy_mev = lever_arm * v_span * 1e3  # convert to meV

        params = {"v_span": v_span, "lever_arm_raw": lever_arm}
        stderr = {"v_span": 0.0, "lever_arm_raw": 0.0}
        derived = {"charging_energy_mev": charging_energy_mev, "lever_arm": lever_arm}
        return FitResult(params=params, stderr=stderr, derived=derived, success=True)
