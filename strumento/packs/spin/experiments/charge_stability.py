"""Charge-stability map — a 2D slow-DC gate sweep with a fast sensor readout at each point.

The slow axes live on an external :class:`~strumento.core.dc.DCSource` (QICK does not own
them), so this is a **host-driven** experiment: the host steps the two gate voltages and takes a
fast charge-sensor point at each setting. It therefore composes a probe function rather than
subclassing the program-building :class:`~strumento.core.experiment.Experiment`.

Pure: no ``qick`` (the probe callable owns any hardware access).
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from strumento.core.dc import DCSource
from strumento.core.result import MeasurementResult


class ChargeStability:
    """Sweep ``x_axis`` × ``y_axis`` on a DCSource; record the sensor probe at each point.

    ``probe()`` returns the charge-sensor signal for the *current* DC state — on hardware it
    acquires a fast readout; in tests it's any callable. ``compensated=True`` uses lever-arm
    compensated moves.
    """

    def __init__(
        self,
        dc: DCSource,
        *,
        x_axis: str,
        x_volts,
        y_axis: str,
        y_volts,
        probe: Callable[[], float],
        compensated: bool = False,
    ) -> None:
        self.dc = dc
        self.x_axis, self.x_volts = x_axis, np.asarray(x_volts, dtype=float)
        self.y_axis, self.y_volts = y_axis, np.asarray(y_volts, dtype=float)
        self.probe = probe
        self.compensated = compensated

    def run(self) -> MeasurementResult:
        setv = self.dc.set_compensated if self.compensated else self.dc.set
        grid = np.zeros((len(self.y_volts), len(self.x_volts)))
        for j, vy in enumerate(self.y_volts):
            setv(self.y_axis, float(vy))
            for i, vx in enumerate(self.x_volts):
                setv(self.x_axis, float(vx))
                grid[j, i] = float(self.probe())
        return MeasurementResult(
            xpts=self.x_volts,
            avgi=grid,
            avgq=np.zeros_like(grid),
            meta={
                "y_axis": self.y_axis,
                "ypts": self.y_volts.tolist(),
                "x_axis": self.x_axis,
                "dc_snapshot": self.dc.snapshot(),
            },
        )
