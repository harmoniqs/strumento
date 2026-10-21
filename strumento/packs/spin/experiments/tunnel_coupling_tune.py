"""Tunnel coupling tuning — inter-dot tunnel coupling extraction (Layer 0).

Measures the curvature (anti-crossing gap) of a charge transition as a function of
barrier gate voltage to calibrate the inter-dot tunnel coupling t_c. The coupling
is exponentially sensitive to barrier voltage.

Applicable to: all multi-dot encodings. Frequent recalibration needed.

Standalone DC-sweep class — uses ``lmfit`` directly for its domain-specific exponential
fit model (see ``rf_reflectometry_tune.py`` for the design rationale).

Pure at import: no ``qick``.
"""

from __future__ import annotations

from collections.abc import Callable

import lmfit
import numpy as np

from strumento.core.dc import DCSource
from strumento.core.fitters import FitResult
from strumento.core.result import MeasurementResult


def _lorentzian(x, amp, x0, gamma, offset):
    """Lorentzian lineshape: amp * gamma^2 / ((x - x0)^2 + gamma^2) + offset."""
    return amp * gamma**2 / ((x - x0)**2 + gamma**2) + offset


class TunnelCouplingTune:
    """Measure inter-dot tunnel coupling t_c from the charge anti-crossing width.

    Host-driven DC sweep — NOT an Experiment subclass. The ``DCSource`` handles
    voltage clamping via ``max_v``; no additional safety validation needed here.

    Swept: detuning epsilon across the inter-dot transition at fixed barrier voltage.
    Measured: sensor signal vs detuning — the transition broadens as t_c increases.
    Extracts: t_c from Lorentzian fit to transition width (HWHM ~ 2*t_c).
    Prerequisites: CC-3 (CSD), CC-5 (virtual gates); near a (1,1)-(0,2) transition.
    """

    def __init__(
        self,
        dc: DCSource,
        *,
        barrier_name: str = "b0",
        voltages=np.linspace(-0.5, 0.5, 51),
        probe: Callable[[], float],
        compensated: bool = False,
    ) -> None:
        self.dc = dc
        self.barrier_name = barrier_name
        self.voltages = np.asarray(voltages, dtype=float)
        self.probe = probe
        self.compensated = compensated

    def run(self) -> MeasurementResult:
        """Sweep barrier voltage, record probe signal at each point."""
        setv = self.dc.set_compensated if self.compensated else self.dc.set
        data = np.zeros(len(self.voltages))
        for i, v in enumerate(self.voltages):
            setv(self.barrier_name, float(v))
            data[i] = float(self.probe())
        return MeasurementResult(
            xpts=self.voltages,
            avgi=data.reshape(1, -1),
            avgq=np.zeros((1, len(self.voltages))),
            meta={"axis": self.barrier_name, "dc_snapshot": self.dc.snapshot()},
        )

    def analyze(self, result: MeasurementResult) -> FitResult:
        """Lorentzian fit to extract tunnel coupling from transition linewidth."""
        x = np.asarray(result.xpts, dtype=float)
        signal = np.asarray(result.avgi).ravel()

        # Fit Lorentzian to the charge transition lineshape.
        # The HWHM gamma is related to the tunnel coupling: t_c ~ gamma * lever_arm_factor.
        span = float(x.max() - x.min()) or 1.0
        model = lmfit.Model(_lorentzian)
        pars = model.make_params(
            amp=float(np.max(signal) - np.min(signal)),
            x0=float(x[np.argmax(np.abs(np.gradient(signal)))]),
            gamma=span / 10.0,
            offset=float(np.mean(signal)),
        )
        pars["gamma"].set(min=1e-6)

        out = model.fit(signal, pars, x=x)

        params = {k: float(v.value) for k, v in out.params.items()}
        stderr = {k: (float(v.stderr) if v.stderr is not None else float("nan"))
                  for k, v in out.params.items()}

        # tunnel coupling ~ HWHM in GHz (assuming detuning axis is in meV, lever arm ~ 1)
        gamma = abs(params["gamma"])
        tunnel_coupling_ghz = gamma * 241.8  # meV to GHz: 1 meV ~ 241.8 GHz

        derived = {"tunnel_coupling_ghz": tunnel_coupling_ghz}
        return FitResult(params=params, stderr=stderr, derived=derived, success=bool(out.success))
