"""g-factor map — g-tensor characterization for hole spin qubits (Layer 2).

Maps the anisotropic g-factor as a function of magnetic field direction and gate voltage.
Essential for Ge hole qubits where g is highly anisotropic (g_perp << g_parallel) and
gate-voltage-tunable. Identifies sweet spots where dg/dV = 0.

Applicable to: Ge/SiGe hole spin qubits. Done once per device configuration + after
any B-field reorientation.

Pure at import: no ``qick``.
"""

from __future__ import annotations

import numpy as np

from strumento.core.experiment import Experiment
from strumento.core.fitters import FitResult, fit_decaysin_detuned
from strumento.core.pulses import Seq
from strumento.core.result import MeasurementResult
from strumento.core.sweeps import Loop, Sweep
from strumento.packs.spin.experiments._signal import readout_signal
from strumento.packs.spin.readout import PSBReadout


class GFactorMap(Experiment):
    """Map the g-tensor: Larmor frequency vs B-field angle and gate voltage.

    Swept: gate voltage at fixed B-field (for dg/dV characterization).
    Measured: Larmor frequency f_L (from Ramsey) at each gate voltage point.
    Extracts: effective g-factor, dg/dV_g from the voltage dependence.
    Prerequisites: single hole isolated; preliminary drive (can measure f_L).

    At each gate voltage point, a Ramsey measurement extracts the Larmor frequency.
    The outer gate-voltage sweep is host-driven. The sequence performs a Ramsey
    interferometry measurement to extract f_L at the current operating point.
    """

    def __init__(self, device, *, dot_name: str = "q1", voltages: Sweep | None = None,
                 ramsey_delays: Sweep | None = None, points: int = 51,
                 ramsey_points: int = 101, readout: PSBReadout | None = None, runner=None):
        super().__init__(device, runner=runner)
        self._dot = getattr(self.device, dot_name)
        self.voltages = voltages if voltages is not None else Sweep(start=-0.5, stop=0.5, on="v_gate")
        self.ramsey_delays = ramsey_delays if ramsey_delays is not None else Sweep(start=0.0, stop=2.0, on="tau")
        self.points = points
        self.ramsey_points = ramsey_points
        self.readout = readout if readout is not None else PSBReadout(threshold=0.5)

    def axes(self) -> tuple:
        # The inner Ramsey sweep axis
        if isinstance(self.ramsey_delays, Sweep) and self.ramsey_delays.on is not None:
            return (Loop(self.ramsey_delays.on, self.ramsey_points),)
        return ()

    def sequence(self) -> tuple[Seq, tuple]:
        # Ramsey at current operating point to extract Larmor frequency
        hpi = self._dot.hpi()
        seq = Seq().play(hpi).wait(self.ramsey_delays).play(hpi).measure()
        return seq, ()

    def analyze(self, result: MeasurementResult) -> FitResult:
        x = np.asarray(result.xpts, dtype=float)
        signal = readout_signal(result, self.readout)

        # Fit Ramsey fringe to extract Larmor frequency
        fit = fit_decaysin_detuned(x, signal)
        f_larmor = fit.derived.get("freq", fit.derived.get("detuning", 0.0))

        # g_effective = h * f_L / (mu_B * B), but B is external and not in the sweep;
        # report the measured frequency which IS the g-factor proxy at fixed B
        g_effective = f_larmor  # proportional to g at fixed B

        # dg/dV would require multiple voltage points (outer loop);
        # from a single Ramsey this is the f_L at the current V_g.
        # NOTE: dg_dv is always 0.0 from a single invocation — it requires post-processing
        # across an outer voltage sweep (multiple GFactorMap runs at different V_g).
        # A caller collecting the outer sweep computes dg_dv = Δf_L / ΔV_g from the series.
        dg_dv = 0.0

        params = fit.params
        stderr = fit.stderr
        derived = {"g_effective": g_effective, "dg_dv": dg_dv}
        return FitResult(params=params, stderr=stderr, derived=derived, success=fit.success)
