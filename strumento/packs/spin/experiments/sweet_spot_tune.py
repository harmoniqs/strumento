"""Sweet spot tuning — charge-noise-insensitive operating point (Layer 2).

Finds the gate voltage where df_L/dV_g = 0 (g-factor sweet spot), maximizing T2* by
making the qubit first-order insensitive to charge noise. The sweet spot is where g'(V) = 0.

Applicable to: Ge/SiGe hole spin qubits (essential); potentially useful for any platform
with gate-voltage-dependent g-factor.

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


class SweetSpotTune(Experiment):
    """Find the charge-noise sweet spot: V_g where df_L/dV_g = 0.

    Swept: free evolution time tau (Ramsey at current gate voltage).
    Measured: P_up(tau) — oscillates at f_L and decays with T2*.
    Extracts: f_L and T2* at the current operating point; sweet spot found from
              the outer gate-voltage sweep (where T2* is maximized / df/dV = 0).
    Prerequisites: g-factor mapped (approximate g(V_g) known); Ramsey works.

    The sequence is a standard Ramsey measurement. The outer gate-voltage sweep is
    host-driven: at each V_g, this experiment extracts f_L and T2*. The sweet spot
    is the V_g where T2* peaks (first-order noise insensitivity).
    """

    def __init__(self, device, *, dot_name: str = "q1", delays: Sweep | None = None,
                 points: int = 101, readout: PSBReadout | None = None, runner=None):
        super().__init__(device, runner=runner)
        self._dot = getattr(self.device, dot_name)
        self.delays = delays if delays is not None else Sweep(start=0.0, stop=20.0, on="tau")
        self.points = points
        self.readout = readout if readout is not None else PSBReadout(threshold=0.5)

    def axes(self) -> tuple:
        if isinstance(self.delays, Sweep) and self.delays.on is not None:
            return (Loop(self.delays.on, self.points),)
        return ()

    def sequence(self) -> tuple[Seq, tuple]:
        # Ramsey measurement at the current gate voltage operating point
        hpi = self._dot.hpi()
        seq = Seq().play(hpi).wait(self.delays).play(hpi).measure()
        return seq, ()

    def analyze(self, result: MeasurementResult) -> FitResult:
        x = np.asarray(result.xpts, dtype=float)
        signal = readout_signal(result, self.readout)

        # Fit Ramsey to get T2* and frequency at this operating point
        fit = fit_decaysin_detuned(x, signal)

        t2_star_us = fit.derived.get("T2", fit.derived.get("tau", 0.0))

        # The sweet spot is where T2* is maximized across the outer V_g sweep.
        # From a single Ramsey, report T2* at this V_g.
        # NOTE: v_sweet_spot is always 0.0 from a single invocation — it requires
        # post-processing across an outer voltage sweep (multiple SweetSpotTune runs at
        # different V_g). The caller finds v_sweet_spot = argmax(T2*(V_g)) from the series.
        v_sweet_spot = 0.0

        params = fit.params
        stderr = fit.stderr
        derived = {"v_sweet_spot": v_sweet_spot, "t2_star_at_ss_us": t2_star_us}
        return FitResult(params=params, stderr=stderr, derived=derived, success=fit.success)
