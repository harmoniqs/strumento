"""Spin Ramsey interferometry — T2* and detuning measurement (Layer 2).

Two pi/2 pulses separated by a free evolution time tau. The spin precesses at the
detuning frequency Delta_f = f_MW - f_L during the free evolution, producing oscillations
that decay with the dephasing time T2*.

Applicable to: LD, Ge hole, S-T (with exchange pulse for pi/2). Very frequent — used
as a "canary" for frequency drift.

**Encoding-dependent semantics of the wait:**

- **LD / Ge hole**: ``hpi()`` is an EDSR-driven π/2 rotation (X-axis); the wait is
  free precession under the Zeeman splitting, producing a Z-rotation at the detuning
  frequency Δf = f_MW − f_Larmor. The oscillation frequency directly gives the detuning.

- **S-T0 (singlet-triplet)**: ``hpi()`` is an exchange pulse (J-driven X-rotation);
  the wait is free precession under the inter-dot magnetic gradient ΔB_z, which provides
  the Z-rotation. The oscillation frequency gives ΔB_z (in MHz), not a drive detuning.
  The wait duration matters differently: longer waits accumulate more ΔB_z phase, so
  T2* here reflects dephasing from charge noise (which modulates J) and nuclear-spin
  noise (which modulates ΔB_z), not the same noise channels as LD.

In both cases the sequence structure ``hpi → wait → hpi → measure`` is correct — the
encoding determines what physical quantity the fitter extracts from the oscillation.

Pure at import: no ``qick`` (the Runner does the qick-facing acquire, lazily).
"""

from __future__ import annotations

from strumento.core.experiment import Experiment
from strumento.core.fitters import FitResult, fit_decaysin_detuned
from strumento.core.pulses import Seq
from strumento.core.result import MeasurementResult
from strumento.core.sweeps import Loop, Sweep
from strumento.packs.spin.experiments._signal import readout_signal
from strumento.packs.spin.readout import PSBReadout


class SpinRamsey(Experiment):
    """Ramsey interferometry: hpi → wait(tau) → hpi → measure.

    Swept: free evolution time tau (µs).
    Measured: P_up(tau) — oscillates at Delta_f and decays with T2*.
    Extracts: T2* (dephasing time), detuning Delta_f.

    The default delay sweep rides the declared loop axis ``"tau"`` (D36).
    ``dot_name`` selects which dot provides the pi/2 pulses.
    """

    def __init__(self, device, *, dot_name: str = "q1", delays: Sweep | None = None,
                 points: int = 101, readout: PSBReadout | None = None, runner=None):
        super().__init__(device, runner=runner)
        self._dot = getattr(self.device, dot_name)
        self.delays = delays if delays is not None else Sweep(start=0.0, stop=10.0, on="tau")
        self.points = points
        self.readout = readout if readout is not None else PSBReadout(threshold=0.5)

    def axes(self) -> tuple:
        if isinstance(self.delays, Sweep) and self.delays.on is not None:
            return (Loop(self.delays.on, self.points),)
        return ()

    def sequence(self) -> tuple[Seq, tuple]:
        seq = Seq().play(self._dot.hpi()).wait(self.delays).play(self._dot.hpi()).measure()
        return seq, ()

    def analyze(self, result: MeasurementResult) -> FitResult:
        return fit_decaysin_detuned(result.xpts, readout_signal(result, self.readout))
