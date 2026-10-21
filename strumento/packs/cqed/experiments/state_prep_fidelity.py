"""State preparation fidelity experiment (cqed expansion).

Verify the preparation fidelity of target cavity states using parity-based measurement.
A displacement followed by parity mapping measures the Wigner function along the real axis;
for known target states (Fock, coherent) the fidelity is extracted from the parity signal.

Pure at import: no ``qick`` (the Runner does the qick-facing acquire, lazily).
"""

from __future__ import annotations

import numpy as np

from strumento.core.experiment import Experiment
from strumento.core.fitters import FitResult
from strumento.core.pulses import Seq
from strumento.core.result import MeasurementResult
from strumento.core.sweeps import Loop, Sweep
from strumento.packs.cqed.experiments._signal import readout_signal
from strumento.packs.cqed.library.composite import parity
from strumento.packs.cqed.library.targets import resolve_mode
from strumento.packs.cqed.readout import DispersiveIQ


class StatePrepFidelity(Experiment):
    """Measure state-preparation fidelity via parity or Wigner reconstruction.

    The default method (``"parity"``) displaces by swept beta along the real axis and reads
    parity — a 1D Wigner slice. The fidelity is estimated from the parity signal: for a
    vacuum or Fock(1) target, the Wigner function at the origin gives a direct fidelity bound.

    For the ``"parity"`` method the sweep axis is ``"beta_re"``; for a single-point parity
    check (no displacement scan) set ``points=1``.

    ``prepared_state`` accepts an optional ``Seq`` that replaces the default ``prepare(dev, ...)``
    call — use this to verify an OC-imported pulse (from ``from_solution`` or ``load_oc_pulse``)
    through the standard fidelity measurement without subclassing.
    """

    def __init__(self, device, *, betas: Sweep | None = None, points: int = 21,
                 mode_name: str | None = None, target: str = "fock1",
                 method: str = "parity", prepared_state: Seq | None = None,
                 readout: DispersiveIQ | None = None, runner=None):
        super().__init__(device, runner=runner)
        mode_name = resolve_mode(device, mode_name, purpose=type(self).__name__)
        self.betas = betas if betas is not None else Sweep(start=0.0, stop=0.25, on="beta_re")
        self.points = points
        self.mode_name = mode_name
        self.target = target
        self.method = method
        self.prepared_state = prepared_state
        self.readout = readout if readout is not None else DispersiveIQ.from_calib(device.calib.readout)

    def axes(self) -> tuple:
        if isinstance(self.betas, Sweep) and self.betas.on is not None:
            return (Loop(self.betas.on, self.points),)
        return ()

    def sequence(self) -> tuple[Seq, tuple]:
        dev = self.device
        mode = getattr(dev, self.mode_name)
        # Use the provided prepared_state if given, else the default displacement
        if self.prepared_state is not None:
            prep = self.prepared_state
        else:
            # The same beta-vs-gain unit mismatch WignerTomography had --
            # betas is a physical displacement amplitude, not a lab-native int code. Route
            # through the calibrated gain-per-alpha (displacement_cal), matching
            # ModeOps.displace_alpha() and WignerTomography's fix.
            mode_cal = dev.calib.modes.get(self.mode_name)
            if mode_cal is None or mode_cal.displacement_cal is None:
                raise KeyError(
                    f"mode {self.mode_name!r} has no displacement_cal — "
                    "run DisplacementCalibration to calibrate the gain-per-alpha"
                )
            slope = mode_cal.displacement_cal
            gains = self.betas.mapped(lambda beta: slope * beta, inverse=lambda gain: gain / slope)
            prep = Seq().play(mode.displace(gain=gains))
        # #100: parity() drives the readout and measures internally by default now -- chaining
        # our own .measure() here would double-measure.
        seq = prep + parity(dev, self.mode_name)
        return seq, ()

    def analyze(self, result: MeasurementResult) -> FitResult:
        signal = readout_signal(result, self.readout)
        xpts = np.asarray(result.xpts, dtype=float)

        # The Wigner function at the origin (beta=0) gives W(0) = (2/pi) * <Parity>.
        # For Fock(1): W(0) = -2/pi (negative); for vacuum: W(0) = +2/pi.
        # Fidelity bound from the origin parity value.
        mid_idx = len(xpts) // 2
        w_origin = float(signal[mid_idx]) if signal.size > 0 else 0.0

        # Crude fidelity estimate: |W(0)| * pi/2 (saturates at 1 for pure states)
        fidelity_estimate = min(abs(w_origin) * np.pi / 2.0, 1.0)

        return FitResult(
            params={},
            stderr={},
            derived={
                "w_origin": w_origin,
                "fidelity_estimate": fidelity_estimate,
                "w_max": float(signal.max()) if signal.size > 0 else 0.0,
                "w_min": float(signal.min()) if signal.size > 0 else 0.0,
            },
            success=True,
        )
