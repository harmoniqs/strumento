"""AllXY calibration — single-qubit gate error diagnostic (Layer 2).

21 two-pulse sequences from all combinations of {I, X, Y, X/2, Y/2} pairs. Deviations
from the ideal pattern (P_up = 0, 0.5, or 1) diagnose amplitude errors, detuning errors,
and DRAG coefficient errors simultaneously.

The implementation plays all 21 pairs in one program with one readout per pair (21 reads
per shot, no loop axis). This is the hardware-faithful structure: the tProc executes all
pairs in sequence within a single acquisition.

Applicable to: LD, Ge hole. Routine fast diagnostic, fully automated.

Pure at import: no ``qick`` (the Runner does the qick-facing acquire, lazily).
"""

from __future__ import annotations

import numpy as np

from strumento.core.experiment import Experiment
from strumento.core.fitters import FitResult
from strumento.core.pulses import Seq
from strumento.core.result import MeasurementResult
from strumento.packs.spin.readout import PSBReadout

# The 21 AllXY pulse pairs and their ideal outcomes.
# Convention: "I" = identity (no pulse), "X" = pi_x, "Y" = pi_y,
# "x" = hpi_x (pi/2 around X), "y" = hpi_y (pi/2 around Y).
# Ideal P_up: 0 for ground-state outcome, 1 for excited, 0.5 for equator.
ALLXY_PAIRS: list[tuple[str, str, float]] = [
    ("I", "I", 0.0),
    ("X", "X", 0.0),
    ("Y", "Y", 0.0),
    ("X", "Y", 0.0),
    ("Y", "X", 0.0),
    ("x", "I", 0.5),
    ("y", "I", 0.5),
    ("x", "x", 0.5),
    ("y", "y", 0.5),
    ("x", "y", 0.5),
    ("y", "x", 0.5),
    ("x", "X", 1.0),
    ("y", "Y", 1.0),
    ("x", "Y", 1.0),
    ("y", "X", 1.0),
    ("X", "I", 1.0),
    ("Y", "I", 1.0),
    ("X", "x", 1.0),
    ("Y", "y", 1.0),
    ("X", "y", 1.0),
    ("Y", "x", 1.0),
]


def _gate_pulse(dot, code: str):
    """Map a gate code to the corresponding spin-qubit pulse (or None for identity)."""
    if code == "I":
        return None
    if code == "X":
        return dot.pi()
    if code == "Y":
        return dot.pi(phase=90.0)
    if code == "x":
        return dot.hpi()
    if code == "y":
        return dot.hpi(phase=90.0)
    raise ValueError(f"unknown gate code {code!r}")


class AllXY(Experiment):
    """AllXY diagnostic: 21 two-pulse sequences testing all gate-error signatures.

    Plays all 21 gate pairs in one shot (21 reads per acquisition, no loop axis). The
    analyzer compares the measured 21-point pattern against the ideal staircase and reports
    amplitude error, detuning error, and RMS deviation.
    """

    def __init__(self, device, *, dot_name: str = "q1",
                 readout: PSBReadout | None = None, runner=None):
        super().__init__(device, runner=runner)
        self._dot = getattr(self.device, dot_name)
        self._dot_name = dot_name
        self.points = 21
        self.readout = readout if readout is not None else PSBReadout(threshold=0.5)

    def axes(self) -> tuple:
        return ()

    def sequence(self) -> tuple[Seq, tuple]:
        seq = Seq()
        for gate1_code, gate2_code, _ in ALLXY_PAIRS:
            g1 = _gate_pulse(self._dot, gate1_code)
            g2 = _gate_pulse(self._dot, gate2_code)
            if g1 is not None:
                seq.play(g1)
            if g2 is not None:
                seq.play(g2)
            seq.measure()
        return seq, ()

    def analyze(self, result: MeasurementResult) -> FitResult:
        # Each of the 21 reads gives one population value
        measured = np.asarray(result.avgi, dtype=float).ravel()
        ideal = np.array([p[2] for p in ALLXY_PAIRS], dtype=float)

        # Truncate or pad to 21 if shape doesn't match (defensive)
        n = min(len(measured), len(ideal))
        deviations = measured[:n] - ideal[:n]
        rms_error = float(np.sqrt(np.mean(deviations**2)))

        # Amplitude error: mean shift of equator points (indices 5-10, ideal = 0.5)
        equator = deviations[5:11] if n >= 11 else deviations
        amplitude_error = float(np.mean(equator))

        # Detuning indicator: asymmetry between x-I and y-I (indices 5 vs 6)
        detuning_error = float(deviations[5] - deviations[6]) if n > 6 else 0.0

        return FitResult(
            params={"rms_error": rms_error},
            stderr={"rms_error": 0.0},
            derived={
                "rms_deviation": rms_error,
                "amplitude_error": amplitude_error,
                "detuning_error": detuning_error,
            },
            success=True,
        )
