"""AllXY diagnostic — 21 gate-pair benchmark for transmon single-qubit control (cqed expansion).

Runs all 21 pairs of {I, X, Y, X/2, Y/2} pulses and measures the qubit state after each pair.
An ideal system produces a characteristic staircase pattern (five groups at |g⟩, superposition,
and |e⟩). Deviations diagnose amplitude errors, DRAG miscalibration, frequency offsets, and
decoherence.

The implementation plays all 21 pairs in one program with one readout per pair (21 reads per
shot, no loop axis). This is the hardware-faithful structure: the tProc executes all pairs in
sequence within a single acquisition.

Pure at import: no ``qick`` (the Runner does the qick-facing acquire, lazily).
"""

from __future__ import annotations

import numpy as np

from strumento.core.experiment import Experiment
from strumento.core.fitters import FitResult
from strumento.core.pulses import Seq
from strumento.core.result import MeasurementResult
from strumento.packs.cqed.experiments._common import PassiveReset, scoped_ops
from strumento.packs.cqed.readout import DispersiveIQ

#: The 21 AllXY pulse pairs and their ideal excited-state populations, in the canonical
#: staircase order (five |g>, twelve equator, four |e>). Convention: "I" = identity, "X" = pi_x,
#: "Y" = pi_y, "x" = pi/2 around X, "y" = pi/2 around Y. The ideal is the ACTUAL end population of
#: each pair applied to |g> (e.g. x·x = two pi/2 about X = a full pi -> |e> = 1.0), not an
#: assign-by-position guess.
ALLXY_PAIRS: list[tuple[str, str, float]] = [
    # Group 1 — end in |g> (P_e = 0): net identity / 2pi rotations.
    ("I", "I", 0.0),
    ("X", "X", 0.0),
    ("Y", "Y", 0.0),
    ("X", "Y", 0.0),
    ("Y", "X", 0.0),
    # Group 2 — end on the equator (P_e = 0.5): a net pi/2 rotation.
    ("x", "I", 0.5),
    ("y", "I", 0.5),
    ("x", "y", 0.5),
    ("y", "x", 0.5),
    ("x", "Y", 0.5),
    ("y", "X", 0.5),
    ("X", "y", 0.5),
    ("Y", "x", 0.5),
    ("x", "X", 0.5),
    ("X", "x", 0.5),
    ("y", "Y", 0.5),
    ("Y", "y", 0.5),
    # Group 3 — end in |e> (P_e = 1.0): a net pi rotation.
    ("X", "I", 1.0),
    ("Y", "I", 1.0),
    ("x", "x", 1.0),
    ("y", "y", 1.0),
]


def _gate_pulse(q, code: str):
    """Map a gate code to the corresponding transmon pulse.

    Identity ("I") is a **zero-gain pulse on the same Gaussian envelope** as the real gates — it
    occupies exactly one gate slot in time but applies no rotation. Playing it (rather than
    omitting it) keeps every AllXY pair two gate-lengths long, so the 21 sequences are
    equal-duration and decoherence weights them uniformly (the timing AllXY depends on). This
    needs no knowledge of the gate duration: the envelope carries it.
    """
    if code == "I":
        return q.ge_pi(gain=0.0)
    if code == "X":
        return q.ge_pi()
    if code == "Y":
        return q.ge_pi(phase=90.0)
    if code == "x":
        return q.ge_hpi()
    if code == "y":
        return q.ge_hpi(phase=90.0)
    raise ValueError(f"unknown gate code {code!r}")


class TransmonAllXY(PassiveReset, Experiment):
    """AllXY 21-gate-pair diagnostic for transmon single-qubit calibration quality.

    Plays all 21 gate pairs in one shot (21 reads per acquisition, no loop axis). The
    analyzer compares the measured 21-point pattern against the ideal staircase and reports
    amplitude error, detuning error, and RMS deviation. Identity gates are matched-length
    zero-gain pulses, so all 21 pairs share the same duration.
    """

    def __init__(self, device, *, readout: DispersiveIQ | None = None,
                 reset_delay_us: float | None = None, overrides: dict | None = None, runner=None):
        super().__init__(device, runner=runner)
        self.readout = readout if readout is not None else DispersiveIQ.from_calib(device.calib.readout)
        # Passive reset between the 21 pairs (µs). Defaults to the calibrated readout relax_delay.
        self.reset_delay_us = reset_delay_us
        # Pulse overrides for the gates every pair is built from, scoped to
        # THIS experiment (the device's own ops are untouched).
        self.overrides = dict(overrides) if overrides else {}

    def _ops(self):
        """The transmon ops the pairs play through (with this experiment's overrides)."""
        return scoped_ops(self.device, self.overrides)

    def axes(self) -> tuple:
        return ()

    def sequence(self) -> tuple[Seq, tuple]:
        q = self._ops()
        reset = self._reset_delay()
        seq = Seq()
        for gate1_code, gate2_code, _ in ALLXY_PAIRS:
            # Reset to |g> before every pair. The readouts are projective, so without this the
            # qubit state carries across pairs: the first superposition's measurement leaves it
            # maximally mixed and every subsequent pair averages to 0.5 (the equator/|e> pairs
            # become indistinguishable). Each pair must start from a fresh |g>.
            seq.wait(reset)
            seq.play(_gate_pulse(q, gate1_code))
            seq.play(_gate_pulse(q, gate2_code))
            seq.play(self.device.readout.dispersive_measure())
        return seq, ()

    def analyze(self, result: MeasurementResult) -> FitResult:
        # One averaged read per gate pair; project onto the discrimination axis (rotated I).
        avgi = np.asarray(result.avgi, dtype=float).ravel()
        avgq = np.asarray(result.avgq, dtype=float).ravel()
        iq = np.stack([avgi, avgq], axis=-1)
        signal = self.readout.project(iq).ravel()  # rotated-I readout signal (ADC units)

        ideal = np.array([p[2] for p in ALLXY_PAIRS], dtype=float)
        n = min(len(signal), len(ideal))
        signal, ideal = signal[:n], ideal[:n]

        # The averaged signal is in ADC units, not [0,1]. Normalize to excited-state population via
        # the ideal groups — pairs that end in |g> set the 0 level, pairs that end in |e> set the 1
        # level. Any projection axis is linear in population, so this is angle-robust.
        ground_mask = np.isclose(ideal, 0.0)
        excited_mask = np.isclose(ideal, 1.0)
        ground = signal[ground_mask].mean() if np.any(ground_mask) else float(np.min(signal))
        excited = signal[excited_mask].mean() if np.any(excited_mask) else float(np.max(signal))
        scale = excited - ground
        measured = (signal - ground) / scale if scale != 0.0 else signal - ground

        deviations = measured - ideal
        rms_error = float(np.sqrt(np.mean(deviations**2)))

        # Equator points isolate the single-qubit-gate errors: their MEAN shift is the pi/2
        # amplitude error; their SPREAD (std) is the detuning/phase zig-zag.
        equator = deviations[np.isclose(ideal, 0.5)]
        amplitude_error = float(np.mean(equator)) if equator.size else 0.0
        detuning_error = float(np.std(equator)) if equator.size else 0.0

        return FitResult(
            params={"rms_error": rms_error},
            stderr={"rms_error": 0.0},
            derived={
                "rms_deviation": rms_error,
                "amplitude_error": amplitude_error,
                "detuning_error": detuning_error,
                "populations": measured.tolist(),  # normalized staircase, for the display
            },
            success=True,
        )
