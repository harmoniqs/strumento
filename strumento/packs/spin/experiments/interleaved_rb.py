"""Interleaved randomized benchmarking — per-gate fidelity (Layer 4).

Same as standard RB but with a specific gate G interleaved between each random Clifford.
The ratio of the interleaved decay constant to the reference RB decay isolates the
error of gate G specifically: F_G = 1 - (1 - p_int/p_ref)(d-1)/d.

The implementation builds one program with ``points`` interleaved-RB blocks (one per depth),
each followed by a readout (multi-read, no loop axis). Each block interleaves the target gate
between random Cliffords and appends the exact inverse via the Cayley table.

Applicable to: all encodings. Used to characterize individual gates after calibration.

Pure at import: no ``qick`` (the Runner does the qick-facing acquire, lazily).
"""

from __future__ import annotations

import numpy as np

from strumento.core.experiment import Experiment
from strumento.core.fitters import FitResult, fit_exp
from strumento.core.pulses import Seq
from strumento.core.result import MeasurementResult
from strumento.packs.spin.experiments.randomized_benchmarking import (
    CAYLEY_TABLE,
    INVERSE_TABLE,
    N_CLIFFORDS,
    _play_clifford,
)
from strumento.packs.spin.readout import PSBReadout

# Gate name → Clifford index mapping for the interleaved gate.
# These are the gates a user might want to characterize individually.
_GATE_TO_CLIFFORD: dict[str, int] = {
    "pi": 1,      # C1: X (π around X)
    "pi_y": 2,    # C2: Y (π around Y)
    "hpi": 3,     # C3: X/2 (π/2 around X)
    "hpi_y": 5,   # C5: Y/2 (π/2 around Y)
}


def _build_interleaved_block(dot, gate_cliff_idx: int, depth: int,
                             rng: np.random.Generator) -> Seq:
    """Build a random Clifford sequence with gate G interleaved, with exact inversion.

    For a sequence of depth m, the structure is:
        C_1 → G → C_2 → G → ... → C_m → G → C_inv

    The accumulated Clifford tracks both the random Cliffords AND the interleaved gate,
    and the inverse is computed exactly via the Cayley table.
    """
    seq = Seq()
    accumulated = 0  # Start at identity (C0)

    for _ in range(depth):
        # Random Clifford
        cliff_idx = int(rng.integers(0, N_CLIFFORDS))
        _play_clifford(dot, cliff_idx, seq)
        accumulated = int(CAYLEY_TABLE[cliff_idx, accumulated])

        # Interleaved gate G
        _play_clifford(dot, gate_cliff_idx, seq)
        accumulated = int(CAYLEY_TABLE[gate_cliff_idx, accumulated])

    # Append the exact inverse of the accumulated Clifford
    inv_idx = int(INVERSE_TABLE[accumulated])
    _play_clifford(dot, inv_idx, seq)

    return seq


class InterleavedRB(Experiment):
    """Interleaved randomized benchmarking for per-gate fidelity.

    Builds ``points`` interleaved-RB blocks (one per depth) in a single program, with one
    readout per block (multi-read, no loop axis). Each block interleaves gate G between
    random Cliffords and appends the exact inverse.

    The analyzer fits the interleaved decay and, if ``p_reference`` is provided (from a
    prior standard RB run), computes the gate-specific fidelity.

    ``gate`` names the interleaved gate: "pi", "pi_y", "hpi", or "hpi_y".
    ``p_reference`` is the depolarizing parameter from standard RB (needed for F_G).
    ``seed`` ensures reproducibility of the random sequences.
    """

    def __init__(self, device, *, dot_name: str = "q1", gate: str = "pi",
                 depth_start: int = 1, depth_stop: int = 100,
                 points: int = 10, seed: int = 42, p_reference: float | None = None,
                 readout: PSBReadout | None = None, runner=None):
        super().__init__(device, runner=runner)
        self._dot = getattr(self.device, dot_name)
        self._dot_name = dot_name
        self.gate = gate
        self._gate_cliff_idx = _GATE_TO_CLIFFORD.get(gate, 1)  # default to pi
        self.depth_start = depth_start
        self.depth_stop = depth_stop
        self.points = points
        self.seed = seed
        self.p_reference = p_reference
        self.readout = readout if readout is not None else PSBReadout(threshold=0.5)
        # Pre-compute the depth schedule
        self._depths = np.linspace(depth_start, depth_stop, points, dtype=int)

    def axes(self) -> tuple:
        return ()

    def sequence(self) -> tuple[Seq, tuple]:
        rng = np.random.default_rng(self.seed)
        seq = Seq()
        for depth in self._depths:
            block = _build_interleaved_block(
                self._dot, self._gate_cliff_idx, int(depth), rng
            )
            seq += block
            seq.measure()
        return seq, ()

    def analyze(self, result: MeasurementResult) -> FitResult:
        # Each read gives the survival probability at that depth
        measured = np.asarray(result.avgi, dtype=float).ravel()

        # Use result.xpts when provided (synthetic/unit-test path), else internal schedule
        if result.xpts is not None and len(np.asarray(result.xpts).ravel()) > 0:
            xpts = np.asarray(result.xpts, dtype=float).ravel()
        else:
            xpts = self._depths.astype(float)

        # Truncate if shapes don't match (defensive)
        n = min(len(measured), len(xpts))

        # Fit the interleaved decay: P_0(m) = A·exp(-m/tau) + B
        fit = fit_exp(xpts[:n], measured[:n])

        tau = fit.derived.get("T1", 1.0)
        p_int = float(np.exp(-1.0 / tau)) if tau > 0 else 0.0
        d = 2  # single qubit

        # Compute gate-specific fidelity if reference p is available
        if self.p_reference is not None and self.p_reference > 0:
            p_ratio = p_int / self.p_reference
            gate_fidelity = 1.0 - (1.0 - p_ratio) * (d - 1) / d
        else:
            p_ratio = p_int
            gate_fidelity = 1.0 - (1.0 - p_int) * (d - 1) / d

        fit.derived["p_interleaved"] = p_int
        fit.derived["p_reference"] = self.p_reference if self.p_reference is not None else float("nan")
        fit.derived["gate_fidelity"] = gate_fidelity
        fit.derived["gate"] = self.gate
        return fit
