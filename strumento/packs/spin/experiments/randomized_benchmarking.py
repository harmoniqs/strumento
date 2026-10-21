"""Randomized benchmarking — average Clifford gate fidelity (Layer 4).

Applies random sequences of Clifford gates of increasing length m, followed by an
inversion gate. The decay of the return probability P_0(m) = A·p^m + B gives the
average gate fidelity F = 1 - (1-p)(d-1)/d where d=2 for a single qubit.

The implementation builds one program with ``points`` random-Clifford blocks (one per
depth), each followed by a readout (multi-read, no loop axis). This is hardware-faithful:
the tProc executes all depths in one shot. Statistical averaging over random seeds
is handled by running the experiment multiple times (different ``seed`` values).

The Clifford group is tracked via a Cayley table computed from the 24 single-qubit
Clifford unitaries. The inversion gate is computed exactly (not approximated) by
looking up the inverse of the accumulated Clifford and decomposing it into native gates.

Applicable to: all encodings with a calibrated gate set.

Pure at import: no ``qick`` (the Runner does the qick-facing acquire, lazily).
"""

from __future__ import annotations

import numpy as np

from strumento.core.experiment import Experiment
from strumento.core.fitters import FitResult, fit_exp
from strumento.core.pulses import Seq
from strumento.core.result import MeasurementResult
from strumento.packs.spin.readout import PSBReadout

# =============================================================================
# Single-qubit Clifford group — 24 elements, Cayley table, decompositions
# =============================================================================

# Native gate unitaries (the physical gates available on a spin qubit via EDSR).
# Convention: X rotation by angle θ = exp(-i θ/2 σ_x), etc.
_I2 = np.eye(2, dtype=complex)
_X = np.array([[0, 1], [1, 0]], dtype=complex)       # σ_x (Pauli X)
_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)    # σ_y (Pauli Y)


def _rot(axis: np.ndarray, angle: float) -> np.ndarray:
    """Rotation unitary: exp(-i angle/2 * axis)."""
    return np.cos(angle / 2) * _I2 - 1j * np.sin(angle / 2) * axis


# The 7 native gates (including identity):
#   0: I,  1: X(π),  2: Y(π),  3: X(π/2),  4: X(-π/2),  5: Y(π/2),  6: Y(-π/2)
_NATIVE_GATES = [
    _I2,                          # 0: I
    _rot(_X, np.pi),              # 1: X (pi)
    _rot(_Y, np.pi),              # 2: Y (pi)
    _rot(_X, np.pi / 2),          # 3: X/2
    _rot(_X, -np.pi / 2),         # 4: -X/2
    _rot(_Y, np.pi / 2),          # 5: Y/2
    _rot(_Y, -np.pi / 2),         # 6: -Y/2
]

# Build the 24 single-qubit Clifford unitaries by composing native gates.
# Each Clifford is specified as a tuple of native-gate indices applied left to right.
_CLIFFORD_DECOMPOSITIONS: list[tuple[int, ...]] = [
    (0,),           # C0:  I
    (1,),           # C1:  X
    (2,),           # C2:  Y
    (3,),           # C3:  X/2
    (4,),           # C4:  -X/2
    (5,),           # C5:  Y/2
    (6,),           # C6:  -Y/2
    (1, 2),         # C7:  X · Y
    (1, 5),         # C8:  X · Y/2
    (1, 6),         # C9:  X · -Y/2
    (2, 3),         # C10: Y · X/2
    (2, 4),         # C11: Y · -X/2
    (3, 5),         # C12: X/2 · Y/2
    (3, 6),         # C13: X/2 · -Y/2
    (4, 5),         # C14: -X/2 · Y/2
    (4, 6),         # C15: -X/2 · -Y/2
    (5, 3),         # C16: Y/2 · X/2
    (5, 4),         # C17: Y/2 · -X/2
    (6, 3),         # C18: -Y/2 · X/2
    (6, 4),         # C19: -Y/2 · -X/2
    (3, 5, 3),      # C20: X/2 · Y/2 · X/2
    (3, 5, 4),      # C21: X/2 · Y/2 · -X/2
    (3, 6, 3),      # C22: X/2 · -Y/2 · X/2
    (3, 6, 4),      # C23: X/2 · -Y/2 · -X/2
]


def _compute_clifford_tables():
    """Compute the 24 Clifford unitaries, Cayley table, and inverse table."""
    n = len(_CLIFFORD_DECOMPOSITIONS)

    # Build unitaries from decompositions
    unitaries = []
    for decomp in _CLIFFORD_DECOMPOSITIONS:
        u = _I2.copy()
        for gate_idx in decomp:
            u = _NATIVE_GATES[gate_idx] @ u
        unitaries.append(u)

    # Build Cayley table: cayley[i][j] = index of U_i @ U_j
    cayley = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(n):
            product = unitaries[i] @ unitaries[j]
            for k in range(n):
                overlap = abs(np.trace(unitaries[k].conj().T @ product))
                if overlap > 1.99:
                    cayley[i, j] = k
                    break

    # Build inverse table: inverse[i] = index of U_i†
    inverse = np.zeros(n, dtype=int)
    for i in range(n):
        inv_u = unitaries[i].conj().T
        for k in range(n):
            overlap = abs(np.trace(unitaries[k].conj().T @ inv_u))
            if overlap > 1.99:
                inverse[i] = k
                break

    return cayley, inverse


# Compute tables at module load (pure numpy, fast — ~1ms)
CAYLEY_TABLE, INVERSE_TABLE = _compute_clifford_tables()

#: Number of Cliffords in the single-qubit group.
N_CLIFFORDS: int = 24


# =============================================================================
# Sequence builder
# =============================================================================


def _play_clifford(dot, clifford_idx: int, seq: Seq) -> None:
    """Append the native-gate decomposition of Clifford ``clifford_idx`` to ``seq``."""
    decomp = _CLIFFORD_DECOMPOSITIONS[clifford_idx]
    for gate_idx in decomp:
        if gate_idx == 0:
            continue  # Identity: no pulse
        elif gate_idx == 1:
            seq.play(dot.pi())               # X (π at 0°)
        elif gate_idx == 2:
            seq.play(dot.pi(phase=90.0))     # Y (π at 90°)
        elif gate_idx == 3:
            seq.play(dot.hpi())              # X/2 (π/2 at 0°)
        elif gate_idx == 4:
            seq.play(dot.hpi(phase=180.0))   # -X/2 (π/2 at 180°)
        elif gate_idx == 5:
            seq.play(dot.hpi(phase=90.0))    # Y/2 (π/2 at 90°)
        elif gate_idx == 6:
            seq.play(dot.hpi(phase=270.0))   # -Y/2 (π/2 at 270°)


def _build_rb_block(dot, depth: int, rng: np.random.Generator) -> Seq:
    """Build a random Clifford sequence of ``depth`` gates with exact inversion.

    Draws ``depth`` random Cliffords, tracks the accumulated Clifford via the Cayley
    table, then appends the inverse Clifford. The sequence should return the qubit
    to ground (survival probability = 1 for a perfect gate set).
    """
    seq = Seq()
    accumulated = 0  # Start at identity (C0)

    for _ in range(depth):
        cliff_idx = int(rng.integers(0, N_CLIFFORDS))
        _play_clifford(dot, cliff_idx, seq)
        # Update accumulated: new = C_cliff @ C_accumulated
        accumulated = int(CAYLEY_TABLE[cliff_idx, accumulated])

    # Append the exact inverse of the accumulated Clifford
    inv_idx = int(INVERSE_TABLE[accumulated])
    _play_clifford(dot, inv_idx, seq)

    return seq


# =============================================================================
# Experiment class
# =============================================================================


class RandomizedBenchmarking(Experiment):
    """Single-qubit Clifford randomized benchmarking.

    Builds ``points`` random-Clifford blocks (one per depth) in a single program, with one
    readout per block (multi-read, no loop axis). The depths are linearly spaced from
    ``depth_start`` to ``depth_stop``. Each block draws random Cliffords, tracks the
    accumulated group element via a Cayley table, and appends the exact inverse gate.

    The analyzer fits the survival probability decay P_0(m) = A·exp(-m/τ) + B and derives
    the depolarizing parameter p, average gate fidelity F = 1 - (1-p)/2, and error per
    Clifford.

    Statistical averaging: run with different ``seed`` values and average the fitted
    parameters; each invocation uses one random sequence per depth.
    """

    def __init__(self, device, *, dot_name: str = "q1",
                 depth_start: int = 1, depth_stop: int = 100,
                 points: int = 10, seed: int = 42,
                 readout: PSBReadout | None = None, runner=None):
        super().__init__(device, runner=runner)
        self._dot = getattr(self.device, dot_name)
        self._dot_name = dot_name
        self.depth_start = depth_start
        self.depth_stop = depth_stop
        self.points = points
        self.seed = seed
        self.readout = readout if readout is not None else PSBReadout(threshold=0.5)
        # Pre-compute the depth schedule
        self._depths = np.linspace(depth_start, depth_stop, points, dtype=int)

    def axes(self) -> tuple:
        return ()

    def sequence(self) -> tuple[Seq, tuple]:
        rng = np.random.default_rng(self.seed)
        seq = Seq()
        for depth in self._depths:
            block = _build_rb_block(self._dot, int(depth), rng)
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

        # Fit the survival probability decay: P_0(m) = A·exp(-m/tau) + B
        fit = fit_exp(xpts[:n], measured[:n])

        # Derive RB quantities from the decay constant
        tau = fit.derived.get("T1", 1.0)
        p = float(np.exp(-1.0 / tau)) if tau > 0 else 0.0
        # Single-qubit (d=2): F = 1 - (1-p)(d-1)/d = 1 - (1-p)/2
        fidelity = 1.0 - (1.0 - p) / 2.0
        error_per_clifford = (1.0 - p) / 2.0

        fit.derived["p"] = p
        fit.derived["fidelity"] = fidelity
        fit.derived["error_per_clifford"] = error_per_clifford
        return fit
