"""Exchange-only (EO) Clifford decomposition — 24 Cliffords from J₁₂/J₂₃ exchange pulses.

In the exchange-only encoding, the logical qubit lives in the S=1/2 doublet subspace of
three electrons. Two exchange interactions provide full SU(2) control:

    J₁₂: rotation around ẑ (logical Z)
    J₂₃: rotation around n̂ = (-√3/2, 0, -1/2) — at 120° from ẑ in the XZ plane

Any single-qubit rotation decomposes into alternating J₁₂/J₂₃ pulses via a three-pulse
Euler factorization for two non-orthogonal axes:

    R = R₁₂(α) · R₂₃(β) · R₁₂(γ)

This module precomputes decompositions of all 24 single-qubit Cliffords into (J₁₂, J₂₃)
pulse sequences, with a Cayley table and inverse table for exact RB inversion.

Reference: DiVincenzo et al., Nature 408, 339 (2000); Fong & Wandzura, QIC 11, 1003 (2011).

Pure: no ``qick`` import.
"""

from __future__ import annotations

import numpy as np

from strumento.core.pulses import Seq

# =============================================================================
# Rotation generators for the 120° geometry
# =============================================================================

_I2 = np.eye(2, dtype=complex)
_SX = np.array([[0, 1], [1, 0]], dtype=complex)
_SY = np.array([[0, -1j], [1j, 0]], dtype=complex)
_SZ = np.array([[1, 0], [0, -1]], dtype=complex)

# J₂₃ axis: n̂ = (-√3/2, 0, -1/2) — at 120° from ẑ in the XZ plane
_N23 = np.array([-np.sqrt(3) / 2, 0.0, -0.5])
_SIGMA_N23 = _N23[0] * _SX + _N23[1] * _SY + _N23[2] * _SZ


def _rot_z(theta: float) -> np.ndarray:
    """J₁₂ rotation: exp(-i θ/2 σ_z)."""
    return np.cos(theta / 2) * _I2 - 1j * np.sin(theta / 2) * _SZ


def _rot_n23(theta: float) -> np.ndarray:
    """J₂₃ rotation: exp(-i θ/2 (n̂₂₃ · σ⃗))."""
    return np.cos(theta / 2) * _I2 - 1j * np.sin(theta / 2) * _SIGMA_N23


def _rot_axis(axis: np.ndarray, theta: float) -> np.ndarray:
    """General rotation exp(-i θ/2 (n̂·σ⃗))."""
    sigma_n = axis[0] * _SX + axis[1] * _SY + axis[2] * _SZ
    return np.cos(theta / 2) * _I2 - 1j * np.sin(theta / 2) * sigma_n


def _overlap(u1: np.ndarray, u2: np.ndarray) -> float:
    """Fidelity-like overlap: |tr(U₁†U₂)|/2. Equals 1 when U₁ = U₂ up to global phase."""
    return abs(np.trace(u1.conj().T @ u2)) / 2.0


# =============================================================================
# Build the 24 Clifford unitaries (same enumeration as the LD RB module)
# =============================================================================

_NATIVE_GATES = [
    _I2,
    _rot_axis(np.array([1, 0, 0]), np.pi),       # X(π)
    _rot_axis(np.array([0, 1, 0]), np.pi),       # Y(π)
    _rot_axis(np.array([1, 0, 0]), np.pi / 2),   # X(π/2)
    _rot_axis(np.array([1, 0, 0]), -np.pi / 2),  # X(-π/2)
    _rot_axis(np.array([0, 1, 0]), np.pi / 2),   # Y(π/2)
    _rot_axis(np.array([0, 1, 0]), -np.pi / 2),  # Y(-π/2)
]

_CLIFFORD_PRODUCTS: list[tuple[int, ...]] = [
    (0,), (1,), (2,), (3,), (4,), (5,), (6,),
    (1, 2), (1, 5), (1, 6), (2, 3), (2, 4),
    (3, 5), (3, 6), (4, 5), (4, 6), (5, 3), (5, 4), (6, 3), (6, 4),
    (3, 5, 3), (3, 5, 4), (3, 6, 3), (3, 6, 4),
]


def _build_cliffords() -> list[np.ndarray]:
    cliffs = []
    for prod in _CLIFFORD_PRODUCTS:
        u = _I2.copy()
        for idx in prod:
            u = _NATIVE_GATES[idx] @ u
        cliffs.append(u)
    return cliffs


_CLIFFORDS = _build_cliffords()
N_CLIFFORDS: int = 24


# =============================================================================
# Euler decomposition: R = R₁₂(α) · R₂₃(β) · R₁₂(γ)
# =============================================================================


def _euler_decompose(target: np.ndarray) -> list[tuple[str, float]]:
    """Decompose ``target`` into a sequence of (axis, angle) pairs.

    Tries three-pulse Euler forms (ZNZ and NZN) first; falls back to a five-pulse
    ZNZNZ sequence for the few Cliffords that need it.

    Returns a list of ``("z", angle)`` or ``("n", angle)`` tuples.
    """
    from scipy.optimize import minimize

    def _try_decompose(order, target):
        """Try to decompose using a given axis order (e.g. 'znz', 'nzn', 'znznz')."""
        rot_fns = {"z": _rot_z, "n": _rot_n23}

        def cost(angles):
            u = _I2.copy()
            for ax, ang in zip(order, angles):
                u = rot_fns[ax](ang) @ u
            return 1.0 - _overlap(u, target)

        n = len(order)
        best = None
        rng = np.random.default_rng(42)
        starts = [np.zeros(n)]
        # Structured starts at multiples of π
        for combo in [[np.pi] * n, [-np.pi] * n]:
            starts.append(combo)
        for i in range(n):
            x = np.zeros(n)
            x[i] = np.pi
            starts.append(x.tolist())
            x[i] = -np.pi
            starts.append(x.tolist())
        # Random starts
        for _ in range(24):
            starts.append(rng.uniform(-2 * np.pi, 2 * np.pi, n).tolist())

        for x0 in starts:
            r = minimize(cost, x0, method="Nelder-Mead",
                         options={"xatol": 1e-12, "fatol": 1e-14, "maxiter": 10000})
            if best is None or r.fun < best.fun:
                best = r
            if best.fun < 1e-8:
                break

        if best.fun < 1e-6:
            return list(zip(order, [float(a) for a in best.x]))
        return None

    # Try three-pulse forms first
    for order in ["znz", "nzn"]:
        result = _try_decompose(order, target)
        if result is not None:
            return result

    # Fall back to five-pulse form
    result = _try_decompose("znznz", target)
    if result is not None:
        return result

    raise RuntimeError("EO Clifford decomposition failed for all attempted forms")


# =============================================================================
# Precompute decompositions, Cayley table, inverse table
# =============================================================================


def _compute_eo_tables():
    """Decompose all 24 Cliffords into (axis, angle) sequences, build Cayley + inverse."""
    decomps = []
    for i, cliff in enumerate(_CLIFFORDS):
        steps = _euler_decompose(cliff)
        # Verify reconstruction
        rot_fns = {"z": _rot_z, "n": _rot_n23}
        reconstructed = _I2.copy()
        for ax, ang in steps:
            reconstructed = rot_fns[ax](ang) @ reconstructed
        ov = _overlap(reconstructed, cliff)
        assert ov > 0.999, (
            f"EO Clifford {i} decomposition failed: overlap={ov:.6f}"
        )
        decomps.append(steps)

    # Cayley table
    n = N_CLIFFORDS
    cayley = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(n):
            product = _CLIFFORDS[i] @ _CLIFFORDS[j]
            for k in range(n):
                if _overlap(product, _CLIFFORDS[k]) > 0.99:
                    cayley[i, j] = k
                    break

    # Inverse table
    inverse = np.zeros(n, dtype=int)
    for i in range(n):
        inv_u = _CLIFFORDS[i].conj().T
        for k in range(n):
            if _overlap(inv_u, _CLIFFORDS[k]) > 0.99:
                inverse[i] = k
                break

    return decomps, cayley, inverse


EO_DECOMPOSITIONS, EO_CAYLEY_TABLE, EO_INVERSE_TABLE = _compute_eo_tables()


# =============================================================================
# Sequence builder
# =============================================================================


def play_eo_clifford(b12, b23, clifford_idx: int, seq: Seq) -> None:
    """Append the J₁₂/J₂₃ decomposition of Clifford ``clifford_idx`` to ``seq``.

    Parameters
    ----------
    b12 : ExchangeOps
        Barrier ops for the J₁₂ exchange axis.
    b23 : ExchangeOps
        Barrier ops for the J₂₃ exchange axis.
    clifford_idx : int
        Index into the 24-element Clifford group (0 = identity).
    seq : Seq
        The sequence to append pulses to (modified in place).
    """
    steps = EO_DECOMPOSITIONS[clifford_idx]
    barriers = {"z": b12, "n": b23}
    eps = 1e-6

    for axis, angle in steps:
        if abs(angle) < eps:
            continue
        b = barriers[axis]
        ref = b.pulse()
        # The calibrated pulse does a π rotation and exchange is non-negative, so a pulse can only
        # rotate forward: a negative angle θ is played as the equivalent forward rotation 2π − |θ|
        # (the same rotation up to a global phase), never as |θ| — that would mirror it.
        forward = angle % (2 * np.pi)
        scale = forward / np.pi
        if scale > eps:
            seq.play(b.pulse(length=ref.envelope.length_us * scale))


def build_eo_rb_block(b12, b23, depth: int, rng: np.random.Generator) -> Seq:
    """Build a random EO Clifford sequence of ``depth`` gates with exact inversion.

    Analogous to ``_build_rb_block`` in the LD RB module, but using J₁₂/J₂₃ exchange
    pulses instead of EDSR pi/hpi gates.

    Deliberate public API: this is the exchange-only RB entry point the module exists for
    (see ``docs/spin-pack.md``). No registered experiment calls it yet — it is provided for
    callers composing exchange-only RB on an EO device themselves, and is kept on purpose.
    """
    seq = Seq()
    accumulated = 0

    for _ in range(depth):
        cliff_idx = int(rng.integers(0, N_CLIFFORDS))
        play_eo_clifford(b12, b23, cliff_idx, seq)
        accumulated = int(EO_CAYLEY_TABLE[cliff_idx, accumulated])

    # Exact inverse
    inv_idx = int(EO_INVERSE_TABLE[accumulated])
    play_eo_clifford(b12, b23, inv_idx, seq)

    return seq
