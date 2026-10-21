"""cqed coupling forms (D31) — the pack-registered interaction union.

Each form is the interaction analogue of a D22 envelope: a class carrying a **symmetry** tag and an
**operator template** ``operator(ops_a, ops_b, strength)`` that builds the interaction's drift term
from the two endpoints' embedded mode operators. Strengths (χ, ζ, g, J) are calibration values,
passed in — never baked into the form.

Registered at pack import so ``coupling(..., form="BeamSplitter")`` (string, e.g. from YAML) and
``form=BeamSplitter`` (the class, from the builder) both resolve. Operators are angular-units
(2π·strength), matching :mod:`strumento.core.operators`.
"""

from __future__ import annotations

import numpy as np

from strumento.core.coupling import register_coupling_form
from strumento.core.operators import TWO_PI


def _sigma_z(ops) -> np.ndarray:
    """Pauli-Z in the {|0>,|1>} convention, on the full embedded space: I − 2n."""
    return np.eye(ops.n.shape[0], dtype=complex) - 2 * ops.n


@register_coupling_form
class Dispersive:
    """Dispersive / cross-Kerr shift: χ · n_a n_b (diagonal)."""

    symmetry = "symmetric"

    @staticmethod
    def operator(ops_a, ops_b, strength):
        return TWO_PI * strength * (ops_a.n @ ops_b.n)


@register_coupling_form
class ZZ:
    """Ising ZZ: ζ · σz_a σz_b (σz = I − 2n)."""

    symmetry = "symmetric"

    @staticmethod
    def operator(ops_a, ops_b, strength):
        return TWO_PI * strength * (_sigma_z(ops_a) @ _sigma_z(ops_b))


@register_coupling_form
class BeamSplitter:
    """Excitation swap: g · (a†_a a_b + a_a a†_b).

    Note: this operator is byte-identical to JaynesCummings and
    Exchange below. Intentional at this leading order -- all three are the same
    excitation-swap term; they differ physically only in what's being swapped
    (photons between two bosonic modes here) and in higher-order corrections neither
    this pack nor the others below currently model. If a higher-order term is ever
    added for one of these three, it almost certainly should NOT be added to the
    other two without checking the physics first -- they'd silently diverge from a
    shared implementation into three real ones.
    """

    symmetry = "symmetric"

    @staticmethod
    def operator(ops_a, ops_b, strength):
        return TWO_PI * strength * (ops_a.adag @ ops_b.a + ops_a.a @ ops_b.adag)


@register_coupling_form
class JaynesCummings:
    """Qubit–cavity excitation exchange, g · (σ+_a a_b + σ-_a a†_b); leading term = the swap.

    Note: implemented identically to BeamSplitter/Exchange (see
    BeamSplitter's note above) -- this class's docstring names the physical σ+/σ-
    qubit-raising/lowering operators, but the actual operator() below uses the same
    bosonic adag/a ladder operators as the other two, which is only the LEADING term
    of the true Jaynes-Cummings interaction (it drops the counter-rotating term this
    class's own name usually implies excluding, which is fine for RWA, but also drops
    nothing distinguishing "qubit" from "bosonic mode" -- both endpoints are treated
    as generic ladder systems here). If qubit-specific JC physics (e.g. the
    non-bosonic commutation of σ+/σ- at higher excitation) is ever needed, this is
    the class to change -- not BeamSplitter or Exchange.
    """

    symmetry = "symmetric"

    @staticmethod
    def operator(ops_a, ops_b, strength):
        return TWO_PI * strength * (ops_a.adag @ ops_b.a + ops_a.a @ ops_b.adag)


@register_coupling_form
class Exchange:
    """Spin exchange (leading transverse term): J · (a†_a a_b + a_a a†_b).

    Note: implemented identically to BeamSplitter/JaynesCummings
    (see BeamSplitter's note above). This docstring already says "leading transverse
    term," correctly flagging that the full spin-exchange interaction has more to it
    (e.g. the longitudinal J_z S^z_a S^z_b term for anisotropic exchange, and the
    fermionic sign convention a true spin-exchange Hamiltonian carries that a bosonic
    ladder operator doesn't) -- this class is where that would be added, not the
    other two.
    """

    symmetry = "symmetric"

    @staticmethod
    def operator(ops_a, ops_b, strength):
        return TWO_PI * strength * (ops_a.adag @ ops_b.a + ops_a.a @ ops_b.adag)
