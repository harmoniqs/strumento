"""cqed coupling forms (spec §5.3, §6.6, D31) — the pack-registered interaction union.

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
    """Excitation swap: g · (a†_a a_b + a_a a†_b)."""

    symmetry = "symmetric"

    @staticmethod
    def operator(ops_a, ops_b, strength):
        return TWO_PI * strength * (ops_a.adag @ ops_b.a + ops_a.a @ ops_b.adag)


@register_coupling_form
class JaynesCummings:
    """Qubit–cavity excitation exchange, g · (σ+_a a_b + σ-_a a†_b); leading term = the swap."""

    symmetry = "symmetric"

    @staticmethod
    def operator(ops_a, ops_b, strength):
        return TWO_PI * strength * (ops_a.adag @ ops_b.a + ops_a.a @ ops_b.adag)


@register_coupling_form
class Exchange:
    """Spin exchange (leading transverse term): J · (a†_a a_b + a_a a†_b)."""

    symmetry = "symmetric"

    @staticmethod
    def operator(ops_a, ops_b, strength):
        return TWO_PI * strength * (ops_a.adag @ ops_b.a + ops_a.a @ ops_b.adag)
