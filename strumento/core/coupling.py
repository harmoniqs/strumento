"""Typed coupling edges + a pack-registered form registry (D31).

A physical coupler (with a DAC) is an *element* you ``add``; an *interaction* is a separate,
first-class **typed edge**: ``coupling(a, b, form=…, mediated_by=…)``. The ``form`` is a
**pack-registered union** — the D22 envelope pattern applied to interactions — and carries the
interaction's **symmetry** plus an **operator template** (what makes the model Hamiltonian-
extractable; see :mod:`strumento.core.operators`). Coupling *strengths* (g, χ, J) are
calibration values → data, never in the edge.

``core`` owns the edge type, the registry, and generic connectivity patterns; the *forms* live in
packs (``cqed`` registers ``Dispersive`` / ``ZZ`` / ``Exchange`` / ``BeamSplitter`` /
``JaynesCummings``). This module is **pure**: no ``qick`` import.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

_FORMS: dict[str, type] = {}


def register_coupling_form(cls: type) -> type:
    """Register a coupling form so instances (and YAML ``form: <name>``) resolve. Decorator-friendly."""
    _FORMS[cls.__name__] = cls
    return cls


def resolve_form(name_or_cls: Any) -> type:
    """Resolve a form given a class or its registered name."""
    if isinstance(name_or_cls, type):
        return name_or_cls
    try:
        return _FORMS[name_or_cls]
    except KeyError:
        raise KeyError(f"no coupling form {name_or_cls!r} registered (have {sorted(_FORMS)})")


@runtime_checkable
class CouplingForm(Protocol):
    """A coupling form: a symmetry tag + an operator template.

    ``operator(ops_a, ops_b, strength)`` returns the interaction's contribution to the drift
    Hamiltonian, given the two endpoints' embedded mode operators (see
    :class:`strumento.core.operators.ModeOperators`) and a calibrated strength.
    """

    symmetry: str  # "symmetric" | "directional" (control -> target)

    @staticmethod
    def operator(ops_a, ops_b, strength): ...


@dataclass(frozen=True)
class Coupling:
    """A typed interaction edge between two named elements (strength comes from calibration)."""

    a: str
    b: str
    form: Any  # a CouplingForm class, or its registered name
    mediated_by: str | None = None

    @property
    def endpoints(self) -> tuple[str, str]:
        return (self.a, self.b)

    @property
    def symmetry(self) -> str:
        return getattr(resolve_form(self.form), "symmetry", "symmetric")

    def operator(self, ops_a, ops_b, strength):
        """Build this edge's Hamiltonian term (delegates to the resolved form's template)."""
        return resolve_form(self.form).operator(ops_a, ops_b, strength)


# --------------------------------------------------------------------------------------
# Connectivity patterns — plain functions that emit typed edges onto a builder (D30)
# --------------------------------------------------------------------------------------


def ring_couple(builder, names, *, form, mediated_by=None):
    """Couple ``names`` in a ring (each adjacent pair, closing the loop) with one form."""
    n = len(names)
    for i in range(n):
        builder.coupling(names[i], names[(i + 1) % n], form=form, mediated_by=mediated_by)
    return builder


def grid_nn(builder, rows, cols, *, form, name=lambda r, c: f"q{r}_{c}"):
    """Couple a ``rows × cols`` grid with nearest-neighbor edges (right + down) of one form."""
    for r in range(rows):
        for c in range(cols):
            if c + 1 < cols:
                builder.coupling(name(r, c), name(r, c + 1), form=form)
            if r + 1 < rows:
                builder.coupling(name(r, c), name(r + 1, c), form=form)
    return builder
