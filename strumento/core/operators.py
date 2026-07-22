"""Device-model → Hamiltonian export (spec §6.6, D31) — the Piccolo *input* seam, mirror of §6.5.

Once element types carry bare-Hamiltonian templates and couplings carry operator forms, the
topology graph + calibrated values **are** a Hamiltonian spec:

    H(t) = Σ_i H_i^elem  +  Σ_<i,j> H_ij^coupling  +  Σ_k H_k^drive · u_k(t)

:func:`to_operator_model` lowers a device into a **framework-neutral** :class:`OperatorModel`
(drift H0 + drive operators + basis labels + drive_map + provenance). It stays Piccolo-free — a
thin ``to_piccolo_system`` bridge lives *upstream* and consumes this spec (as could QuTiP). Per
§19.4, strumento owns the export; the solve stays upstream.

``include=`` extracts a **subgraph** (e.g. just the f0g1 subsystem) so a tractable Hilbert space is
built instead of the full device — the line between a possible and an impossible solve.

This module is **pure** (numpy only; no qick, no Piccolo). Operators are dense complex arrays in
**angular-frequency units** (2π·f, f in the calibration's MHz).
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any

import numpy as np

TWO_PI = 2 * np.pi


def annihilation(dim: int) -> np.ndarray:
    """The truncated annihilation operator on a ``dim``-level space."""
    return np.diag(np.sqrt(np.arange(1, dim)), 1).astype(complex)


def _embed(local: np.ndarray, index: int, dims: list[int]) -> np.ndarray:
    """Embed a local operator (acting on element ``index``) into the full tensor-product space."""
    before = int(np.prod(dims[:index])) if index else 1
    after = int(np.prod(dims[index + 1:])) if index + 1 < len(dims) else 1
    return np.kron(np.kron(np.eye(before), local), np.eye(after))


@dataclass
class ModeOperators:
    """One element's ladder operators, embedded in the full space: ``a``, ``adag``, ``n``."""

    a: np.ndarray
    adag: np.ndarray
    n: np.ndarray


@dataclass
class HilbertSpace:
    """A tensor-product space over named elements with per-element cutoffs."""

    names: list[str]
    dims: list[int]

    @property
    def dim(self) -> int:
        return int(np.prod(self.dims)) if self.dims else 1

    def ops(self, name: str) -> ModeOperators:
        i = self.names.index(name)
        a = _embed(annihilation(self.dims[i]), i, self.dims)
        adag = a.conj().T
        return ModeOperators(a=a, adag=adag, n=adag @ a)

    def labels(self) -> list[tuple[int, ...]]:
        """Fock/occupation labels for each basis state (one tuple per element)."""
        return list(itertools.product(*(range(d) for d in self.dims)))


@dataclass
class OperatorModel:
    """A framework-neutral Hamiltonian spec (what a solver optimizes against)."""

    H0: np.ndarray
    drives: dict[str, np.ndarray]
    basis_labels: list
    drive_map: dict[str, str]
    dims: dict[str, int]
    frame: str
    provenance: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """A JSON-serializable form (complex arrays split into real/imag) for an upstream consumer.

        This is the framework-neutral wire format the Piccolo (Julia) or QuTiP bridge reads.
        """

        def _c(m):
            m = np.asarray(m)
            return {"real": m.real.tolist(), "imag": m.imag.tolist()}

        return {
            "H0": _c(self.H0),
            "drives": {k: _c(v) for k, v in self.drives.items()},
            "basis_labels": [list(label) for label in self.basis_labels],
            "drive_map": dict(self.drive_map),
            "dims": dict(self.dims),
            "frame": self.frame,
            "provenance": dict(self.provenance),
        }


def to_operator_model(dev, *, frame: str = "rotating", cutoffs: dict[str, int], include=None) -> OperatorModel:
    """Lower ``dev`` (topology + couplings + calibration) into an :class:`OperatorModel`.

    Args:
        frame: ``"lab"`` keeps each element's bare ω·n; ``"rotating"`` drops it (rotating at the
            element's own frequency), leaving anharmonicity + couplings — the usual solve frame.
        cutoffs: per-element Hilbert-space dimension (Fock cutoff for a mode). Required for every
            selected element — not derivable.
        include: element names to keep (a subgraph, for tractability). Default: every element that
            carries a Hamiltonian template.
    """
    if include is not None:
        sel = list(include)
        for name in sel:
            if not hasattr(dev.elements[name], "hamiltonian"):
                raise ValueError(f"element {name!r} has no Hamiltonian template (cannot export it)")
    else:
        sel = [n for n, el in dev.elements.items() if hasattr(el, "hamiltonian")]
    if not sel:
        raise ValueError("no elements with a Hamiltonian template were selected")

    missing = [n for n in sel if n not in cutoffs]
    if missing:
        raise ValueError(f"cutoffs missing for {missing} (Fock cutoffs are required, not derivable)")

    dims = [int(cutoffs[n]) for n in sel]
    space = HilbertSpace(sel, dims)
    H0 = np.zeros((space.dim, space.dim), dtype=complex)

    for name in sel:
        H0 = H0 + dev.elements[name].hamiltonian(space.ops(name), frame=frame)

    strengths = getattr(dev.calib, "couplings", None) or {}
    for c in dev.couplings:
        if c.a in sel and c.b in sel:
            g = strengths.get(f"{c.a}-{c.b}", strengths.get(f"{c.b}-{c.a}", 0.0))
            H0 = H0 + c.operator(space.ops(c.a), space.ops(c.b), g)

    drives: dict[str, np.ndarray] = {}
    drive_map: dict[str, str] = {}
    for name in sel:
        ops = space.ops(name)
        for ref in dev.elements[name].lines():
            drives[ref.line] = ops.a + ops.adag  # the drive couples to the element's x quadrature
            drive_map[ref.line] = f"{name}:x"

    provenance = {"pack": dev.pack.__name__, "frame": frame, "elements": sel}
    return OperatorModel(
        H0=H0, drives=drives, basis_labels=space.labels(), drive_map=drive_map,
        dims=dict(zip(sel, dims)), frame=frame, provenance=provenance,
    )
