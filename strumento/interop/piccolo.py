"""Upstream Piccolo bridge (spec §6.6) — shape an OperatorModel into Piccolo's system inputs.

Lives in ``interop``, **not** ``core``: ``core`` stays Piccolo-free. Piccolo itself is Julia, so
this adapter does not import or construct a `QuantumSystem`; it prepares the framework-neutral
pieces a `QuantumSystem` is built from — the drift Hamiltonian, an **ordered** list of drive
operators, their labels, and the basis — in the convention the Julia-side consumer expects. That
consumer reads these (or ``OperatorModel.to_dict()``) and constructs the actual system.

This closes the codesign loop's model→pulse direction: the same ``drive_map`` that labels these
drives feeds ``from_solution`` on the way back, so operator labels round-trip.

Pure: numpy only; no qick, no Piccolo.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from strumento.core.operators import OperatorModel


@dataclass
class PiccoloSystemSpec:
    """Piccolo-convention inputs: a drift + an ordered drive list (with labels) + the basis.

    ``H_drives[k]`` is the operator the k-th control ``u_k(t)`` multiplies; ``drive_labels[k]`` is
    its ``drive_map`` label (e.g. ``"qubit:x"``'s line name), so a solved pulse maps back through
    ``from_solution``.
    """

    H_drift: np.ndarray
    H_drives: list[np.ndarray]
    drive_labels: list[str]
    basis_labels: list


def to_piccolo_system(model: OperatorModel) -> PiccoloSystemSpec:
    """Shape a :class:`OperatorModel` into :class:`PiccoloSystemSpec` (deterministic drive order)."""
    labels = sorted(model.drives)
    return PiccoloSystemSpec(
        H_drift=model.H0,
        H_drives=[model.drives[name] for name in labels],
        drive_labels=labels,
        basis_labels=model.basis_labels,
    )
