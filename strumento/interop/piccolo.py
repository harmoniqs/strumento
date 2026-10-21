"""Upstream Piccolo bridge (D31) — shape an OperatorModel into Piccolo's system inputs.

Lives in ``interop``, **not** ``core``: ``core`` stays Piccolo-free. Piccolo itself is Julia, so
this adapter does not import or construct a `QuantumSystem`; it prepares the framework-neutral
pieces a `QuantumSystem` is built from — the drift Hamiltonian, an **ordered** list of drive
operators, their labels, and the basis — in the convention the Julia-side consumer expects. That
consumer reads these (or ``OperatorModel.to_dict()``) and constructs the actual system.

This closes the codesign loop's model→pulse direction: the same ``drive_map`` that labels these
drives feeds ``from_solution`` on the way back, so operator labels round-trip.

The pulse→hardware direction is :func:`piccolo_controls_to_solution`: Piccolo's solved ``(u_x, u_y)``
controls (GHz, ns grid) → the ``{times, controls}`` mapping ``from_solution`` consumes (MHz Rabi
rate, µs grid), owning the factor-of-2 and Q-sign convention traps.

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


#: MHz of Rabi rate (ν = Ω/2π) per GHz of Piccolo control amplitude. Piccolo's transmon drive term
#: is ``2π u_x (a + a†)``; the hardware convention is ``½ Ω (a + a†)`` for real Ω, so Ω/2π = 2 u_x
#: and, with GHz → MHz, the factor is 2 × 1000.
PICCOLO_GHZ_TO_RABI_MHZ = 2000.0


def piccolo_controls_to_solution(times_ns, controls_ghz, *, drive: str) -> dict:
    """Piccolo I/Q controls → the ``{times, controls}`` mapping ``from_solution`` consumes.

    ``controls_ghz`` is ``(2, N)``: rows ``u_x`` and ``u_y`` of Piccolo's two-quadrature transmon
    drive (GHz) on ``times_ns`` (ns)::

        H_piccolo = 2π [u_x (a + a†) + u_y · i(a − a†)]

    Returns ``{"times": µs, "controls": {drive: ν}}`` with ν = Ω/2π the complex Rabi rate in MHz,
    in the hardware convention ``H = ½(Ω a† + Ω* a)``, ``Ω ∝ I + iQ``::

        ν = 2000 · (u_x − i · u_y)

    Derivation: ``½(Ω a† + Ω* a)`` with ``Ω = Ω_I + iΩ_Q`` is ``½Ω_I (a + a†) + ½Ω_Q · i(a† − a)``.
    Matching ``2π u_x (a + a†)`` gives ``Ω_I = 4π u_x``, i.e. ν_I = 2 u_x; matching
    ``2π u_y · i(a − a†)`` gives ``Ω_Q = −4π u_y``, i.e. ν_Q = −2 u_y. GHz → MHz is the 1000.

    The two traps this owns (each checked against the Hamiltonians, not documentation, in the
    interop tests):

    * **the factor of 2** — the ½ in the hardware convention. A naive ``GHz × 1000`` import plays
      the pulse at *half* amplitude;
    * **the sign of Q** — Piccolo's ``+u_y · i(a − a†)`` is ``Q = −u_y`` in hardware terms. A naive
      import mirrors a DRAG-like correction, so it increases leakage instead of suppressing it.

    Pure numpy. Raises ``ValueError`` when ``controls_ghz`` is not ``(2, N)`` or its ``N`` does not
    match the length of ``times_ns``.
    """
    t = np.asarray(times_ns, dtype=float)
    u = np.asarray(controls_ghz, dtype=float)
    if u.ndim != 2 or u.shape[0] != 2:
        raise ValueError(f"controls_ghz must be (2, N) rows [u_x; u_y]; got shape {u.shape}")
    if t.ndim != 1 or u.shape[1] != t.shape[0]:
        raise ValueError(
            f"times has {t.shape[0] if t.ndim == 1 else t.shape} points but controls have "
            f"{u.shape[1]} — they must share one grid"
        )
    nu = PICCOLO_GHZ_TO_RABI_MHZ * (u[0] - 1j * u[1])
    return {"times": t * 1e-3, "controls": {drive: nu}}
