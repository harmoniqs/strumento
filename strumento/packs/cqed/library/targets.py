"""Which mode, coupler swap or multiphoton transition a cqed experiment acts on.

Nothing in the pack hardcodes an element name (D21): a bosonic experiment takes its target as an
argument, and when the argument is ``None`` it is resolved from the device here:

* **A mode** (:func:`resolve_mode`) is the one ``BosonicMode`` whose ``role`` is ``"cavity"``
  (the mode the qubit maps parity through), else the device's only ``BosonicMode``. Several
  candidates and no single cavity is an error naming them, so the caller chooses.
* **A swap pair** (:func:`resolve_swap_pair`) is the only calibrated coupler swap, or the only one
  that involves a named mode.

The multiphoton experiments are not resolved here: they implement the f0-g1 transition's physics
(prepare |f,0⟩ with ``ge_pi · ef_pi``, then drive |f,0⟩↔|g,1⟩), so they read the transition by
its physics name, ``calib.multiphoton["f0-g1"]``, not by a device's choice of label.

An explicit name is used as given; the factories then report a missing calibration themselves.

Pure: no ``qick`` import.
"""

from __future__ import annotations

from typing import Any

#: The ``BosonicMode.role`` that marks the mode a bosonic experiment acts on by default.
CAVITY_ROLE = "cavity"


def bosonic_modes(device: Any) -> dict[str, Any]:
    """The device's ``BosonicMode`` elements, by name, in topology order."""
    return {
        name: el for name, el in device.elements.items() if type(el).__name__ == "BosonicMode"
    }


def resolve_mode(device: Any, mode: str | None, *, purpose: str, arg: str = "mode_name") -> str:
    """The bosonic mode ``purpose`` acts on: ``mode`` if given, else the device's cavity.

    ``arg`` is the caller's parameter name, quoted in the error so the fix is explicit.
    """
    if mode is not None:
        return mode
    modes = bosonic_modes(device)
    if not modes:
        raise ValueError(f"{purpose}: the device has no bosonic mode to act on")
    cavities = [name for name, el in modes.items() if getattr(el, "role", None) == CAVITY_ROLE]
    if len(cavities) == 1:
        return cavities[0]
    if len(modes) == 1:
        return next(iter(modes))
    raise ValueError(
        f"{purpose}: cannot choose a bosonic mode — the device has {sorted(modes)} and "
        f"{len(cavities)} with role {CAVITY_ROLE!r}; pass {arg}= to choose one"
    )


def resolve_swap_pair(
    device: Any, pair: str | None, *, purpose: str, mode: str | None = None
) -> str:
    """The calibrated coupler swap ``purpose`` plays: ``pair`` if given, else the only candidate.

    With ``mode``, the candidates are the swaps that involve it, and an explicit ``pair`` must
    involve it too.
    """
    if pair is not None:
        if mode is not None and mode not in pair.split("-"):
            raise ValueError(f"{purpose}: swap pair {pair!r} does not involve mode {mode!r}")
        return pair
    swaps = sorted(device.calib.swaps)
    candidates = [p for p in swaps if mode is None or mode in p.split("-")]
    if len(candidates) == 1:
        return candidates[0]
    where = f" involving {mode!r}" if mode is not None else ""
    if not candidates:
        raise ValueError(f"{purpose}: no coupler swap{where} is calibrated (have {swaps})")
    raise ValueError(
        f"{purpose}: cannot choose a coupler swap{where} — calibrated: {candidates}; "
        f"pass pair= to choose one"
    )
