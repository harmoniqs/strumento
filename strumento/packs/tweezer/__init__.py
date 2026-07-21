"""tweezer — neutral-atom AOD modality pack (spec §17; Thompson, arXiv 2506.13724).

QICK as the dynamic-tweezer / transport controller: AOD axes as 32-tone mux generators, moves as
`PolyChirp` envelopes, fluorescence **imaging** readout on an external detector. **Designed now,
built when scheduled** — this is a *skeleton*: element types, calibration schema, imaging readout,
and the `PolyChirp` envelope, wired through the standard pack contract (``CALIBRATION`` +
``compose_elements``) so ``core.device.load`` composes it with no core change. Pulse libraries
(the move factory + its envelope lowering) and imaging-interleaved experiments are later work.

First device instance: ``devices/thompson_tweezer``.
"""

from strumento.packs.tweezer.calibration import TweezerCalibration
from strumento.packs.tweezer.elements import AODAxis, AtomArray

#: the pack's calibration schema (consumed by ``core.device.load``)
CALIBRATION = TweezerCalibration


def compose_elements(topology: list[dict], calib: TweezerCalibration) -> dict[str, object]:
    """Build ``{instance-name: element}`` from a tweezer instance's topology."""
    out: dict[str, object] = {}
    for entry in topology:
        kind, name = entry["type"], entry["name"]
        if kind == "AODAxis":
            out[name] = AODAxis(name=name, axis=entry.get("axis", "x"))
        elif kind == "AtomArray":
            out[name] = AtomArray(name=name, n_traps=entry.get("n_traps", 0))
        else:
            raise KeyError(f"unknown tweezer element type {kind!r}")
    return out
