"""spin — quantum-dot modality pack (spec §15.5) — spinQICK-modeled (HRL).

Gate-defined quantum dots (`Dot` / `Plunger` / `Barrier`), Loss–DiVincenzo and exchange-only
encodings, PSB / charge-sensing readout. **Designed now, built when a spin user appears** — this is
a *skeleton*: element types, calibration schema, and PSB readout, wired through the standard pack
contract (``CALIBRATION`` + ``compose_elements``) so ``core.device.load`` composes it with no core
change. Pulse libraries (LD / exchange-only control) and tune-up experiments are later work.

The spin pack mostly *exercises* the core seam rather than extending it — a cheap honesty check that
``core`` isn't superconducting/atom-shaped. First device instance: ``devices/spin_demo``.
"""

from strumento.packs.spin.calibration import SpinCalibration
from strumento.packs.spin.elements import Barrier, Dot, Plunger

#: the pack's calibration schema (consumed by ``core.device.load``)
CALIBRATION = SpinCalibration


def compose_elements(topology: list[dict], calib: SpinCalibration) -> dict[str, object]:
    """Build ``{instance-name: element}`` from a spin instance's topology."""
    out: dict[str, object] = {}
    for entry in topology:
        kind, name = entry["type"], entry["name"]
        if kind == "Dot":
            out[name] = Dot(name=name, encoding=entry.get("encoding", "ld"))
        elif kind == "Plunger":
            out[name] = Plunger(name=name)
        elif kind == "Barrier":
            out[name] = Barrier(name=name)
        else:
            raise KeyError(f"unknown spin element type {kind!r}")
    return out
