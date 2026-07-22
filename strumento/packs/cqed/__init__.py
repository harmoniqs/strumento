"""cqed — superconducting (transmon + bosonic) modality pack.

Element types (Transmon / BosonicMode / Coupler / ReadoutResonator), pulse
libraries (transmon / sideband / mode / coupler / composite), the DispersiveIQ
readout kind, a Pydantic calibration schema, and experiments (Rabi / T1 / T2 /
parity / active_reset).

First device instance: the Schuster "multimode" system (``devices/schuster_multimode``).
Pure-transmon, multi-transmon, and transmon+multi-bosonic systems are other
*topologies* of this same pack.

This module is the **pack contract** that ``core.device.load`` consumes after resolving the
pack via the ``strumento.packs`` entry-point group (so ``core`` never imports the pack by
name). A pack exposes:

* ``CALIBRATION`` — the Pydantic calibration schema for the pack.
* ``compose_elements(topology, calib)`` — build the named element instances from an instance's
  topology list + parsed calibration (this is where physics-specific element↔calib wiring
  lives, keeping ``core`` modality-general).
"""

from strumento.packs.cqed import couplings as _couplings  # noqa: F401  (registers coupling forms)
from strumento.packs.cqed.calibration import Calibration
from strumento.packs.cqed.elements import (
    BosonicMode,
    Coupler,
    ReadoutResonator,
    Transmon,
)

#: The pack's calibration schema (consumed by ``core.device.load``).
CALIBRATION = Calibration


def compose_elements(topology: list[dict], calib: Calibration) -> dict[str, object]:
    """Build ``{instance-name: element}`` from a device's topology + parsed calibration.

    Slice-1 mapping: the single transmon binds to ``calib.qubit`` (the readout/coupler/modes
    carry no per-element calib yet — their ops land in later slices). Multi-qubit composition
    (per-element calib sections) is a later-slice extension.
    """
    out: dict[str, object] = {}
    for entry in topology:
        kind, name = entry["type"], entry["name"]
        if kind == "Transmon":
            qc = calib.qubit
            out[name] = Transmon(name=name, f_ge=qc.f_ge, f_ef=qc.f_ef, pulses=qc.pulses)
        elif kind == "BosonicMode":
            out[name] = BosonicMode(
                name=name, role=entry.get("role", "storage"), line=entry.get("line")
            )
        elif kind == "Coupler":
            out[name] = Coupler(name=name)
        elif kind == "ReadoutResonator":
            out[name] = ReadoutResonator(name=name)
        else:
            raise KeyError(f"unknown cqed element type {kind!r}")
    return out

