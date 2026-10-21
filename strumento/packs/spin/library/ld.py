"""``LDOps`` — Loss–DiVincenzo single-spin drive factories (spinQICK-style workflows).

EDSR/ESR rotations on a dot's drive line. Spin stacks are **fraction-native** (no legacy
int-code DB), so calibrated gains pass straight through — the D23 boundary conversion is a
cqed-pack concern, not a core one.

Pure: no ``qick`` import.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from strumento.core.pulses import Gaussian, Pulse
from strumento.core.wiring import LineRef
from strumento.packs.spin.safety import validate_gain

if TYPE_CHECKING:  # elements.py names the ops classes: import cycle guard
    from strumento.core.device import Device
    from strumento.packs.spin.elements import Dot


class LDOps:
    """Drive factories bound to one named Dot element."""

    def __init__(self, device: Device, element: Dot) -> None:
        self.device = device
        self.element = element

    def pi(self, *, freq=None, gain=None, phase=0.0) -> Pulse:
        """A π rotation on the spin (X by default; set ``phase`` for Y)."""
        return self._pulse("pi", freq=freq, gain=gain, phase=phase)

    def hpi(self, *, freq=None, gain=None, phase=0.0) -> Pulse:
        """A π/2 rotation."""
        return self._pulse("hpi", freq=freq, gain=gain, phase=phase)

    def _pulse(self, key: str, *, freq, gain, phase) -> Pulse:
        cal = self.device.calib.dots[self.element.name]
        if key not in cal.pulses:
            raise KeyError(f"dot {self.element.name!r} has no calibrated {key!r} pulse")
        pc = cal.pulses[key]
        resolved_gain = gain if gain is not None else pc.gain
        validate_gain(resolved_gain, label=f"{self.element.name}_{key}")
        return Pulse(
            line=LineRef(self.element.name, "drive"),
            freq_mhz=freq if freq is not None else cal.f_qubit_mhz,
            gain=resolved_gain,
            phase_deg=phase,
            envelope=Gaussian(sigma_us=pc.sigma),
            label=f"{self.element.name}_{key}",
        )
