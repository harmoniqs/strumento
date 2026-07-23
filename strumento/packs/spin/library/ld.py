"""``LDOps`` — Loss–DiVincenzo single-spin drive factories (spinQICK-style workflows).

EDSR/ESR rotations on a dot's drive line. Spin stacks are **fraction-native** (no legacy
int-code DB), so calibrated gains pass straight through — the D23 boundary conversion is a
cqed-pack concern, not a core one.

Pure: no ``qick`` import.
"""

from __future__ import annotations

from strumento.core.pulses import Gaussian, Pulse
from strumento.core.wiring import LineRef


class LDOps:
    """Drive factories bound to one named Dot element."""

    def __init__(self, device: object, element: object) -> None:
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
        return Pulse(
            line=LineRef(self.element.name, "drive"),
            freq_mhz=freq if freq is not None else cal.f_qubit_mhz,
            gain=gain if gain is not None else pc.gain,   # fraction-native
            phase_deg=phase,
            envelope=Gaussian(sigma_us=pc.sigma),
            label=f"{self.element.name}_{key}",
        )
