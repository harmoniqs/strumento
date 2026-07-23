"""``ModeOps`` — bosonic-mode factories (slice 4): displacement drives.

``dev.manipulate`` is a ``ModeOps`` because the instance YAML names a ``BosonicMode``
``manipulate`` (D21); several storage modes may share one physical line at different
frequencies.

Pure: no ``qick`` import.
"""

from __future__ import annotations

from strumento.core.pulses import Gaussian, Pulse
from strumento.core.wiring import LineRef
from strumento.packs.cqed.library._gain import gain_out, maxv_for


class ModeOps:
    """Factories bound to one named bosonic mode."""

    def __init__(self, device: object, element: object) -> None:
        self.device = device
        self.element = element

    def _ref(self) -> LineRef:
        return LineRef(self.element.line or self.element.name, "main")

    def displace(self, *, freq=None, gain=None, phase=0.0, length=None) -> Pulse:
        """A calibrated displacement drive on this mode (``calib.modes[<name>].displace``)."""
        mode_cal = self.device.calib.modes.get(self.element.name)
        if mode_cal is None or mode_cal.displace is None:
            raise KeyError(f"no displacement calibration for mode {self.element.name!r}")
        cal = mode_cal.displace

        freq_mhz = freq if freq is not None else self.element.freq_mhz
        if freq_mhz is None:
            raise ValueError(f"mode {self.element.name!r} has no frequency (set freq_mhz or pass freq=)")
        ref = self._ref()
        maxv = maxv_for(self.device, ref, freq_mhz if not hasattr(freq_mhz, "start") else freq_mhz.start)
        return Pulse(
            line=ref,
            freq_mhz=freq_mhz,
            gain=gain_out(cal.gain, gain, maxv),
            phase_deg=phase,
            envelope=Gaussian(sigma_us=cal.sigma),
            label=f"displace_{self.element.name}",
        )
