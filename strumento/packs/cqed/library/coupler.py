"""``CouplerOps`` — beamsplitter-swap factories on the driven coupler.

A swap is a flat-top pulse on the coupler flux line at the difference frequency of the two
modes; the wiring's band-gated generators decide which DAC realizes it (the magic
``freq < 1800`` thresholds of register-level stacks live in the wiring *data* here).

Pure: no ``qick`` import.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from strumento.core.pulses import FlatTop, Pulse
from strumento.core.sweeps import Sweep
from strumento.core.wiring import LineRef
from strumento.packs.cqed.library._gain import gain_out, maxv_for
from strumento.packs.cqed.library.targets import resolve_swap_pair
from strumento.packs.cqed.safety import MAX_COUPLER_GAIN, validate_gain, validate_gain_sweep

if TYPE_CHECKING:  # elements.py names the ops classes: import cycle guard
    from strumento.core.device import Device
    from strumento.packs.cqed.elements import Coupler


class CouplerOps:
    """Swap factories bound to the named coupler element."""

    def __init__(self, device: Device, element: Coupler) -> None:
        self.device = device
        self.element = element

    def swap(self, pair: str | None = None, kind: str = "pi", *, freq=None, gain=None, phase=0.0, length=None) -> Pulse:
        """A calibrated beamsplitter swap between the two elements named in ``pair``.

        ``kind="hpi"`` plays half the calibrated flat length (a 50:50 beamsplitter). ``pair``
        defaults to the device's only calibrated swap; with several, name it.

        gain : see "Gain units" in :mod:`strumento.packs.cqed.library`.
        """
        pair = resolve_swap_pair(self.device, pair, purpose="coupler.swap")
        cal = self.device.calib.swaps.get(pair)
        if cal is None:
            raise KeyError(f"no swap calibration for {pair!r} (have {sorted(self.device.calib.swaps)})")

        freq_mhz = freq if freq is not None else cal.frequency
        freq_scalar = freq_mhz.start if hasattr(freq_mhz, "start") else freq_mhz
        # A frequency Sweep (e.g. SwapSpectroscopy) used to resolve its
        # generator/band from .start alone -- a scan straddling the 1.8 GHz low/high band
        # boundary would silently stay compiled onto whichever generator the start
        # frequency picked, with no check that .stop is still in that same band. Check both
        # endpoints resolve to the same role; refuse rather than silently mis-route.
        if hasattr(freq_mhz, "stop") and freq_mhz.stop != freq_mhz.start:
            start_role = self._role_for(freq_mhz.start)
            stop_role = self._role_for(freq_mhz.stop)
            if start_role != stop_role:
                raise ValueError(
                    f"swap frequency sweep [{freq_mhz.start}, {freq_mhz.stop}] MHz straddles "
                    f"a coupler band boundary (start resolves to generator role "
                    f"{start_role!r}, stop to {stop_role!r}) — split the sweep so each half "
                    f"stays within one band."
                )
        ref = LineRef(self.element.name, self._role_for(freq_scalar))
        maxv = maxv_for(self.device, ref, freq_scalar)

        if length is not None:
            flat = length
        else:
            flat = cal.length if kind == "pi" else cal.length / 2.0

        resolved_gain = gain_out(cal.gain, gain, maxv)

        # Safety: a gain sweep (e.g. a swap-calibration sweep) is validated at its raw
        # endpoints against the same limit a scalar gain faces.
        if isinstance(gain, Sweep):
            validate_gain_sweep(gain, limit=MAX_COUPLER_GAIN, maxv=maxv, role="coupler")
        if not isinstance(resolved_gain, Sweep):
            validate_gain(resolved_gain, limit=MAX_COUPLER_GAIN,
                          role="coupler", label=f"swap_{pair}_{kind}")

        return Pulse(
            line=ref,
            freq_mhz=freq_mhz,
            gain=resolved_gain,
            phase_deg=phase,
            envelope=FlatTop(ramp_sigma_us=cal.ramp_sigma, length_us=flat),
            label=f"swap_{pair}_{kind}",
        )

    def _role_for(self, freq_mhz: float) -> str:
        """Pick the coupler generator role whose band contains ``freq_mhz`` (data, not magic)."""
        line = self.device.wiring.lines[self.element.name]
        for g in line.generators:
            if g.band_hz is None or (g.band_hz[0] <= freq_mhz * 1e6 < g.band_hz[1]):
                return cast(str, g.role)  # DeviceBuilder sets it from the wire-key
        raise ValueError(f"no coupler generator band covers {freq_mhz} MHz")
