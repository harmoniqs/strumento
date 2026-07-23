"""``CouplerOps`` — beamsplitter-swap factories on the driven coupler (slice 4).

A swap is a flat-top pulse on the coupler flux line at the difference frequency of the two
modes; the wiring's band-gated generators decide which DAC realizes it (the magic
``freq < 1800`` thresholds of register-level stacks live in the wiring *data* here).

Pure: no ``qick`` import.
"""

from __future__ import annotations

from strumento.core.pulses import FlatTop, Pulse
from strumento.core.wiring import LineRef
from strumento.packs.cqed.library._gain import gain_out, maxv_for


class CouplerOps:
    """Swap factories bound to the named coupler element."""

    def __init__(self, device: object, element: object) -> None:
        self.device = device
        self.element = element

    def swap(self, pair: str = "manipulate-S1", kind: str = "pi", *, freq=None, gain=None, phase=0.0, length=None) -> Pulse:
        """A calibrated beamsplitter swap between the two elements named in ``pair``.

        ``kind="hpi"`` plays half the calibrated flat length (a 50:50 beamsplitter).
        """
        cal = self.device.calib.swaps.get(pair)
        if cal is None:
            raise KeyError(f"no swap calibration for {pair!r} (have {sorted(self.device.calib.swaps)})")

        freq_mhz = freq if freq is not None else cal.frequency
        freq_scalar = freq_mhz.start if hasattr(freq_mhz, "start") else freq_mhz
        ref = LineRef(self.element.name, self._role_for(freq_scalar))
        maxv = maxv_for(self.device, ref, freq_scalar)

        if length is not None:
            flat = length
        else:
            flat = cal.length if kind == "pi" else cal.length / 2.0
        return Pulse(
            line=ref,
            freq_mhz=freq_mhz,
            gain=gain_out(cal.gain, gain, maxv),
            phase_deg=phase,
            envelope=FlatTop(ramp_sigma_us=cal.ramp_sigma, length_us=flat),
            label=f"swap_{pair}_{kind}",
        )

    def _role_for(self, freq_mhz: float) -> str:
        """Pick the coupler generator role whose band contains ``freq_mhz`` (data, not magic)."""
        line = self.device.wiring.lines[self.element.name]
        for g in line.generators:
            if g.band_hz is None or (g.band_hz[0] <= freq_mhz * 1e6 < g.band_hz[1]):
                return g.role
        raise ValueError(f"no coupler generator band covers {freq_mhz} MHz")
