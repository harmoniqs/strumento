"""``SidebandOps`` — the multiphoton-ladder (f0g1 &c.) factories (slice 4).

Bound as an **instance-declared alias** (spec §5.2/D21): the instance YAML's ``aliases:``
section names e.g. ``sideband`` and points it at a transmon element; the pulses come out on
that element's ``sideband`` generator role — a different DAC combined into the same physical
port as the qubit drive.

Pure: no ``qick`` import.
"""

from __future__ import annotations

from strumento.core.pulses import FlatTop, Pulse
from strumento.core.wiring import LineRef
from strumento.packs.cqed.library._gain import gain_out, maxv_for


class SidebandOps:
    """Multiphoton-ladder pulse factories bound to a named transmon's sideband generator."""

    def __init__(self, device: object, element: object) -> None:
        self.device = device
        self.element = element

    def f0g1(self, kind: str = "pi", photon: int = 0, *, freq=None, gain=None, phase=0.0, length=None) -> Pulse:
        """The |f,0⟩↔|g,1⟩ exchange pulse — the common case of :meth:`multiphoton`."""
        return self.multiphoton("f0-g1", kind, photon, freq=freq, gain=gain, phase=phase, length=length)

    def multiphoton(
        self, transition: str, kind: str = "pi", photon: int = 0,
        *, freq=None, gain=None, phase=0.0, length=None,
    ) -> Pulse:
        """A calibrated ladder pulse: ``transition`` (e.g. ``"f0-g1"``), ``kind`` (pi/hpi),
        indexed by ``photon`` number."""
        ladder = self.device.calib.multiphoton.get(transition)
        if ladder is None:
            raise KeyError(
                f"no multiphoton calibration for transition {transition!r} "
                f"(have {sorted(self.device.calib.multiphoton)})"
            )
        pulses = getattr(ladder, kind)
        if photon >= len(pulses):
            raise IndexError(
                f"transition {transition!r} has {len(pulses)} calibrated {kind!r} pulses; "
                f"photon={photon} is out of range"
            )
        cal = pulses[photon]

        ref = LineRef(self.element.name, "sideband")
        freq_mhz = freq if freq is not None else cal.frequency
        maxv = maxv_for(self.device, ref, freq_mhz if not hasattr(freq_mhz, "start") else freq_mhz.start)
        return Pulse(
            line=ref,
            freq_mhz=freq_mhz,
            gain=gain_out(cal.gain, gain, maxv),
            phase_deg=phase,
            envelope=FlatTop(ramp_sigma_us=cal.ramp_sigma, length_us=length if length is not None else cal.length),
            label=f"{transition}_{kind}{photon}",
        )
