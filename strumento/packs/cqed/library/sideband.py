"""``SidebandOps`` — the multiphoton-ladder (f0g1 &c.) factories.

Bound as an **instance-declared alias** (D21): the instance YAML's ``aliases:``
section names e.g. ``sideband`` and points it at a transmon element; the pulses come out on
that element's ``sideband`` generator role — a different DAC combined into the same physical
port as the qubit drive.

Pure: no ``qick`` import.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from strumento.core.pulses import FlatTop, Gaussian, Pulse
from strumento.core.sweeps import Sweep
from strumento.core.wiring import LineRef
from strumento.packs.cqed.library._gain import gain_out, maxv_for
from strumento.packs.cqed.safety import MAX_SIDEBAND_GAIN, validate_gain, validate_gain_sweep

if TYPE_CHECKING:  # elements.py names the ops classes: import cycle guard
    from strumento.core.device import Device
    from strumento.core.pulses import Envelope
    from strumento.packs.cqed.elements import Transmon


class SidebandOps:
    """Multiphoton-ladder pulse factories bound to a named transmon's sideband generator."""

    def __init__(self, device: Device, element: Transmon) -> None:
        self.device = device
        self.element = element

    def f0g1(self, kind: str = "pi", photon: int = 0, *, freq=None, gain=None, phase=0.0, length=None) -> Pulse:
        """The |f,0⟩↔|g,1⟩ exchange pulse — the common case of :meth:`multiphoton`.

        gain : see "Gain units" in :mod:`strumento.packs.cqed.library`.
        """
        return self.multiphoton("f0-g1", kind, photon, freq=freq, gain=gain, phase=phase, length=length)

    def multiphoton(
        self, transition: str, kind: str = "pi", photon: int = 0,
        *, freq=None, gain=None, phase=0.0, length=None,
    ) -> Pulse:
        """A calibrated ladder pulse: ``transition`` (e.g. ``"f0-g1"``), ``kind`` (pi/hpi),
        indexed by ``photon`` number.

        gain : see "Gain units" in :mod:`strumento.packs.cqed.library`.
        """
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

        # Safety: a gain sweep (e.g. SidebandRabi) is validated at its raw endpoints against
        # the same limit a scalar gain faces (before this check, a pack-wide gap let
        # the default 0->30000 sweep, fraction ~0.916, past the 0.5 sideband ceiling silently).
        if isinstance(gain, Sweep):
            validate_gain_sweep(gain, limit=MAX_SIDEBAND_GAIN, maxv=maxv, role="sideband")
        resolved_gain = gain_out(cal.gain, gain, maxv)

        # Safety: validate resolved scalar gain against sideband limit.
        if not isinstance(resolved_gain, Sweep):
            validate_gain(resolved_gain, limit=MAX_SIDEBAND_GAIN,
                          role="sideband", label=f"{transition}_{kind}{photon}")

        envelope: Envelope
        if cal.type == "gauss":
            if length is not None:
                raise ValueError("gaussian sideband pulses do not accept a flat length override")
            if cal.sigma is None:  # Defensive guard for manually-constructed calibration objects.
                raise ValueError("gaussian sideband pulse is missing sigma")
            envelope = Gaussian(sigma_us=cal.sigma)
        else:
            if cal.ramp_sigma is None:  # Defensive guard for manually-constructed calibration objects.
                raise ValueError("flat-top sideband pulse is missing ramp_sigma")
            envelope = FlatTop(
                ramp_sigma_us=cal.ramp_sigma,
                length_us=length if length is not None else cal.length,
            )

        return Pulse(
            line=ref,
            freq_mhz=freq_mhz,
            gain=resolved_gain,
            phase_deg=phase,
            envelope=envelope,
            label=f"{transition}_{kind}{photon}",
        )
