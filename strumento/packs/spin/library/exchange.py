"""``ExchangeOps`` — fast exchange pulses on a barrier gate.

An exchange interaction is turned on by a fast baseband voltage pulse on the barrier (flat-top);
the rotation angle is set by ``J·t``. The slow DC operating point stays on the external source
(:mod:`strumento.core.dc`); this is only the fast axis.

Pure: no ``qick`` import.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from strumento.core.pulses import FlatTop, Pulse
from strumento.core.wiring import LineRef
from strumento.packs.spin.safety import validate_gain

if TYPE_CHECKING:  # elements.py names the ops classes: import cycle guard
    from strumento.core.device import Device
    from strumento.packs.spin.elements import Barrier


class ExchangeOps:
    """Exchange-pulse factories bound to one named Barrier element."""

    def __init__(self, device: Device, element: Barrier) -> None:
        self.device = device
        self.element = element

    def pulse(self, *, gain=None, length=None, phase=0.0) -> Pulse:
        """The calibrated exchange pulse: a baseband flat-top on the barrier's ``fast`` role.

        Calibration comes from ``calib.gates[<barrier>].exchange`` (gain is a v2 fraction —
        spin stacks are fraction-native); ``gain``/``length`` override it.
        """
        cal = self.device.calib.gates.get(self.element.name)
        if cal is None or cal.exchange is None:
            raise KeyError(f"no exchange pulse calibrated on gate {self.element.name!r}")
        ex = cal.exchange
        resolved_gain = gain if gain is not None else ex.gain
        validate_gain(resolved_gain, label=f"exchange_{self.element.name}")
        return Pulse(
            line=LineRef(self.element.name, "fast"),
            freq_mhz=0.0,                       # baseband
            gain=resolved_gain,
            phase_deg=phase,
            envelope=FlatTop(ramp_sigma_us=0.006, length_us=length if length is not None else ex.length),
            label=f"exchange_{self.element.name}",
        )
