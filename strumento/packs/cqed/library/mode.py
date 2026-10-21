"""``ModeOps`` — bosonic-mode factories: displacement drives.

``dev.cavity`` is a ``ModeOps`` because the instance YAML names a ``BosonicMode``
``cavity`` (D21); several storage modes may share one physical line at different
frequencies.

Pure: no ``qick`` import.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from strumento.core.pulses import Gaussian, Pulse
from strumento.core.sweeps import Sweep
from strumento.core.wiring import LineRef
from strumento.packs.cqed.calibration import gain_int_to_frac
from strumento.packs.cqed.library._gain import gain_out, maxv_for
from strumento.packs.cqed.safety import MAX_DISPLACEMENT_GAIN, validate_gain, validate_gain_sweep

if TYPE_CHECKING:  # elements.py names the ops classes: import cycle guard
    from strumento.core.device import Device
    from strumento.packs.cqed.elements import BosonicMode


class ModeOps:
    """Factories bound to one named bosonic mode."""

    def __init__(self, device: Device, element: BosonicMode) -> None:
        self.device = device
        self.element = element

    def _ref(self) -> LineRef:
        return LineRef(self.element.line or self.element.name, "main")

    def displace(self, *, freq=None, gain=None, phase=0.0, length=None) -> Pulse:
        """A calibrated displacement drive on this mode (``calib.modes[<name>].displace``).

        gain : see "Gain units" in :mod:`strumento.packs.cqed.library`.
        """
        mode_cal = self.device.calib.modes.get(self.element.name)
        if mode_cal is None or mode_cal.displace is None:
            raise KeyError(f"no displacement calibration for mode {self.element.name!r}")
        cal = mode_cal.displace

        freq_mhz = freq if freq is not None else self.element.freq_mhz
        if freq_mhz is None:
            raise ValueError(f"mode {self.element.name!r} has no frequency (set freq_mhz or pass freq=)")
        ref = self._ref()
        maxv = maxv_for(self.device, ref, freq_mhz if not hasattr(freq_mhz, "start") else freq_mhz.start)

        # Safety: validate displacement gain against cavity-safe limit
        # Per-mode limit (from calibration YAML) overrides the module-level constant.
        limit = mode_cal.max_displace_gain if mode_cal.max_displace_gain is not None else MAX_DISPLACEMENT_GAIN
        # A gain sweep (e.g. a Wigner/displacement-calibration sweep) is validated at its raw
        # endpoints against the same limit a scalar gain faces.
        if isinstance(gain, Sweep):
            validate_gain_sweep(gain, limit=limit, maxv=maxv, role="displacement",
                                element=self.element.name)
        resolved_gain = gain_out(cal.gain, gain, maxv)
        if not isinstance(resolved_gain, Sweep):
            validate_gain(resolved_gain, limit=limit,
                          role="displacement", label=f"displace_{self.element.name}",
                          element=self.element.name)

        return Pulse(
            line=ref,
            freq_mhz=freq_mhz,
            gain=resolved_gain,
            phase_deg=phase,
            envelope=Gaussian(sigma_us=cal.sigma),
            label=f"displace_{self.element.name}",
        )

    def displace_alpha(self, alpha: float, *, phase: float = 0.0) -> Pulse:
        """A displacement to coherent state |α⟩ using the calibrated gain-per-alpha.

        Reads ``calib.modes[<name>].displacement_cal`` as a native gain code per
        ``|alpha|`` and converts it to the v2 gain fraction required by the pulse IR.

        Parameters
        ----------
        alpha : float
            Target coherent-state amplitude (photon number |α|²).
        phase : float
            Displacement phase in degrees.

        Raises
        ------
        KeyError
            If ``displacement_cal`` is not set (run ``DisplacementCalibration`` first).
        """
        mode_cal = self.device.calib.modes.get(self.element.name)
        if mode_cal is None or mode_cal.displacement_cal is None:
            raise KeyError(
                f"mode {self.element.name!r} has no displacement_cal — "
                "run DisplacementCalibration to calibrate the gain-per-alpha"
            )
        freq_mhz = self.element.freq_mhz
        if freq_mhz is None:
            raise ValueError(f"mode {self.element.name!r} has no frequency")
        native_gain = mode_cal.displacement_cal * abs(alpha)
        gain = gain_int_to_frac(native_gain, maxv_for(self.device, self._ref(), freq_mhz))
        return self.displace(gain=gain, phase=phase)
