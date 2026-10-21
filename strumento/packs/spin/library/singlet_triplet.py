"""``SingletTripletOps`` — S-T0 qubit control factories (double-dot encoding).

The singlet-triplet qubit uses:
- X-rotation: exchange pulse J(epsilon) via detuning (baseband, not microwave)
- Z-rotation: free precession under Delta_Bz gradient (timed idle, not a driven pulse)

These are fundamentally different from LD's microwave-driven rotations — the control
is entirely baseband (detuning pulses and timed waits), no RF drive needed.

Pure: no ``qick`` import.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from strumento.core.pulses import FlatTop, Pulse
from strumento.core.wiring import LineRef
from strumento.packs.spin.safety import validate_gain

if TYPE_CHECKING:  # elements.py names the ops classes: import cycle guard
    from strumento.core.device import Device
    from strumento.packs.spin.elements import STDot

# Ramp sigma for flat-top edges (a structural constant, not a calibration value)
_DEFAULT_RAMP_US = 0.005


class SingletTripletOps:
    """S-T0 qubit control factories bound to a named Dot element (the "left" dot of the pair).

    The S-T0 qubit lives in a double dot. The ops are:

    - ``j_pi()``: a full exchange rotation (baseband detuning pulse toward (2,0))
    - ``j_hpi()``: a half exchange rotation
    - ``z_idle(duration_ns)``: timed idle at J~0 for Z-rotation under Delta_Bz
    - ``singlet_prep()``: adiabatic ramp from (2,0) to (1,1) to prepare |S>

    The physical drive is a baseband voltage pulse on the detuning axis (virtual gate),
    NOT a microwave tone. The exchange strength J is set by the detuning amplitude;
    the rotation angle is set by J * t.
    """

    def __init__(self, device: Device, element: STDot) -> None:
        self.device = device
        self.element = element

    def j_pi(self, *, gain: float | None = None, length: float | None = None, phase: float = 0.0) -> Pulse:
        """Exchange pi-pulse: full X-rotation in the S-T0 Bloch sphere.

        A baseband flat-top detuning pulse that activates exchange J for time
        t_pi = h/(2J). The detuning amplitude (gain) sets J; the pulse length
        sets the rotation angle.

        Parameters
        ----------
        gain : float, optional
            Override the calibrated detuning amplitude (v2 fraction).
        length : float, optional
            Override the calibrated pulse length (microseconds).
        phase : float
            Pulse phase in degrees (baseband; usually 0).
        """
        g, dur = self._j_cal("j_pi", gain=gain, length=length)
        return Pulse(
            line=LineRef(self.element.name, "detuning"),
            freq_mhz=0.0,  # baseband
            gain=g,
            phase_deg=phase,
            envelope=FlatTop(ramp_sigma_us=_DEFAULT_RAMP_US, length_us=dur),
            label=f"{self.element.name}_j_pi",
        )

    def j_hpi(self, *, gain: float | None = None, length: float | None = None, phase: float = 0.0) -> Pulse:
        """Exchange pi/2-pulse: half X-rotation in the S-T0 Bloch sphere.

        Same detuning amplitude as ``j_pi`` but half the duration, giving
        theta = pi/2 (a sqrt-X equivalent in the encoded basis).

        Parameters
        ----------
        gain : float, optional
            Override the calibrated detuning amplitude (v2 fraction).
        length : float, optional
            Override the calibrated pulse length (microseconds). If not given,
            uses half the calibrated ``j_pi`` length.
        phase : float
            Pulse phase in degrees (baseband; usually 0).
        """
        g, dur_full = self._j_cal("j_pi", gain=gain, length=None)
        # Half duration for pi/2 rotation at the same J
        dur = length if length is not None else dur_full / 2.0
        return Pulse(
            line=LineRef(self.element.name, "detuning"),
            freq_mhz=0.0,  # baseband
            gain=g,
            phase_deg=phase,
            envelope=FlatTop(ramp_sigma_us=_DEFAULT_RAMP_US, length_us=dur),
            label=f"{self.element.name}_j_hpi",
        )

    def z_idle(self, duration_ns: float) -> float:
        """Return idle duration in microseconds for Z-rotation under Delta_Bz.

        The Z-rotation angle is theta = 2*pi*f_z*tau where f_z = g*mu_B*Delta_Bz/h.
        This is NOT a pulse — the caller inserts a :class:`~strumento.core.pulses.Delay`
        of the returned duration into the sequence.

        Parameters
        ----------
        duration_ns : float
            Requested idle duration in nanoseconds.

        Returns
        -------
        float
            The same duration converted to microseconds (the IR's native time unit).
        """
        return duration_ns / 1000.0

    def singlet_prep(self, *, gain: float | None = None, length: float | None = None) -> Pulse:
        """Prepare the singlet state |S> by adiabatic passage from (2,0) to (1,1).

        A slow detuning ramp from deep in (2,0) — where the ground state is
        |S(2,0)> — to the (1,1) operating point. The adiabatic condition requires
        the ramp to be slow compared to the tunnel coupling: dε/dt << 2*t_c^2/hbar.

        Approximated as a FlatTop with a long flat section; the Gaussian ramps provide
        smooth turn-on/off that helps maintain adiabaticity at the edges.

        Parameters
        ----------
        gain : float, optional
            Override the detuning amplitude (v2 fraction). Sets how deep into
            (2,0) we start.
        length : float, optional
            Override the ramp duration (microseconds). Longer = more adiabatic.
        """
        cal = self.device.calib.dots.get(self.element.name)
        pcal = cal.pulses.get("singlet_prep") if (cal and cal.pulses) else None

        if gain is not None:
            g = gain
        elif pcal and pcal.gain is not None:
            g = pcal.gain
        else:
            raise KeyError(
                f"S-T dot {self.element.name!r} has no calibrated 'singlet_prep' gain and no "
                f"gain= override was given"
            )
        validate_gain(g, label=f"{self.element.name}_singlet_prep")
        if length is not None:
            dur = length
        elif pcal and pcal.length:
            dur = pcal.length
        else:
            raise KeyError(
                f"S-T dot {self.element.name!r} has no calibrated 'singlet_prep' length and no "
                f"length= override was given"
            )

        return Pulse(
            line=LineRef(self.element.name, "detuning"),
            freq_mhz=0.0,  # baseband
            gain=g,
            phase_deg=0.0,
            envelope=FlatTop(ramp_sigma_us=dur / 5.0, length_us=dur),
            label=f"{self.element.name}_singlet_prep",
        )

    # -- internals ---------------------------------------------------------------

    def _j_cal(self, key: str, *, gain: float | None, length: float | None) -> tuple[float, float]:
        """Resolve gain and length from calibration — raise if absent.

        Looks up ``calib.dots[element].pulses[key]``. No silent fallback to
        hardcoded defaults (Rule 4: config is data).
        """
        cal = self.device.calib.dots.get(self.element.name)
        pcal = cal.pulses.get(key) if (cal and cal.pulses) else None

        if gain is not None:
            g = gain
        elif pcal and pcal.gain is not None:
            g = pcal.gain
        else:
            raise KeyError(
                f"S-T dot {self.element.name!r} has no calibrated {key!r} gain and no "
                f"gain= override was given"
            )
        validate_gain(g, label=f"{self.element.name}_{key}")
        if length is not None:
            dur = length
        elif pcal and pcal.length:
            dur = pcal.length
        else:
            raise KeyError(
                f"S-T dot {self.element.name!r} has no calibrated {key!r} length and no "
                f"length= override was given"
            )
        return g, dur
