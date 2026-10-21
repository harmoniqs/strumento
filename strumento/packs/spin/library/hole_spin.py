"""``HoleSpinOps`` — Ge/SiGe hole-spin qubit drive factories (native SOC, AC gate).

Hole spins in germanium are driven by AC electric fields on the plunger gate itself —
the strong spin-orbit coupling (SOC) converts orbital motion into spin rotation directly,
with no micromagnet or dedicated antenna needed. Rabi frequencies can exceed 100 MHz.

The drive line role is ``"drive"`` (not ``"plunger"``): the physical fact that the drive
goes through the plunger gate is a WIRING concern — the device YAML maps the ``drive``
role to that DAC channel. The ops layer addresses the line by its logical role.

Key differences from LDOps:
- Drive goes through the PLUNGER gate (no separate drive line needed)
- Rabi frequency is power-dependent (nonlinear due to SOC)
- g-factor is anisotropic and gate-voltage-tunable
- Sweet-spot operation (dg/dV = 0) is critical for coherence
- Bloch-Siegert / AC Stark shifts at high power

Pure: no ``qick`` import.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from strumento.core.pulses import FlatTop, Gaussian, Pulse
from strumento.core.wiring import LineRef
from strumento.packs.spin.safety import validate_gain

if TYPE_CHECKING:  # elements.py names the ops classes: import cycle guard
    from strumento.core.device import Device
    from strumento.packs.spin.elements import HoleDot

# Ramp sigma for flat-top edges (a structural constant, not a calibration value)
_DEFAULT_RAMP_US = 0.006


class HoleSpinOps:
    """AC-gate-driven hole-spin control factories bound to a named Dot element.

    The drive is applied to the dot's plunger gate at the Larmor frequency.
    SOC converts the AC electric field into a spin rotation. No micromagnet or
    dedicated MW antenna is needed — the plunger IS the drive.

    Gate times are typically 1-10 ns (among the fastest in spin qubits).
    """

    def __init__(self, device: Device, element: HoleDot) -> None:
        self.device = device
        self.element = element

    def pi(self, *, freq: float | None = None, gain: float | None = None, phase: float = 0.0) -> Pulse:
        """A pi rotation via AC gate drive (SOC-mediated EDSR).

        The drive frequency is the Larmor frequency f_L = g(theta)*mu_B*B/h,
        which depends on the g-tensor and B-field orientation. The Gaussian
        envelope is played on the plunger gate itself.

        Parameters
        ----------
        freq : float, optional
            Override the calibrated Larmor frequency (MHz).
        gain : float, optional
            Override the calibrated pi-pulse amplitude (v2 fraction).
        phase : float
            Pulse phase in degrees (0 = X, 90 = Y).
        """
        f, g, s = self._drive_cal("pi", freq=freq, gain=gain)
        return Pulse(
            line=LineRef(self.element.name, "drive"),
            freq_mhz=f,
            gain=g,
            phase_deg=phase,
            envelope=Gaussian(sigma_us=s),
            label=f"{self.element.name}_pi",
        )

    def hpi(self, *, freq: float | None = None, gain: float | None = None, phase: float = 0.0) -> Pulse:
        """A pi/2 rotation via AC gate drive.

        Half the gain of the pi pulse at the same sigma — the rotation angle
        scales linearly with amplitude for hole spins (small-angle regime).

        Parameters
        ----------
        freq : float, optional
            Override the calibrated Larmor frequency (MHz).
        gain : float, optional
            Override the calibrated amplitude (v2 fraction). If not given, uses
            half the pi-pulse gain.
        phase : float
            Pulse phase in degrees (0 = X, 90 = Y).
        """
        f, g_pi, s = self._drive_cal("pi", freq=freq, gain=None)
        g = gain if gain is not None else g_pi / 2.0
        return Pulse(
            line=LineRef(self.element.name, "drive"),
            freq_mhz=f,
            gain=g,
            phase_deg=phase,
            envelope=Gaussian(sigma_us=s),
            label=f"{self.element.name}_hpi",
        )

    def pi_at_sweet_spot(self, *, freq: float | None = None, gain: float | None = None, phase: float = 0.0) -> Pulse:
        """A pi rotation assuming the gate voltage is at the g-factor sweet spot.

        At the sweet spot (dg/dV_g = 0), the qubit is first-order insensitive
        to charge noise, maximizing T2*. This method builds the same Gaussian
        pulse as :meth:`pi` — the DC operating point (sweet-spot voltage) is
        assumed to be set externally via the DC source before the sequence runs.

        Use this when coherence matters more than Rabi speed: the sweet spot
        often has a slightly lower g (and thus lower Rabi), but the T2*
        improvement more than compensates for typical gate lengths.

        Parameters
        ----------
        freq : float, optional
            Override the calibrated Larmor frequency (MHz).
        gain : float, optional
            Override the calibrated pi-pulse amplitude (v2 fraction).
        phase : float
            Pulse phase in degrees (0 = X, 90 = Y).
        """
        f, g, s = self._drive_cal("pi", freq=freq, gain=gain)
        return Pulse(
            line=LineRef(self.element.name, "drive"),
            freq_mhz=f,
            gain=g,
            phase_deg=phase,
            envelope=Gaussian(sigma_us=s),
            label=f"{self.element.name}_pi_ss",
        )

    def exchange_cz(self, *, barrier: str | None = None, gain: float | None = None, length: float | None = None) -> Pulse:
        """Exchange-mediated CZ gate with an adjacent hole-spin qubit.

        In the dispersive regime (J << Delta_E_Z), the exchange interaction provides
        an effective ZZ coupling. The conditional phase accumulates as
        phi_cond ~ J^2 / (hbar * Delta_E_Z) * t. A baseband flat-top pulse on the
        inter-dot barrier gate activates the exchange for a calibrated duration to
        accumulate exactly pi conditional phase.

        Parameters
        ----------
        barrier : str, optional
            Name of the barrier element to pulse. If not given, uses
            ``"<element>_B"`` as the conventional barrier name.
        gain : float, optional
            Override the calibrated barrier pulse amplitude (v2 fraction).
        length : float, optional
            Override the calibrated pulse duration (microseconds).
        """
        barrier_name = barrier if barrier is not None else f"{self.element.name}_B"

        cal = self.device.calib.gates.get(barrier_name)
        ex = cal.exchange if (cal and cal.exchange) else None

        if gain is not None:
            g = gain
        elif ex and ex.gain is not None:
            g = ex.gain
        else:
            raise KeyError(
                f"barrier {barrier_name!r} has no calibrated exchange gain and no "
                f"gain= override was given"
            )
        validate_gain(g, label=f"cz_{self.element.name}_{barrier_name}")
        if length is not None:
            dur = length
        elif ex and ex.length:
            dur = ex.length
        else:
            raise KeyError(
                f"barrier {barrier_name!r} has no calibrated exchange length and no "
                f"length= override was given"
            )

        return Pulse(
            line=LineRef(barrier_name, "fast"),
            freq_mhz=0.0,  # baseband
            gain=g,
            phase_deg=0.0,
            envelope=FlatTop(ramp_sigma_us=_DEFAULT_RAMP_US, length_us=dur),
            label=f"cz_{self.element.name}_{barrier_name}",
        )

    # -- internals ---------------------------------------------------------------

    def _drive_cal(self, key: str, *, freq: float | None, gain: float | None) -> tuple[float, float, float]:
        """Resolve freq, gain, sigma from calibration or defaults.

        Looks up ``calib.dots[element].pulses[key]`` for the pulse parameters and
        ``calib.dots[element].f_qubit_mhz`` for the Larmor frequency.

        Returns
        -------
        tuple of (freq_mhz, gain, sigma_us)
        """
        cal = self.device.calib.dots.get(self.element.name)
        pcal = cal.pulses.get(key) if (cal and cal.pulses) else None

        if freq is not None:
            f = freq
        elif cal and cal.f_qubit_mhz:
            f = cal.f_qubit_mhz
        else:
            raise KeyError(
                f"hole-spin dot {self.element.name!r} has no calibrated f_qubit_mhz and no "
                f"freq= override was given — cannot determine the Larmor frequency"
            )
        if gain is not None:
            g = gain
        elif pcal and pcal.gain is not None:
            g = pcal.gain
        else:
            raise KeyError(
                f"hole-spin dot {self.element.name!r} has no calibrated {key!r} gain and no "
                f"gain= override was given"
            )
        validate_gain(g, label=f"{self.element.name}_{key}")
        if pcal and pcal.sigma:
            s = pcal.sigma
        else:
            raise KeyError(
                f"hole-spin dot {self.element.name!r} has no calibrated {key!r} sigma"
            )
        return f, g, s
