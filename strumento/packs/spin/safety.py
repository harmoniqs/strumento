"""spin pack safety guards — gain and voltage bounds enforcement.

All spin control is baseband voltage on gate electrodes. An out-of-range pulse amplitude
or DC sweep can damage the device (oxide breakdown, irreversible charge rearrangement).
These guards enforce the physical constraints at the pack boundary — before any pulse
reaches the compiler.

Pure: no ``qick`` import.
"""

from __future__ import annotations

#: Default maximum |gain| for fast pulses (v2 fraction, full DAC scale = 1.0).
#: Individual calibrations may tighten this via GateCal.max_v, but this is the hard ceiling.
MAX_GAIN: float = 1.0

#: Default maximum |voltage| for DC sweeps (V) when no calibration is available.
#: Conservative — real devices should always have a GateCal.max_v that overrides this.
DEFAULT_MAX_VOLTAGE: float = 2.0


class GainOutOfRange(ValueError):
    """A requested pulse gain exceeds the safe operating range."""


class VoltageOutOfRange(ValueError):
    """A requested voltage exceeds the gate's calibrated max_v."""


def validate_gain(gain: float, *, label: str = "", limit: float = MAX_GAIN) -> float:
    """Raise :class:`GainOutOfRange` if ``|gain|`` exceeds ``limit``.

    Called by every library ops factory after resolving the final gain from calibration
    or user override — before constructing the Pulse.

    Parameters
    ----------
    gain : float
        The resolved gain value (v2 fraction).
    label : str
        Human-readable context for the error message (e.g. "q1_pi").
    limit : float
        The maximum allowed |gain|. Defaults to 1.0 (full DAC scale).

    Returns
    -------
    float
        The validated gain (unchanged if within range).
    """
    if abs(gain) > limit:
        ctx = f" ({label})" if label else ""
        raise GainOutOfRange(
            f"gain={gain}{ctx} exceeds safe limit |gain| <= {limit}; "
            "refusing to build pulse that could damage the device"
        )
    return gain


def validate_sweep_voltage(
    start: float,
    stop: float,
    *,
    max_v: float | None = None,
    gate_name: str = "",
) -> None:
    """Raise :class:`VoltageOutOfRange` if sweep endpoints exceed the gate's max_v.

    Called by DC-sweep experiments at construction time, before any hardware interaction.

    Parameters
    ----------
    start, stop : float
        Sweep endpoints (V).
    max_v : float or None
        The calibrated maximum voltage for this gate. If None, uses DEFAULT_MAX_VOLTAGE.
    gate_name : str
        Human-readable gate name for the error message.
    """
    limit = max_v if max_v is not None else DEFAULT_MAX_VOLTAGE
    for endpoint, name in [(start, "start"), (stop, "stop")]:
        if abs(endpoint) > limit:
            ctx = f" on gate {gate_name!r}" if gate_name else ""
            raise VoltageOutOfRange(
                f"sweep {name}={endpoint} V{ctx} exceeds max_v={limit} V; "
                "refusing to build sweep that could damage the device"
            )
