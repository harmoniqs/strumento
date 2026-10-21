"""cqed pack safety guards — gain and sweep bounds enforcement.

Superconducting cQED systems have distinct failure modes from spin qubits: overdriving
a transmon heats TLS and shifts frequencies for hours; high readout power pushes into the
bright-state regime (photon-induced transitions); large cavity displacements leave the
dispersive regime; and coupler overdrive activates parasitic multi-photon transitions.

These guards enforce physical constraints at the pack boundary — before any pulse reaches
the compiler. Each ops factory calls the appropriate validator after resolving the final
gain fraction; experiments call sweep validators at construction time.

**Two tiers of limit (#190).** The ``MAX_*`` values below are conservative **default** limits.
A device may replace one for a single element through its per-element calibration field
(``qubit.max_drive_gain``, ``readout.max_readout_gain``, ``modes.<name>.max_displace_gain``) —
lower to tighten it, or higher for an element behind a weaker amplifier chain. That field is
bounded by the role's absolute **hard cap** (the ``*_CAP`` values): the calibration schema
accepts ``0 < limit <= cap``, so a device whose per-element limit is above its cap does not
load. There is no override flag. The coupler pump and the sideband drive have no per-element
field yet, so their limit is always the default. Past a cap (or one of those defaults), the only
way is to construct the ``Pulse`` directly (Tenet 0), outside the factories. A refusal names the
field to raise (:func:`raise_hint`).

The per-element limits are human-set: in the device YAML, or through the library's
``strumento.core.store.record_calibration``. The pack declares them as gain ceilings
(``CALIBRATION_CEILINGS``), and the MCP session refuses any write-back that touches one, so an
agent cannot raise its own ceiling.

Pure: no ``qick`` import.
"""

from __future__ import annotations

from strumento.core.sweeps import Sweep

# ---------------------------------------------------------------------------
# Configurable per-role gain limits (v2 fraction; full DAC scale = 1.0)
# ---------------------------------------------------------------------------

#: Maximum |gain| for transmon qubit drive pulses (ge and ef transitions).
#: No single-qubit gate should need more than ~0.3 fraction; 0.5 gives headroom for
#: calibration sweeps without risking TLS activation or environment heating.
MAX_QUBIT_DRIVE_GAIN: float = 0.5

#: Maximum |gain| for readout probe pulses.
#: Avoids populating the resonator into the bright-state (Jaynes-Cummings) regime
#: where measurement-induced T1 dominates.
MAX_READOUT_GAIN: float = 0.6

#: Maximum |gain| for cavity displacement pulses.
#: Keeps |alpha| within the dispersive regime; large displacements expose the state
#: to Kerr self-rotation and amplified photon loss.
MAX_DISPLACEMENT_GAIN: float = 0.4

#: Maximum |gain| for coupler pump pulses (beamsplitter swaps).
#: Prevents parasitic multi-photon transitions and coupler junction heating.
MAX_COUPLER_GAIN: float = 0.5

#: Maximum |gain| for sideband/multiphoton transition pulses.
#: Avoids Stark-shifting both qubit and cavity, and leaking population into
#: unintended ladder states.
MAX_SIDEBAND_GAIN: float = 0.5

# ---------------------------------------------------------------------------
# Hard caps on the per-element limits (v2 fraction; full DAC scale = 1.0)
# ---------------------------------------------------------------------------
#
# A per-element limit may sit above the role's default (an element behind a weaker amplifier
# chain needs more DAC fraction for the same drive at the device), but never above its cap. Each
# cap stays below full scale: a cap of 1.0 would bound nothing, since a v2 gain cannot exceed
# full scale anyway, and the top of the DAC range is where a DRAG pulse's added Q quadrature
# clips and the output compresses. These are the calibration schema's ``le=`` bounds.

#: Hard cap on ``qubit.max_drive_gain``: 1.6x the 0.5 default. Leaves the top fifth of the DAC
#: range out of reach, so a DRAG pulse's Q quadrature on top of the I peak stays inside it.
QUBIT_DRIVE_GAIN_CAP: float = 0.8

#: Hard cap on ``readout.max_readout_gain``: 0.2 above the 0.6 default. The same top fifth of
#: the DAC range stays out of reach; past it, readout heads toward the bright-state regime on any
#: chain the default was set for.
READOUT_GAIN_CAP: float = 0.8

#: Hard cap on ``modes.<name>.max_displace_gain``: 1.5x the 0.4 default. Displacement grows |alpha|
#: linearly with gain, so 1.5x the default amplitude is already 2.25x the photon number; more
#: than that leaves the dispersive regime the parity and Wigner experiments assume.
DISPLACEMENT_GAIN_CAP: float = 0.6

#: The calibration field that raises each role's limit for one element, and that field's hard cap.
#: ``{element}`` is filled with the element's name. A role missing here has no per-element
#: override yet (the coupler pump and the sideband drive): its limit is the module default.
CEILING_FIELDS: dict[str, tuple[str, float]] = {
    "qubit_drive": ("qubit.max_drive_gain", QUBIT_DRIVE_GAIN_CAP),
    "readout": ("readout.max_readout_gain", READOUT_GAIN_CAP),
    "displacement": ("modes.{element}.max_displace_gain", DISPLACEMENT_GAIN_CAP),
    "cavity_probe": ("modes.{element}.max_displace_gain", DISPLACEMENT_GAIN_CAP),
}


def raise_hint(role: str, element: str = "") -> str:
    """How to go past ``role``'s limit, for a refusal message: the field to raise, or the bypass."""
    entry = CEILING_FIELDS.get(role)
    if entry is None:
        return (
            "This role has no per-element override yet; to go past its limit, construct the "
            "Pulse directly (Tenet 0)."
        )
    field, cap = entry
    field = field.format(element=element or "<mode>")
    return (
        f"If this element needs more, raise {field} in the device's calibration (up to {cap}; "
        f"see docs/getting-started.md, section 7). Past the cap, construct the Pulse directly "
        f"(Tenet 0)."
    )


# ---------------------------------------------------------------------------
# Sweep range limits
# ---------------------------------------------------------------------------

#: Maximum allowed sweep span for time axes (µs). A 50 ms sweep is almost certainly
#: a misconfiguration (T1 is ~100 µs, cavity T1 ~2-5 ms at most).
MAX_TIME_SWEEP: float = 50_000.0  # µs

#: Minimum allowed sweep start for time axes (µs). Negative delays are unphysical.
MIN_TIME_SWEEP: float = 0.0


# ---------------------------------------------------------------------------
# Typed exceptions
# ---------------------------------------------------------------------------


class GainOutOfRange(ValueError):
    """A requested pulse gain exceeds the safe operating range for this role."""


class SweepOutOfRange(ValueError):
    """A sweep endpoint exceeds physical bounds."""


# ---------------------------------------------------------------------------
# Validators
# ---------------------------------------------------------------------------


def validate_gain(
    gain: float,
    *,
    limit: float,
    role: str = "",
    label: str = "",
    element: str = "",
) -> float:
    """Raise :class:`GainOutOfRange` if ``|gain|`` exceeds ``limit``.

    Called by every cqed ops factory after resolving the final gain fraction —
    before constructing the Pulse. Sweeps are validated at their endpoints.

    Parameters
    ----------
    gain : float
        The resolved gain value (v2 fraction).
    limit : float
        The maximum allowed |gain| for this role.
    role : str
        The physical role (e.g. "qubit_drive", "readout", "displacement").
    label : str
        Human-readable pulse label for the error message.
    element : str
        The element's name, so the message can name its calibration field (a mode's
        ``modes.<name>.max_displace_gain``).

    Returns
    -------
    float
        The validated gain (unchanged if within range).
    """
    if abs(gain) > limit:
        ctx_parts = []
        if role:
            ctx_parts.append(f"role={role}")
        if label:
            ctx_parts.append(f"pulse={label}")
        ctx = f" ({', '.join(ctx_parts)})" if ctx_parts else ""
        raise GainOutOfRange(
            f"|gain|={abs(gain):.4f}{ctx} exceeds safe limit {limit} for cqed; "
            f"refusing to build pulse that could damage the device. {raise_hint(role, element)}"
        )
    return gain


def validate_gain_sweep(
    sweep: Sweep,
    *,
    limit: float,
    maxv: float,
    role: str = "",
    element: str = "",
) -> None:
    """Validate that a gain Sweep's endpoints, once converted to fractions, stay within limit.

    Sweep gain is always lab-native **int-code** domain (D23) — every cqed gain-Sweep call
    site converts it via int-to-fraction division, unconditionally. A *nonzero* endpoint with
    ``0 < abs(value) < 1.0`` cannot plausibly be a real int code (every calibrated gain in this
    pack sits in the thousands — ``pi_ge=8192``, ``pi_ef=7600``, ``hpi_ef=3800``, ...); it is
    almost certainly a v2 fraction the caller meant to use as-is (the scalar-gain convention),
    passed into the Sweep branch by mistake. Converting it anyway divides by ``maxv`` a
    *second* time and silently collapses the sweep toward zero — and this very check would
    compute a trivially-"safe" fraction from the wrong domain (bug #78), letting a genuinely
    unsafe intended fraction slip past the ceiling below. Such a value is rejected loudly
    instead of silently mis-converted. A zero endpoint is exempt: ``Sweep(start=0, stop=...,
    on=...)`` is a completely normal "off" int code, not a domain mistake.

    Parameters
    ----------
    sweep : Sweep
        The gain sweep (lab-native int codes).
    limit : float
        Maximum allowed fraction.
    maxv : float
        The generator's full-scale code (for int-to-fraction conversion).
    role : str
        Physical role for the error message.
    element : str
        The element's name, for the message (see :func:`validate_gain`).
    """
    for endpoint_name, val in [("start", sweep.start), ("stop", sweep.stop)]:
        if val != 0.0 and abs(val) < 1.0:
            raise GainOutOfRange(
                f"gain sweep {endpoint_name}={val} on {role} looks like a v2 fraction, not a "
                f"lab-native int code (calibrated gains in this pack sit in the thousands); "
                f"Sweep gain is always int-code domain, so this endpoint would silently "
                f"collapse toward zero after conversion. Pass the int-code bound you mean, "
                f"or use a scalar gain (raw fraction, used as-is) instead of a Sweep."
            )
        frac = abs(val) / maxv if maxv > 0 else abs(val)
        if frac > limit:
            raise GainOutOfRange(
                f"gain sweep {endpoint_name}={val} (fraction={frac:.4f}) on {role} "
                f"exceeds safe limit {limit}; refusing to build sweep. {raise_hint(role, element)}"
            )


def validate_sweep_range(
    sweep: Sweep,
    *,
    min_val: float | None = None,
    max_val: float | None = None,
    label: str = "",
) -> None:
    """Raise :class:`SweepOutOfRange` if sweep endpoints are outside [min_val, max_val].

    Called at experiment construction time to catch misconfigured sweeps before
    any hardware interaction.

    Parameters
    ----------
    sweep : Sweep
        The sweep to validate.
    min_val : float or None
        Minimum allowed value (None = no lower bound).
    max_val : float or None
        Maximum allowed value (None = no upper bound).
    label : str
        Human-readable name for the error message (e.g. "delay", "frequency").
    """
    for endpoint_name, val in [("start", sweep.start), ("stop", sweep.stop)]:
        if min_val is not None and val < min_val:
            raise SweepOutOfRange(
                f"sweep {endpoint_name}={val} ({label}) is below minimum {min_val}"
            )
        if max_val is not None and val > max_val:
            raise SweepOutOfRange(
                f"sweep {endpoint_name}={val} ({label}) exceeds maximum {max_val}"
            )
