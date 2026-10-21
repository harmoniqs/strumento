"""Composite pulse sequence builders — BB1, CORPSE, Knill, active reset.

Reusable sequence-construction functions that build error-correcting composite pulse
sequences from a dot's calibrated gate set. Each builder takes the dot's ops object
(which provides ``pi()`` and ``hpi()``) and returns a ``Seq`` ready to compose into
any experiment.

BB1 (Broadband #1): compensates systematic amplitude (over/under-rotation) errors.
    Wimperis, J. Magn. Reson. 109, 221 (1994).

CORPSE (Compensation for Off-Resonance with a Pulse SEquence): compensates systematic
    detuning (off-resonance) errors. Cummins & Jones, New J. Phys. 2, 6 (2000).

Knill: a 5-pulse composite that corrects both amplitude and phase errors simultaneously.
    Knill, Phys. Rev. A 71, 042322 (2005).

All three produce identity-equivalent sequences when the native gates are perfect —
the composite structure only activates its error-correcting properties in the presence
of systematic miscalibration.

Pure: no ``qick`` import.
"""

from __future__ import annotations

import numpy as np

from strumento.core.pulses import Seq


def bb1_pi(dot, *, gain_scale: float = 1.0) -> Seq:
    """BB1 composite π rotation — amplitude-error correcting.

    Structure: π_0 · π_φ₁ · 2π_(3φ₁) · π_φ₁
    where φ₁ = arccos(-1/4) ≈ 104.5°.

    The BB1 sequence corrects systematic amplitude errors to fourth order:
    a bare π pulse has fidelity 1 - ε², while BB1 achieves 1 - ε⁴.

    Parameters
    ----------
    dot : ops object
        A dot element's ops (``LDOps``, ``HoleSpinOps``, etc.) providing
        ``pi(gain=..., phase=...)`` and ``hpi()``.
    gain_scale : float
        Multiplicative scale factor on the calibrated π gain. At 1.0 (default),
        uses the nominal calibration. Values != 1.0 intentionally miscalibrate
        the amplitude — used by ``CompositePulseCalibration`` to sweep the error.

    Returns
    -------
    Seq
        The four-pulse BB1 composite sequence.
    """
    phi1 = np.degrees(np.arccos(-0.25))  # ~104.5°
    ref = dot.pi()
    g = ref.gain * gain_scale

    seq = Seq()
    seq.play(dot.pi(gain=g, phase=0.0))         # π at 0° (the target rotation)
    seq.play(dot.pi(gain=g, phase=phi1))         # π at phase φ₁
    seq.play(dot.pi(gain=g, phase=3.0 * phi1))   # π at phase 3φ₁  (first half of 2π)
    seq.play(dot.pi(gain=g, phase=3.0 * phi1))   # π at phase 3φ₁  (second half of 2π)
    seq.play(dot.pi(gain=g, phase=phi1))         # π at phase φ₁
    return seq


def corpse_pi(dot, *, gain_scale: float = 1.0) -> Seq:
    """CORPSE composite π rotation — detuning-error correcting.

    Structure: (π + π/3)_0 · (2π - 2π/3)_180 · (π/3)_0
    The three rotation angles sum to π, but the phase alternation cancels
    first-order detuning errors.

    The CORPSE angles for a target π rotation are:
        θ₁ = π + π/3 = 4π/3  (240°)
        θ₂ = 2π - 2π/3 = 4π/3  (240°, opposite phase)
        θ₃ = π/3  (60°)

    These are implemented as gain-scaled pulses: each rotation angle is
    achieved by scaling the calibrated π gain by θ/π.

    Parameters
    ----------
    dot : ops object
        A dot element's ops providing ``pi(gain=..., phase=...)``.
    gain_scale : float
        Overall amplitude scale factor (1.0 = nominal).

    Returns
    -------
    Seq
        The three-pulse CORPSE composite sequence.
    """
    ref = dot.pi()
    g_pi = ref.gain * gain_scale

    # CORPSE rotation angles for target θ_target = π:
    # θ₁ = 2π/3 + π/2 - arcsin(sin(π/2) / 2) ... simplified:
    # For a π target, the standard CORPSE angles are 4π/3, 4π/3 (inverted), π/3
    g1 = g_pi * (4.0 / 3.0)   # 4π/3 rotation
    g2 = g_pi * (4.0 / 3.0)   # 4π/3 rotation (phase-inverted)
    g3 = g_pi * (1.0 / 3.0)   # π/3 rotation

    seq = Seq()
    seq.play(dot.pi(gain=g1, phase=0.0))     # 4π/3 at 0°
    seq.play(dot.pi(gain=g2, phase=180.0))   # 4π/3 at 180° (inverted)
    seq.play(dot.pi(gain=g3, phase=0.0))     # π/3 at 0°
    return seq


def knill_pi(dot, *, gain_scale: float = 1.0) -> Seq:
    """Knill composite π rotation — corrects both amplitude and phase errors.

    Structure: five π/2 pulses at specific phases that together implement a
    robust π rotation. Corrects all first-order errors simultaneously.

    Knill, Phys. Rev. A 71, 042322 (2005): the phases are
    [π/6, 0, π/2, 0, π/6] applied to five π/2 pulses, yielding a net π
    rotation that is first-order insensitive to both over-rotation and
    off-resonance errors.

    Parameters
    ----------
    dot : ops object
        A dot element's ops providing ``hpi(gain=..., phase=...)``.
    gain_scale : float
        Overall amplitude scale factor (1.0 = nominal).

    Returns
    -------
    Seq
        The five-pulse Knill composite sequence.
    """
    ref = dot.hpi()
    g = ref.gain * gain_scale

    # Knill phases (degrees): 30°, 0°, 90°, 0°, 30°
    phases = [30.0, 0.0, 90.0, 0.0, 30.0]

    seq = Seq()
    for ph in phases:
        seq.play(dot.hpi(gain=g, phase=ph))
    return seq


# -- Active reset (measurement-conditioned spin initialization) ----------------

#: The label on the corrective π pulse — the response model uses it to identify
#: which pulse is the reset correction (vs. other pulses in the sequence).
RESET_PI_LABEL = "spin_reset_pi"

#: Default settle time (µs) the branch waits after the readout window before reading the
#: accumulator. Lowered to a processor-blocking ``wait_auto(settle_us, ros=True)`` on the
#: :class:`~strumento.core.pulses.Branch` (not a plain idle) so the tProc reads a fresh result —
#: the ADC needs time to finish accumulating before the read.
DEFAULT_FEEDBACK_SETTLE_US = 0.5


def active_reset(
    dot,
    *,
    adc: int = 0,
    rounds: int = 1,
    threshold: float = 0.5,
    settle_us: float | None = None,
    hardware_loop: bool = False,
) -> Seq:
    """Active reset for a spin qubit: ``measure · settle · if(excited) π``.

    Real in-sequence feedback via the tProc-v2 ``Branch`` seam: each round measures
    the spin state (PSB or Elzerman readout), waits for the ADC to settle, then
    conditionally applies a π pulse to flip the spin to ground. Every round's readout
    is a real acquisition read — prepend this to any sequence and the reset reads
    arrive as heralds ahead of the data read.

    Parameters
    ----------
    dot : ops object
        A dot element's ops (``LDOps``, ``HoleSpinOps``) providing ``pi()``.
    adc : int
        Readout input channel to test.
    rounds : int
        Number of reset rounds. Each round independently measures and corrects;
        residual excited population falls with each round.
    threshold : float
        Branch discriminator threshold in the sensor's discriminated units
        (0.5 = midpoint for PSB). A lab run passes its calibrated ADC threshold.
    settle_us : float, optional
        Read-latency margin (µs) lowered to a processor-blocking ``wait_auto(ros=True)`` before the
        branch read, so a fresh accumulator is tested (see :class:`~strumento.core.pulses.Branch`).
        Defaults to :data:`DEFAULT_FEEDBACK_SETTLE_US`. Not a plain idle: a plain wait shifts the
        reference time only and would leave the read stale.
    hardware_loop : bool
        If True, emit rounds as a tProc ``Repeat`` instead of unrolling.
        **Caution:** qick counts trigger macros at assembly, so a ``Measure``
        inside a hardware loop is counted once however many times the tProc
        runs it — the acquisition under-declares reads and those rounds' data
        is lost. Unrolled (default) keeps static = dynamic read count.

    Returns
    -------
    Seq
        The reset composite, ready to compose (append ``.measure()`` for verification).
    """
    from dataclasses import replace

    if int(rounds) < 1:
        raise ValueError(f"active_reset needs rounds >= 1; got {rounds}")

    settle = DEFAULT_FEEDBACK_SETTLE_US if settle_us is None else float(settle_us)
    correction = Seq().play(replace(dot.pi(), label=RESET_PI_LABEL))

    def one_round() -> Seq:
        return (
            Seq()
            .measure()
            .branch(adc=adc, threshold=threshold, test=">=", then=correction, settle_us=settle)
        )

    if hardware_loop:
        return Seq().repeat(one_round(), int(rounds))
    out = Seq()
    for _ in range(int(rounds)):
        out += one_round()
    return out


# -- OC import (optimal-control pulse → Seq) ----------------------------------


def load_oc_pulse(
    solution,
    *,
    drive_map: dict,
    gain_calib,
    dac_rate: float | None = None,
) -> Seq:
    """Import an optimal-control pulse into the spin pack via :func:`~strumento.core.pulses.from_solution`.

    This is the spin equivalent of cqed's ``prepare(dev, target, via="oc", ...)``. It wraps
    the core ``from_solution`` boundary (D17) with spin-specific documentation and defaults.

    Parameters
    ----------
    solution :
        A mapping (or object) with ``times`` (control grid in µs) and ``controls``
        (a mapping of drive name → amplitude samples in physical units, e.g. MHz Rabi rate).
        Typically extracted from a Piccolo ``pulse.jld2`` via a Julia/Python bridge.
    drive_map : dict
        ``{drive_name: (line_ref, carrier_freq_mhz)}`` mapping solver drives to hardware lines.

        Common spin mappings:

        - EDSR drive on a dot: ``("q0", "drive")`` at the Larmor frequency
        - Exchange on a barrier: ``("b0", "fast")`` at 0.0 MHz (baseband)
        - Hole-spin AC gate drive: ``("h1", "drive")`` at the Larmor frequency

        ``line_ref`` is a :class:`~strumento.core.wiring.LineRef` or a ``(line, role)`` tuple.
    gain_calib :
        Physical amplitude → v2 gain fraction: a :class:`~strumento.packs.spin.calibration.SpinGainReference`
        or any callable ``(amp_mhz) -> fraction``.
    dac_rate : float, optional
        Target DAC sample rate (samples/µs). ``None`` keeps the solution's native grid.

    Returns
    -------
    Seq
        A ``Seq`` wrapping a ``Segment`` of ``Arb``-envelope ``Pulse`` objects, with
        provenance in ``seq.meta``. Compose into any experiment: ``seq = load_oc_pulse(...).measure()``.

    Examples
    --------
    Import a GRAPE-optimized exchange gate::

        from strumento.packs.spin.calibration import SpinGainReference

        ref = SpinGainReference(pi_gain_frac=0.4, pi_rabi_mhz=20.0)
        seq = load_oc_pulse(
            solution,
            drive_map={
                "exchange_12": (("b0", "fast"), 0.0),       # baseband exchange
                "drive_1":     (("q0", "drive"), 5000.0),   # EDSR at Larmor
            },
            gain_calib=ref,
        )
    """
    from strumento.core.pulses import from_solution

    return from_solution(solution, drive_map=drive_map, gain_calib=gain_calib, dac_rate=dac_rate)
