"""Device-level composites: ``parity``, ``prepare`` and ``active_reset``.

Composites live on ``dev`` conceptually (they span elements) and take element *names* as
arguments (D21) — they are plain functions over the device, returning a :class:`Seq`.

Pure: no ``qick`` import.
"""

from __future__ import annotations

from dataclasses import replace

from strumento.core.pulses import Seq, from_solution
from strumento.core.sweeps import Sweep
from strumento.packs.cqed.library.targets import resolve_mode

#: label stamped on the conditional π so a response model (and a reader of the IR) can tell the
#: corrective pulse from any other pulse in a sequence. Part of the composite's contract.
RESET_PI_LABEL = "active_reset_pi"

#: settle time (µs) the branch waits *after the readout window ends* before it reads the
#: accumulator: the ADC-transfer + tProc read-latency margin. Board-specific — a lab tunes
#: ``settle_us``.
#:
#: This rides on the :class:`~strumento.core.pulses.Branch` (its ``settle_us``), which lowers it to
#: a **processor-blocking** ``wait_auto(settle_us, ros=True)`` before ``read_and_jump`` — the
#: freshness gate that makes on-board feedback read a real accumulator instead of a stale one. It is
#: deliberately *not* a preceding ``wait()``: a plain wait only advances the *reference* time for
#: timed ops and never stalls the processor, so the branch would read stale data (the bug this
#: replaced). The readout window itself is not part of this margin — ``wait_auto`` keys off the
#: window end directly (and ``Compiler.play``'s cursor already advances past it), so there is no
#: length to double-count.
DEFAULT_FEEDBACK_SETTLE_US = 0.3

#: discriminated-units threshold: the branch fires above the midpoint between the ``|g⟩`` and
#: ``|e⟩`` blobs (the same convention ``DispersiveIQ.threshold`` uses).
DEFAULT_RESET_THRESHOLD = 0.5

#: the parity mapping's closing half-pi phase (deg) — the fixed operating point of a single-phase
#: parity/Wigner measurement, and the phase the ``parity_fringe`` sweep must be read at to fix the
#: excited<->parity sign. A parity fringe scans this through a full turn; the reconstruction samples
#: at this default. Single point of truth for both call sites.
PARITY_CLOSING_PHASE_DEG = 180.0


class Fock:
    """A Fock-state preparation target: ``Fock(1)``."""

    def __init__(self, n: int) -> None:
        self.n = n

    def __repr__(self) -> str:
        return f"Fock({self.n})"


def parity(dev, mode: str | None = None, *, second_phase=PARITY_CLOSING_PHASE_DEG,
           fast: bool = False, measure: bool = True, line=None) -> Seq:
    """The photon-parity mapping sequence: ``hpi · wait(revival) · hpi(second_phase)``.

    ``mode`` defaults to the device's cavity (:func:`~strumento.packs.cqed.library.targets.resolve_mode`).
    The wait is the named mode's calibrated parity revival time (π/χ); ``fast=True`` picks the
    second calibrated revival entry (e.g. an AC-Stark-shifted fast revival) when present.
    ``second_phase`` may be a :class:`~strumento.core.sweeps.Sweep` (a parity-fringe phase scan).

    **Fast-parity AC-Stark correction.** On the fast path, and only there, if the mode
    carries a ``revival_stark_shift`` the closing half-pi's phase is advanced by
    ``360 · revival_stark_shift · revival`` degrees. During the short fast wait the qubit precesses
    at the AC-Stark-shifted frequency; this phase returns the second pulse to the bare frame so the
    fringe is not rotated. The slow path applies no correction: its revival time is fit directly and
    already absorbs any static shift. A ``None`` ``revival_stark_shift`` (or the slow path) is a
    no-op, so shipped slow-parity runs are unchanged. A swept ``second_phase`` is offset by the same
    constant (the axis binding is preserved).

    ``measure`` (default ``True``, #100) appends
    :meth:`~strumento.packs.cqed.library.readout.ReadoutOps.dispersive_measure` -- drive the
    readout probe, then trigger the ADC -- so every caller reads out through a driven resonator
    instead of leaving a bare ``.measure()`` to the caller (which read nothing, on real
    hardware). Investigation of every existing call site (#100) confirmed none inserts a pulse
    between the composite and its ``.measure()``, so this is the composite-level fix rather than
    a per-caller one. Pass ``measure=False`` for the bare state-mapping sequence -- the state it
    prepares is unchanged either way; only the readout tail is conditional.

    ``line`` overrides the readout probe's wiring line (forwarded to ``dispersive_measure``);
    ``None`` resolves the normal ``("readout", "probe")`` line. Pass e.g. ``("qubit", "drive")``
    for a single-generator instance with no dedicated readout DAC line (the same workaround
    :class:`~strumento.packs.cqed.experiments.resonator_spectroscopy.ResonatorSpectroscopy` uses
    against ``loopback_demo``); ignored when ``measure=False``.
    """
    mode = resolve_mode(dev, mode, purpose="parity", arg="mode")
    mode_cal = dev.calib.modes.get(mode)
    if mode_cal is None or not mode_cal.revival_time:
        raise KeyError(f"no revival_time calibration for mode {mode!r}")
    idx = 1 if fast else 0
    if idx >= len(mode_cal.revival_time):
        raise IndexError(f"mode {mode!r} has no fast revival calibrated")
    revival = mode_cal.revival_time[idx]

    # Fast-parity AC-Stark phase correction (fast path only): advance
    # the closing half-pi by 360·f_AC·t_rev so the pulse lands in the bare frame after the qubit
    # precessed at the Stark-shifted frequency during the short fast wait. Slow path / unset shift:
    # no-op (the slow revival is fit directly).
    closing_phase = second_phase
    if fast and mode_cal.revival_stark_shift is not None:
        stark_deg = 360.0 * float(mode_cal.revival_stark_shift) * revival
        if isinstance(closing_phase, Sweep):
            closing_phase = closing_phase.mapped(
                lambda p: p + stark_deg, inverse=lambda p: p - stark_deg
            )
        else:
            closing_phase = (closing_phase + stark_deg) % 360.0

    q = dev.qubit
    seq = Seq().play(q.ge_hpi()).wait(revival).play(q.ge_hpi(phase=closing_phase))
    if measure:
        seq += dev.readout.dispersive_measure(line=line)
    return seq


def active_reset(
    dev,
    *,
    rounds: int = 1,
    threshold: float = DEFAULT_RESET_THRESHOLD,
    branch_threshold: float | None = None,
    adc: int | None = None,
    component: str = "I",
    test: str = ">=",
    settle_us: float | None = None,
    hardware_loop: bool = False,
    readout_phase_deg: float = 0.0,
    line=None,
) -> Seq:
    """Active reset: ``measure · settle · if(|e⟩) π`` — optionally ``rounds`` times.

    Real in-sequence feedback, not host post-selection: each round lowers to a tProc-v2
    ``read_and_jump``/``label``/``jump`` (see :class:`~strumento.core.pulses.Branch`), so the
    corrective π is decided *on the board*, per shot, from that shot's own readout. Every round's
    readout is a real acquisition read, which is what feeds the herald / ``keep_mask`` path —
    prepend this to any sequence and the reset reads arrive as heralds ahead of the data read (the
    pack convention: the **last** read is the data read).

    Each round's ``measure`` (#100) is
    :meth:`~strumento.packs.cqed.library.readout.ReadoutOps.dispersive_measure` -- drives the
    readout probe before triggering the ADC -- rather than a bare ``.measure()``, which read an
    undriven resonator: the branch that decides the corrective π needs a real readout signal to
    condition on.

    Parameters
    ----------
    rounds:
        Reset rounds. Each round is an independent measure-and-correct: the residual excited
        population falls with every round, at the cost of one read and one settle time.
    threshold:
        The branch discriminator threshold used when ``branch_threshold`` is ``None``, in the
        **discriminated units** :class:`~strumento.packs.cqed.readout.DispersiveIQ` uses (0.5 = the
        midpoint between the blobs) — the ``--sim`` default.
    branch_threshold:
        Overrides ``threshold`` for the on-board branch only, in **raw ADC-accumulator units**: the
        ``read_and_jump`` comparison is against the raw accumulator on one quadrature, not the
        projected IQ (trap #1). A hardware run passes its calibrated accumulator threshold here;
        ``None`` (default) falls back to ``threshold`` so ``--sim`` and existing callers are unchanged.
    adc:
        Readout input to test; defaults to the instance's first ADC.
    component:
        The raw readout quadrature the branch compares, ``"I"`` or ``"Q"``. The branch cannot
        rotate, so the readout phase must land the g/e separation on this axis (trap #1).
    test:
        The branch comparison: ``">="`` (default — correct when the accumulator reads *above*
        threshold, i.e. |e⟩ high) or ``"<"``. Flip it when |e⟩ reads low on the chosen quadrature,
        else the corrective π fires on the ground shots (trap #2).
    readout_phase_deg:
        DDC demodulation phase (degrees) folded into every reset-round probe's readout config, so
        the g/e separation lands on the branch's raw ``component`` quadrature — the branch cannot
        rotate (trap #1). Defaults to ``0.0`` (the ``--sim``/software-angle path, where the host
        :class:`~strumento.packs.cqed.readout.DispersiveIQ` does the rotation). A hardware ``--arm``
        run sets this to the calibrated discrimination angle and reads with ``DispersiveIQ(angle=0.0)``.
    settle_us:
        Read-latency margin (µs) carried on the branch, lowered to a **processor-blocking**
        ``wait_auto(settle_us, ros=True)`` before ``read_and_jump`` so the branch reads a *fresh*
        accumulator (see :class:`~strumento.core.pulses.Branch`). Defaults to
        :data:`DEFAULT_FEEDBACK_SETTLE_US`. It is not a plain idle before the branch: a plain
        ``wait`` only shifts the reference time and never stalls the processor, so the read would be
        stale. ``wait_auto`` keys off the readout window end directly, so the window is not part of
        this margin — only the ADC-transfer/tProc read latency after it.
    hardware_loop:
        Emit the rounds as one bounded tProc loop (:class:`~strumento.core.pulses.Repeat`) instead
        of unrolling them. **Read this first:** qick counts trigger *macros* at assembly, so a
        ``Measure`` inside a hardware loop is counted **once** however many times the tProc
        performs it — the acquisition then under-declares its reads and those rounds' reads cannot
        be retrieved. Unrolled (the default) keeps the static read count equal to the dynamic one.
    line:
        Wiring line override for the readout probe, forwarded to
        :meth:`~strumento.packs.cqed.library.readout.ReadoutOps.dispersive_measure`. ``None``
        (default) resolves the normal ``("readout", "probe")`` line; pass e.g.
        ``("qubit", "drive")`` for a single-generator instance with no dedicated readout DAC line
        (the same workaround :class:`~strumento.packs.cqed.experiments.resonator_spectroscopy.ResonatorSpectroscopy`
        uses against ``loopback_demo``).

    Notes
    -----
    On hardware the two branch arms have different timeline lengths (the cursor advance past the
    corrective π lives *inside* the taken arm), so a shot that needed no correction reaches the
    next op earlier by the π's duration. Harmless for reset; pad explicitly if a downstream
    sequence needs the two paths aligned.
    """
    if int(rounds) < 1:
        raise ValueError(f"active_reset needs rounds >= 1; got {rounds}")

    channel = dev.wiring.readout.adc_chs[0] if adc is None else int(adc)
    settle = DEFAULT_FEEDBACK_SETTLE_US if settle_us is None else float(settle_us)
    branch_thr = threshold if branch_threshold is None else float(branch_threshold)
    correction = Seq().play(replace(dev.qubit.ge_pi(), label=RESET_PI_LABEL))

    def one_round() -> Seq:
        return (
            dev.readout.dispersive_measure(line=line, readout_phase_deg=readout_phase_deg)
            .branch(adc=channel, threshold=branch_thr, test=test, then=correction,
                    component=component, settle_us=settle)
        )

    if hardware_loop:
        return Seq().repeat(one_round(), int(rounds))
    out = Seq()
    for _ in range(int(rounds)):
        out += one_round()
    return out


def prepare(dev, target, *, via: str = "analytic", solution=None, drive_map=None, gain_calib=None, dac_rate=None) -> Seq:
    """State preparation addressed by *what it does* (D17): analytic ladder or OC import.

    ``via="analytic"`` builds the multiphoton ladder: ``ge_pi · ef_pi · f0g1(photon=k)``
    repeated for each photon ``k = 0 .. n-1``. Works for any ``Fock(n)`` whose sideband
    transitions are calibrated in ``calib.multiphoton["f0-g1"]``.

    ``via="oc"`` imports a solved waveform through
    :func:`~strumento.core.pulses.from_solution` (pass ``solution``/``drive_map``/``gain_calib``).
    """
    if via == "oc":
        if solution is None or drive_map is None or gain_calib is None:
            raise ValueError("via='oc' needs solution=, drive_map=, and gain_calib=")
        return from_solution(solution, drive_map=drive_map, gain_calib=gain_calib, dac_rate=dac_rate)

    if isinstance(target, Fock):
        n = target.n
        if n < 1:
            return Seq()  # Fock(0) = vacuum, no preparation needed
        q = dev.qubit
        seq = Seq()
        for photon in range(n):
            seq.play(q.ge_pi())
            seq.play(q.ef_pi())
            seq.play(dev.sideband.f0g1(photon=photon))
        return seq
    raise NotImplementedError(
        f"analytic preparation for {target!r} is not implemented "
        f"(Fock(n) is; or pass via='oc' with a solved pulse)"
    )
