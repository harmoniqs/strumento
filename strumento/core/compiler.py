"""Compiler — lower the pulse IR onto a qick program via an envelope-type registry (D22/D23).

The compiler walks a :class:`~strumento.core.pulses.Seq`, resolves each pulse's generator
through the :class:`~strumento.core.wiring.Wiring`, and emits the program build calls
(``declare_gen`` / ``add_gauss`` / ``add_pulse`` / ``pulse`` / ``delay`` / ``trigger`` /
``declare_readout``). It does this by calling methods on an *injected* program object — so this
module imports **no qick**. A real :class:`~strumento.core.program.StrumentoProgram` and a test
recording-fake both satisfy the same duck-typed interface.

Two design points:
  * **Envelope registry (D22).** Lowering dispatches on ``type(envelope)`` through a registry
    (``@lowers(Gaussian)``), never an ``if shape == …`` ladder. A pack extends the compiler by
    registering its envelope + lowering (``tweezer`` registers ``PolyChirp`` there). Core registers
    ``Gaussian``/``FlatTop``/``Const``/``Arb``; packs add their own (``tweezer`` adds ``PolyChirp``).
  * **Gain (D23).** ``Pulse.gain`` is already a v2 fraction by the time it reaches the compiler
    (the cqed factory converted it), so the compiler passes ``gain`` straight through — no
    int→fraction step here.

Envelopes are deduped on ``(generator, envelope)``; the frozen envelope is the hash key.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from typing import cast

import numpy as np

from strumento.core.pulses import (
    Arb,
    Branch,
    Const,
    Delay,
    FlatTop,
    Gaussian,
    Measure,
    Pulse,
    Raw,
    Repeat,
    Segment,
    Seq,
)
from strumento.core.sweeps import Loop, Sweep
from strumento.core.wiring import Wiring

#: default readout window (µs) when the instance/Measure supplies none.
DEFAULT_RO_LENGTH_US = 1.0

#: default Gaussian total window as a multiple of sigma.
_GAUSS_LENGTH_SIGMAS = 4.0

#: the two comparisons tProc-v2 read_and_jump supports (others: invert the branches).
_V2_BRANCH_TESTS = frozenset({">=", "<"})


def _realize_threshold(threshold: float | str) -> int | str:
    """Realize a :class:`~strumento.core.pulses.Branch` threshold to what v2 ``read_and_jump`` wants.

    tProc-v2 compares a 24-bit **signed integer** accumulator value (or a register, passed by name).
    The IR carries a float, so a scalar is rounded — with **round-half-away-from-zero**, not Python's
    built-in :func:`round`, which is banker's rounding and sends the midpoint ``0.5 -> 0`` and
    ``2.5 -> 2``. A comparison boundary that snaps a half to an even integer (and asymmetrically
    across zero) is a foot-gun; half-away-from-zero keeps ``+0.5 -> +1`` / ``-0.5 -> -1``, symmetric
    about zero, which is the intuitive boundary. On ``--arm`` the threshold is a large calibrated
    accumulator value where the rule is immaterial (fractional part far from ½); this only bites the
    small/half-valued defaults such as the ``--sim`` back-compat ``0.5`` (which now realizes to 1).
    """
    if isinstance(threshold, str):
        return threshold
    # int(x) truncates toward zero, so (x + 0.5) for x>=0 is floor(x+0.5) = round-half-up, and
    # (x - 0.5) for x<0 is ceil(x-0.5) = round-half-down — together, round-half-away-from-zero.
    return int(threshold + 0.5) if threshold >= 0 else int(threshold - 0.5)


# --------------------------------------------------------------------------------------
# Envelope lowering registry (D22)
# --------------------------------------------------------------------------------------

@dataclass(frozen=True)
class EnvelopeLowering:
    """How one envelope type reaches the hardware.

    ``declare`` (optional) adds the envelope's sample data to the program once per
    ``(generator, envelope)`` (e.g. ``add_gauss`` / ``add_envelope``); ``None`` means the shape
    needs no waveform memory (``Const``). ``pulse_kwargs`` returns the ``add_pulse`` style
    kwargs — it receives the live compiler so swept fields (e.g. a swept flat-top length) can be
    resolved through :meth:`Compiler._field_value`.
    """

    declare: Callable[..., None] | None
    pulse_kwargs: Callable[..., dict]


def _arb_style_kwargs(compiler: Compiler, env, env_name: str) -> dict:
    return {"style": "arb", "envelope": env_name}


_LOWERINGS: dict[type, EnvelopeLowering] = {}


def register_envelope(
    env_type: type,
    declare: Callable[..., None] | None = None,
    *,
    pulse_kwargs: Callable[..., dict] | None = None,
) -> None:
    """Register a lowering for an envelope type (packs call this to extend the compiler).

    ``declare(program, ch, name, envelope)`` writes the envelope's samples (omit for shapes with
    no waveform memory); ``pulse_kwargs(compiler, envelope, env_name)`` returns the ``add_pulse``
    style kwargs (default: ``style="arb"`` referencing the declared envelope).
    """
    _LOWERINGS[env_type] = EnvelopeLowering(declare, pulse_kwargs or _arb_style_kwargs)


def lowers(env_type: type, *, pulse_kwargs: Callable[..., dict] | None = None):
    """Decorator form of :func:`register_envelope` (the decorated fn is the declare hook)."""

    def deco(fn):
        register_envelope(env_type, fn, pulse_kwargs=pulse_kwargs)
        return fn

    return deco


def _lowering_for(envelope) -> EnvelopeLowering:
    low = _LOWERINGS.get(type(envelope))
    if low is None:
        raise TypeError(
            f"no lowering registered for envelope type {type(envelope).__name__!r}; "
            f"a pack must register_envelope() it before it can be compiled"
        )
    return low


@lowers(Gaussian)
def _lower_gaussian(program, ch, name, env: Gaussian) -> None:
    program.add_gauss(ch=ch, name=name, sigma=env.sigma_us, length=env.sigma_us * _GAUSS_LENGTH_SIGMAS)


def _flattop_kwargs(compiler: Compiler, env: FlatTop, env_name: str) -> dict:
    # v2 flat_top: the declared envelope is the (gaussian) ramp; the flat part is `length`
    return {"style": "flat_top", "envelope": env_name, "length": compiler._field_value(env.length_us)}


@lowers(FlatTop, pulse_kwargs=_flattop_kwargs)
def _lower_flattop(program, ch, name, env: FlatTop) -> None:
    program.add_gauss(ch=ch, name=name, sigma=env.ramp_sigma_us, length=env.ramp_sigma_us * _GAUSS_LENGTH_SIGMAS)


def _const_kwargs(compiler: Compiler, env: Const, env_name: str) -> dict:
    return {"style": "const", "length": compiler._field_value(env.length_us)}


register_envelope(Const, None, pulse_kwargs=_const_kwargs)


@lowers(Arb)
def _lower_arb(program, ch, name, env: Arb) -> None:
    """Write an imported waveform (the D17 import path): normalized [-1, 1] samples -> DAC units.

    Scaling uses the generator's ``maxv`` when the program carries a soccfg (the real path);
    a bare recording fake gets the normalized samples unscaled. Sample counts are padded to the
    generator's ``samps_per_clk`` grid so v2's envelope-length validator passes.
    """
    idata = np.asarray(env.idata, dtype=float)
    qdata = np.asarray(env.qdata, dtype=float) if env.qdata is not None else None

    # A real program carries a subscriptable soccfg (a dict snapshot or a QickConfig); a bare
    # recording fake either lacks one or (with a blanket ``__getattr__`` shim, e.g. the CLI's
    # _RecordingProgram) returns a non-subscriptable stand-in. Probe the ``gens`` table by
    # subscript and fall back to the unscaled recording path when it is not indexable — never scale
    # against a stand-in, and never require a specific concrete type (QickConfig is not a dict).
    soccfg = getattr(program, "soccfg", None)
    try:
        gens = None if soccfg is None else soccfg["gens"]
    except TypeError:
        gens = None
    if gens is not None:
        gen = gens[ch]
        maxv, spc = float(gen["maxv"]), int(gen.get("samps_per_clk", 1))
        pad = (-len(idata)) % spc
        if pad:
            idata = np.pad(idata, (0, pad))
            qdata = np.pad(qdata, (0, pad)) if qdata is not None else None
        idata = idata * maxv
        qdata = qdata * maxv if qdata is not None else None
    kw = {"idata": idata}
    if qdata is not None:
        kw["qdata"] = qdata
    program.add_envelope(ch=ch, name=name, **kw)


# --------------------------------------------------------------------------------------
# Compiler
# --------------------------------------------------------------------------------------


class Compiler:
    """Lower a :class:`Seq` onto a qick program in two passes.

    ``declare(seq)`` emits the setup calls (``declare_gen`` / envelopes / ``add_pulse`` /
    ``declare_readout`` / ``add_loop``) — run from the program's ``_initialize``. ``play(seq)``
    emits the timeline (``pulse`` / ``delay`` / ``trigger`` / ``Raw``) — run from ``_body``. The
    two passes share this object's name maps.

    **Loop structure is declared, not inferred (D36).** ``axes`` is the sequence's set of
    declared :class:`~strumento.core.sweeps.Loop` axes; the compiler emits one ``add_loop`` per
    declared axis — never one per swept parameter (a joint sweep binds several parameters to
    one axis). A sweep naming an undeclared axis is a typed error. A legacy per-name sweep
    auto-declares its axis with a ``DeprecationWarning``.
    """

    def __init__(self, program, wiring: Wiring, soccfg=None, *, axes: Iterable[Loop] = (),
                 trace=None, trig_offset: float = 0.0) -> None:
        self.program = program
        self.wiring = wiring
        self.soccfg = soccfg
        self._trace = trace
        #: readout trigger offset (µs), resolved by the caller (StrumentoProgram, calib-first /
        #: wiring-fallback / 0 — #98) and just played back here; the compiler never resolves it
        #: itself, exactly like it never resolves final_delay.
        self.trig_offset = trig_offset
        self._axes: dict[str, Loop] = {a.name: a for a in axes}
        self._declared_gens: dict[int, int] = {}  # dac_ch -> declared nqz
        self._env_names: dict[tuple[int, object], str] = {}
        self._pulse_names: dict[tuple, str] = {}
        self._declared_ros: set[int] = set()
        self._declared_ro_lengths: dict[int, object] = {}
        self._readout_config_names: dict[tuple, str] = {}
        self._declared_loops: set[str] = set()
        self._label_counter: int = 0
        # site key (see _site_key) -> the qick time-param tag stamped on a swept Delay at
        # lowering time, so the program can read the compiled delay axis back (D37).
        self._delay_tags: dict[tuple, str] = {}

    # -- declaration pass --------------------------------------------------------------

    def declare(self, seq: Seq) -> None:
        pulses = list(self._all_pulses(seq))

        # loops first (D36): one add_loop per DECLARED axis; a sweep_value refers to its axis.
        # Legacy unbound sweeps auto-declare a per-name axis (deprecated) so old sequences
        # keep compiling during the migration.
        for name, axis in self._axes.items():
            if name not in self._declared_loops:
                self.program.add_loop(name, axis.points)
                self._declared_loops.add(name)
        bound: set[str] = set()
        for sw in self._sweeps_in(seq):
            if sw.on is not None:
                if sw.on not in self._axes:
                    raise ValueError(
                        f"sweep {sw.name or f'[{sw.start}, {sw.stop}]'} is bound to undeclared "
                        f"axis {sw.on!r}: declare it on the program "
                        f"(axes=[Loop({sw.on!r}, n)]) — loop structure is explicit (D36)"
                    )
                bound.add(sw.on)
            elif sw.name not in self._declared_loops:
                self.program.add_loop(sw.name, sw.points)
                self._declared_loops.add(cast(str, sw.name))  # Sweep requires a name when unbound
        unused = set(self._axes) - bound
        if unused:
            warnings.warn(
                f"declared loop axis/axes {sorted(unused)} have no sweeps bound — they add an "
                f"acquisition dimension without sweeping anything",
                UserWarning,
                stacklevel=2,
            )

        self._declare_readouts(seq, pulses)

        for p in pulses:
            gen = self._gen(p)
            declared_nqz = self._declared_gens.get(gen.dac_ch)
            if declared_nqz is None:
                if self._trace:
                    self._trace.emit("declare", "gen", ch=gen.dac_ch, nqz=gen.nyquist,
                                     line=f"{p.line.line}.{p.line.role}")
                self.program.declare_gen(ch=gen.dac_ch, nqz=gen.nyquist)
                self._declared_gens[gen.dac_ch] = gen.nyquist
            elif declared_nqz != gen.nyquist:
                raise ValueError(
                    f"generator conflict on DAC ch {gen.dac_ch}: declared with nqz="
                    f"{declared_nqz}, but {p.line.line}.{p.line.role} wants nqz={gen.nyquist} "
                    f"(one physical generator cannot serve two Nyquist zones in one program)"
                )

        for p in pulses:
            ch = self._gen_ch(p)
            low = _lowering_for(p.envelope)
            env_key = (ch, p.envelope)
            if env_key not in self._env_names:
                name = f"env{ch}_{len(self._env_names)}"
                if low.declare is not None:
                    low.declare(self.program, ch, name, p.envelope)
                self._env_names[env_key] = name

        for p in pulses:
            ch = self._gen_ch(p)
            pkey = self._pulse_key(p, ch)
            if pkey not in self._pulse_names:
                name = f"pulse{ch}_{len(self._pulse_names)}"
                low = _lowering_for(p.envelope)
                style_kw = low.pulse_kwargs(self, p.envelope, self._env_names[(ch, p.envelope)])
                if p.readout is not None:
                    style_kw["ro_ch"] = p.readout.adc_ch
                self.program.add_pulse(
                    ch=ch, name=name,
                    freq=self._field_value(p.freq_mhz),
                    phase=self._field_value(p.phase_deg),
                    gain=self._field_value(p.gain),
                    **style_kw,
                )
                self._pulse_names[pkey] = name
                if p.readout is not None:
                    config_name = f"readout{p.readout.adc_ch}_{len(self._readout_config_names)}"
                    self.program.add_readoutconfig(
                        ch=p.readout.adc_ch,
                        name=config_name,
                        freq=self._field_value(p.freq_mhz),
                        phase=p.readout.phase_deg,
                        gen_ch=ch,
                    )
                    self._readout_config_names[pkey] = config_name

    # -- play (timeline) pass ----------------------------------------------------------

    def play(self, seq: Seq) -> None:
        """Lower the timeline under the cursor model (see ``docs/timeline.md``).

        A single reference cursor advances down the sequence. A **sequentially** played pulse
        (``start_us is None``) plays at the cursor, then the cursor advances past it via
        ``delay_auto`` — duration-aware, so ``play(a).play(b)`` means *b after a* without the
        caller computing ``a``'s length. A ``Segment`` (``play_parallel``) plays its pulses at the
        cursor together (overlap/stagger by per-pulse offset), then advances once past the longest.
        A pulse with an explicit ``start_us`` is placed **absolutely** and does *not* advance the
        cursor (you opted into manual timing). ``Delay``/``wait`` advances the cursor explicitly.
        """
        for op in seq.ops:
            if isinstance(op, Pulse):
                self._play_pulse(op)
                if op.start_us is None:
                    if self._trace:
                        self._trace.emit("play", "delay_auto", after="pulse", label=op.label)
                    self.program.delay_auto(0.0)          # advance past this pulse (duration-aware)
            elif isinstance(op, Delay):
                if self._trace:
                    self._trace.emit("play", "delay", duration_us=op.duration_us)
                if isinstance(op.duration_us, Sweep):
                    # a swept delay gets a tag so qick's get_time_param can read the compiled
                    # per-step durations back (D37) — the same register-step truth as pulses.
                    tag = f"strumento_delay_{len(self._delay_tags)}"
                    self._delay_tags[self._site_key(op.duration_us, "time_us")] = tag
                    self.program.delay(self._field_value(op.duration_us), tag=tag)
                else:
                    self.program.delay(self._field_value(op.duration_us))
            elif isinstance(op, Raw):
                if self._trace:
                    self._trace.emit("play", "raw", fn=op.fn.__name__ if hasattr(op.fn, "__name__") else "lambda")
                op.fn(self.program)
            elif isinstance(op, Segment):
                if self._trace:
                    self._trace.emit("play", "segment", n_pulses=len(op.pulses))
                for p in op.pulses:
                    self._play_pulse(p)                   # within a segment, start_us is an offset
                self.program.delay_auto(0.0)              # advance once past the longest pulse
            elif isinstance(op, Measure):
                adc = self.wiring.readout.adc_chs[0]
                if self._trace:
                    self._trace.emit("play", "measure", adc=adc, t=self.trig_offset)
                self.program.trigger(ros=[adc], t=self.trig_offset)
                self.program.delay_auto(0.0, gens=False, ros=True)
            elif isinstance(op, Branch):
                if self._trace:
                    self._trace.emit("play", "branch", adc=op.adc, threshold=op.threshold, test=op.test)
                self._play_branch(op)
            elif isinstance(op, Repeat):
                if self._trace:
                    self._trace.emit("play", "repeat", count=op.count)
                self._play_repeat(op)
            else:  # pragma: no cover - defensive
                raise TypeError(f"cannot lower op {type(op).__name__!r}")

    def _play_branch(self, op: Branch) -> None:
        """Lower a measurement-conditional branch to real tProc-v2 flow control.

        Pattern (no test negation needed): a processor-blocking wait so the accumulator is fresh,
        then read-and-jump to the THEN label when the test passes; the else block falls through and
        jumps past THEN to END::

            wait_auto(settle_us, ros=True)      # BLOCK until the readout is done (fresh accumulator)
            read_and_jump(ro, comp, thresh, test, THEN)
            <else block>
            jump(END)
            label(THEN); <then block>
            label(END)

        The ``wait_auto`` is not optional. ``read_and_jump`` runs at tProc *execution* time, off the
        timed pulse queue, and qick's ``read_input`` returns a **stale value** unless the readout has
        completed. A preceding ``Delay``/``wait`` only advances the *reference* time (it does not
        stall the processor — qick's ``Wait`` macro leaves the timestamps untouched), so only this
        blocking ``wait_auto`` guarantees freshness. It waits on the readout window end
        (``ros=True``, ``gens=False``) plus the branch's ``settle_us`` ADC-transfer margin — the v2
        port of the tProc-v1 idiom of a ``wait_all`` before every ``read``. A
        ``wait_auto`` with no readout in scope compiles to nothing, so it is safe unconditionally.
        """
        if op.test not in _V2_BRANCH_TESTS:
            raise ValueError(
                f"Branch test {op.test!r} is not a tProc-v2 comparison; use one of "
                f"{sorted(_V2_BRANCH_TESTS)} (v2 read_and_jump supports only these two — "
                f"invert the branches to express the others)"
            )
        n = self._next_label()
        then_lbl, end_lbl = f"strumento_then_{n}", f"strumento_end_{n}"
        # Freshness gate (see the pattern above): stall the tProc until the readout has pushed its
        # accumulator, else read_and_jump conditions on a stale value.
        if self._trace:
            self._trace.emit("play", "wait_auto", ros=True, gens=False, t=op.settle_us)
        self.program.wait_auto(op.settle_us, gens=False, ros=True)
        # v2 read_and_jump wants an int threshold (a 24-bit signed ADC-accumulator value) or a
        # register name; realize a float in the IR half-away-from-zero (see _realize_threshold).
        threshold = _realize_threshold(op.threshold)
        self.program.read_and_jump(
            ro_ch=op.adc, component=op.component, threshold=threshold, test=op.test, label=then_lbl
        )
        if op.else_ is not None:
            self.play(op.else_)
        self.program.jump(end_lbl)
        self.program.label(then_lbl)
        self.play(op.then)
        self.program.label(end_lbl)

    def _play_repeat(self, op: Repeat) -> None:
        """Lower a bounded repeat to a tProc-v2 hardware loop (``open_loop``/``close_loop``)."""
        n = self._next_label()
        self.program.open_loop(op.count, name=f"strumento_repeat_{n}")
        self.play(op.body)
        self.program.close_loop()

    def _next_label(self) -> int:
        self._label_counter += 1
        return self._label_counter

    # -- helpers -----------------------------------------------------------------------

    def _play_pulse(self, p: Pulse) -> None:
        ch = self._gen_ch(p)
        pkey = self._pulse_key(p, ch)
        name = self._pulse_names[pkey]
        t = p.start_us or 0.0
        config_name = self._readout_config_names.get(pkey)
        if config_name is not None:
            assert p.readout is not None  # a readout config is declared only for a readout pulse
            if self._trace:
                self._trace.emit("play", "readout_config", adc=p.readout.adc_ch, name=config_name, t=t)
            self.program.send_readoutconfig(ch=p.readout.adc_ch, name=config_name, t=t)
        if self._trace:
            self._trace.emit("play", "pulse", ch=ch, name=name, t=t,
                             line=f"{p.line.line}.{p.line.role}", label=p.label,
                             freq=p.freq_mhz, gain=p.gain,
                             envelope=type(p.envelope).__name__)
        self.program.pulse(ch=ch, name=name, t=t)

    def _gen(self, p: Pulse):
        """Resolve the generator for a pulse, handling a swept frequency.

        A frequency sweep must stay within one generator's band — resolving both endpoints to
        different generators is a band-boundary crossing a single ``QickSweep1D`` can't express.
        """
        freq = p.freq_mhz
        if isinstance(freq, Sweep):
            g_lo = self.wiring.resolve(p.line, freq.start)
            g_hi = self.wiring.resolve(p.line, freq.stop)
            if g_lo != g_hi:
                raise ValueError(
                    f"frequency sweep {freq.name!r} on {p.line.line}.{p.line.role} spans a band "
                    f"boundary: {freq.start} MHz and {freq.stop} MHz resolve to different "
                    f"generators (a single generator must cover the whole sweep)"
                )
            return g_lo
        return self.wiring.resolve(p.line, freq)

    def _gen_ch(self, p: Pulse) -> int:
        return self._gen(p).dac_ch

    def _field_value(self, value):
        """Resolve a possibly-swept field to what ``add_pulse``/``delay`` should receive.

        A :class:`Sweep` becomes the program's sweep handle (``sweep_value`` returns a
        ``QickSweep1D`` on the real program) over **its axis** — the declared axis it is bound
        to (``on=``), or its own name for a legacy unbound sweep. Keeping the qick sweep object
        behind ``program.sweep_value`` is what keeps this module qick-free.
        """
        if isinstance(value, Sweep):
            return self.program.sweep_value(value.on or value.name, value.start, value.stop)
        return value

    @staticmethod
    def _pulse_key(p: Pulse, ch: int) -> tuple:
        return (ch, p.freq_mhz, p.phase_deg, p.gain, p.envelope, p.readout)

    def _declare_readouts(self, seq: Seq, pulses: list[Pulse]) -> None:
        requested: dict[int, float] = {}
        for p in pulses:
            if p.readout is not None:
                length = p.readout.window_us
                if length is None:
                    length = self._default_readout_length()
                if isinstance(length, Sweep):
                    raise ValueError("a dynamically configured readout window cannot be swept")
                previous = requested.setdefault(p.readout.adc_ch, length)
                if previous != length:
                    raise ValueError(
                        f"readout ADC{p.readout.adc_ch} has conflicting declared windows "
                        f"{previous} and {length}"
                    )
        if self._needs_readout(seq):
            requested.setdefault(self.wiring.readout.adc_chs[0], self._default_readout_length())
        for adc, length in requested.items():
            if adc not in self._declared_ros:
                self.program.declare_readout(ch=adc, length=length)
                self._declared_ros.add(adc)
                self._declared_ro_lengths[adc] = length

    def _default_readout_length(self):
        length = self.wiring.readout.lengths_us or DEFAULT_RO_LENGTH_US
        return length[0] if isinstance(length, list) else length

    @staticmethod
    def _all_pulses(seq: Seq) -> Iterator[Pulse]:
        """Every pulse in the sequence, recursing into Branch (then/else) and Repeat (body)."""
        for op in seq.ops:
            if isinstance(op, Pulse):
                yield op
            elif isinstance(op, Segment):
                yield from op.pulses
            elif isinstance(op, Branch):
                yield from Compiler._all_pulses(op.then)
                if op.else_ is not None:
                    yield from Compiler._all_pulses(op.else_)
            elif isinstance(op, Repeat):
                yield from Compiler._all_pulses(op.body)

    @staticmethod
    def _needs_readout(seq: Seq) -> bool:
        """True if the sequence measures or branches on a readout, anywhere (recursively)."""
        for op in seq.ops:
            if isinstance(op, (Measure, Branch)):
                return True
            if isinstance(op, Repeat) and Compiler._needs_readout(op.body):
                return True
            if isinstance(op, Branch) and (
                Compiler._needs_readout(op.then)
                or (op.else_ is not None and Compiler._needs_readout(op.else_))
            ):
                return True
        return False

    @staticmethod
    def _sweep_sites(seq: Seq) -> list[tuple[Sweep, str, Pulse | None]]:
        """Every ``(sweep, kind, host)`` swept-field site, in discovery order (recursive).

        ``kind`` is one of ``"gain_frac"`` / ``"freq_mhz"`` / ``"phase_deg"`` / ``"time_us"`` —
        the quantization domain the field lives in (D37 realization). ``host`` is the pulse
        carrying the field (``None`` for a swept :class:`Delay`)."""
        found: dict = {}

        def note(v: Sweep, kind: str, host) -> None:
            found.setdefault(Compiler._site_key(v, kind), (v, kind, host))

        def visit(s: Seq) -> None:
            for op in s.ops:
                pulses = op.pulses if isinstance(op, Segment) else ([op] if isinstance(op, Pulse) else [])
                for p in pulses:
                    if isinstance(p.gain, Sweep):
                        note(p.gain, "gain_frac", p)
                    if isinstance(p.freq_mhz, Sweep):
                        note(p.freq_mhz, "freq_mhz", p)
                    if isinstance(p.phase_deg, Sweep):
                        note(p.phase_deg, "phase_deg", p)
                    swept_length = getattr(p.envelope, "length_us", None)
                    if isinstance(swept_length, Sweep):
                        note(swept_length, "time_us", p)
                if isinstance(op, Delay) and isinstance(op.duration_us, Sweep):
                    note(op.duration_us, "time_us", None)
                if isinstance(op, Branch):
                    visit(op.then)
                    if op.else_ is not None:
                        visit(op.else_)
                if isinstance(op, Repeat):
                    visit(op.body)

        visit(seq)
        return list(found.values())

    @staticmethod
    def _site_key(sw: Sweep, kind: str) -> tuple:
        """The identity of a swept-field site: legacy sweeps by name, bound sweeps by their
        ``(axis, range)`` — always qualified by the field's quantization domain."""
        return (sw.name, kind) if sw.on is None else ("bound", sw.on, sw.start, sw.stop, kind)

    @staticmethod
    def _sweeps_in(seq: Seq) -> list[Sweep]:
        """Unique sweeps across pulse gain/freq/phase, envelope length, and delay-duration
        fields (recursive). Legacy sweeps dedupe by name; bound sweeps by ``(on, start, stop)``."""
        found: dict = {}
        for sw, _kind, _host in Compiler._sweep_sites(seq):
            key = sw.name if sw.on is None else ("bound", sw.on, sw.start, sw.stop)
            found.setdefault(key, sw)
        return list(found.values())
