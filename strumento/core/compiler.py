"""Compiler — lower the pulse IR onto a qick program via an envelope-type registry (D22/D23).

The compiler walks a :class:`~strumento.core.pulses.Seq`, resolves each pulse's generator
through the :class:`~strumento.core.wiring.Wiring`, and emits the program build calls
(``declare_gen`` / ``add_gauss`` / ``add_pulse`` / ``pulse`` / ``delay`` / ``trigger`` /
``declare_readout``). It does this by calling methods on an *injected* program object — so this
module imports **no qick**. A real :class:`~strumento.core.program.StrumentoProgram` and a test
recording-fake both satisfy the same duck-typed interface.

Two design points from the spec:
  * **Envelope registry (D22).** Lowering dispatches on ``type(envelope)`` through a registry
    (``@lowers(Gaussian)``), never an ``if shape == …`` ladder. A pack extends the compiler by
    registering its envelope + lowering (``tweezer`` registers ``PolyChirp`` there). Core registers
    ``Gaussian``/``FlatTop``/``Const``/``Arb``; packs add their own (``tweezer`` adds ``PolyChirp``).
  * **Gain (D23).** ``Pulse.gain`` is already a v2 fraction by the time it reaches the compiler
    (the cqed factory converted it), so the compiler passes ``gain`` straight through — no
    int→fraction step here.

Envelopes are deduped on ``(generator, envelope)``; the frozen envelope is the hash key (R6).
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass

import numpy as np

from strumento.core.pulses import (
    Arb,
    Const,
    Delay,
    FlatTop,
    Gaussian,
    Measure,
    Pulse,
    Raw,
    Segment,
    Seq,
)
from strumento.core.sweeps import Sweep
from strumento.core.wiring import Wiring

#: default readout window (µs) when the instance/Measure supplies none (refined in Task 16).
DEFAULT_RO_LENGTH_US = 1.0

#: default Gaussian total window as a multiple of sigma (slice-1 convention).
_GAUSS_LENGTH_SIGMAS = 4.0


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
    """Write an imported waveform (§6.5): normalized [-1, 1] samples -> DAC units.

    Scaling uses the generator's ``maxv`` when the program carries a soccfg (the real path);
    a bare recording fake gets the normalized samples unscaled. Sample counts are padded to the
    generator's ``samps_per_clk`` grid so v2's envelope-length validator passes.
    """
    idata = np.asarray(env.idata, dtype=float)
    qdata = np.asarray(env.qdata, dtype=float) if env.qdata is not None else None

    soccfg = getattr(program, "soccfg", None)
    if soccfg is not None:
        gen = soccfg["gens"][ch]
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


def _envelope_length_us(envelope) -> float:
    """Total time footprint of an envelope (µs) — used by :func:`quantize`."""
    if isinstance(envelope, Gaussian):
        return envelope.sigma_us * _GAUSS_LENGTH_SIGMAS
    if isinstance(envelope, FlatTop):
        return envelope.length_us + 2 * envelope.ramp_sigma_us * _GAUSS_LENGTH_SIGMAS
    if isinstance(envelope, Const):
        return envelope.length_us
    raise TypeError(f"cannot compute length for envelope type {type(envelope).__name__!r}")


# --------------------------------------------------------------------------------------
# Compiler
# --------------------------------------------------------------------------------------


class Compiler:
    """Lower a :class:`Seq` onto a qick program in two passes.

    ``declare(seq)`` emits the setup calls (``declare_gen`` / envelopes / ``add_pulse`` /
    ``declare_readout``) — run from the program's ``_initialize``. ``play(seq)`` emits the
    timeline (``pulse`` / ``delay`` / ``trigger`` / ``Raw``) — run from ``_body``. The two
    passes share this object's name maps.
    """

    def __init__(self, program, wiring: Wiring, soccfg=None) -> None:
        self.program = program
        self.wiring = wiring
        self.soccfg = soccfg
        self._declared_gens: dict[int, int] = {}  # dac_ch -> declared nqz
        self._env_names: dict[tuple[int, object], str] = {}
        self._pulse_names: dict[tuple, str] = {}
        self._declared_ros: set[int] = set()
        self._declared_loops: set[str] = set()

    # -- declaration pass --------------------------------------------------------------

    def declare(self, seq: Seq) -> None:
        pulses = list(self._all_pulses(seq))

        # loops first: one add_loop per unique swept-field name (a sweep_value refers to it)
        for sw in self._sweeps_in(seq):
            if sw.name not in self._declared_loops:
                self.program.add_loop(sw.name, sw.points)
                self._declared_loops.add(sw.name)

        for p in pulses:
            gen = self._gen(p)
            declared_nqz = self._declared_gens.get(gen.dac_ch)
            if declared_nqz is None:
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
                self.program.add_pulse(
                    ch=ch, name=name,
                    freq=self._field_value(p.freq_mhz),
                    phase=self._field_value(p.phase_deg),
                    gain=self._field_value(p.gain),
                    **style_kw,
                )
                self._pulse_names[pkey] = name

        if any(isinstance(op, Measure) for op in seq.ops):
            adc = self.wiring.readout.adc_chs[0]
            if adc not in self._declared_ros:
                length = self.wiring.readout.lengths_us or DEFAULT_RO_LENGTH_US
                if isinstance(length, list):
                    length = length[0]
                self.program.declare_readout(ch=adc, length=length)
                self._declared_ros.add(adc)

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
                    self.program.delay_auto(0.0)          # advance past this pulse (duration-aware)
            elif isinstance(op, Delay):
                self.program.delay(self._field_value(op.duration_us))
            elif isinstance(op, Raw):
                op.fn(self.program)
            elif isinstance(op, Segment):
                for p in op.pulses:
                    self._play_pulse(p)                   # within a segment, start_us is an offset
                self.program.delay_auto(0.0)              # advance once past the longest pulse
            elif isinstance(op, Measure):
                adc = self.wiring.readout.adc_chs[0]
                self.program.trigger(ros=[adc], t=0)
            else:  # pragma: no cover - defensive
                raise TypeError(f"cannot lower op {type(op).__name__!r}")

    # -- helpers -----------------------------------------------------------------------

    def _play_pulse(self, p: Pulse) -> None:
        ch = self._gen_ch(p)
        name = self._pulse_names[self._pulse_key(p, ch)]
        self.program.pulse(ch=ch, name=name, t=(p.start_us or 0.0))

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
        ``QickSweep1D`` on the real program); a scalar passes through unchanged. Keeping the
        qick sweep object behind ``program.sweep_value`` is what keeps this module qick-free.
        """
        if isinstance(value, Sweep):
            return self.program.sweep_value(value.name, value.start, value.stop)
        return value

    @staticmethod
    def _pulse_key(p: Pulse, ch: int) -> tuple:
        return (ch, p.freq_mhz, p.phase_deg, p.gain, p.envelope)

    @staticmethod
    def _all_pulses(seq: Seq) -> Iterator[Pulse]:
        for op in seq.ops:
            if isinstance(op, Pulse):
                yield op
            elif isinstance(op, Segment):
                yield from op.pulses

    @staticmethod
    def _sweeps_in(seq: Seq) -> list[Sweep]:
        """Unique (by name) sweeps across pulse gain/freq and delay-duration fields."""
        found: dict[str, Sweep] = {}
        for op in seq.ops:
            pulses = op.pulses if isinstance(op, Segment) else ([op] if isinstance(op, Pulse) else [])
            for p in pulses:
                fields = [p.gain, p.freq_mhz, p.phase_deg]
                if isinstance(getattr(p.envelope, "length_us", None), Sweep):
                    fields.append(p.envelope.length_us)
                for v in fields:
                    if isinstance(v, Sweep):
                        found.setdefault(v.name, v)
            if isinstance(op, Delay) and isinstance(op.duration_us, Sweep):
                found.setdefault(op.duration_us.name, op.duration_us)
        return list(found.values())


# --------------------------------------------------------------------------------------
# quantize — realized (DAC-grid-snapped) values (spec §6.1)
# --------------------------------------------------------------------------------------


@dataclass
class Realized:
    """The DAC-grid-realized values of a pulse (what the hardware actually plays)."""

    freq_mhz_actual: float
    length_us_actual: float
    gain_frac: float
    cycles: int


def quantize(pulse: Pulse, soccfg, wiring: Wiring) -> Realized:
    """Snap a pulse's frequency / length / gain to the generator's DAC resolution.

    Uses the soccfg's unit conversions (version-agnostic math). ``gain`` is realized to the
    generator's amplitude resolution via ``round(gain * maxv) / maxv``.
    """
    gen = wiring.resolve(pulse.line, pulse.freq_mhz)
    ch = gen.dac_ch

    reg = soccfg.freq2reg(pulse.freq_mhz, gen_ch=ch)
    freq_actual = soccfg.reg2freq(reg, gen_ch=ch)

    length_us = _envelope_length_us(pulse.envelope)
    cycles = soccfg.us2cycles(length_us, gen_ch=ch)
    length_actual = soccfg.cycles2us(cycles, gen_ch=ch)

    maxv = _maxv(soccfg, ch)
    gain_frac = round(pulse.gain * maxv) / maxv

    return Realized(
        freq_mhz_actual=freq_actual,
        length_us_actual=length_actual,
        gain_frac=gain_frac,
        cycles=int(cycles),
    )


def _maxv(soccfg, ch: int) -> float:
    for g in soccfg["gens"]:
        if g.get("tproc_ch") == ch:
            return g["maxv"]
    return soccfg["gens"][ch]["maxv"]
