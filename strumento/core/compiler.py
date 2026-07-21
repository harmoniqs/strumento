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
    registering its envelope + lowering (``tweezer`` registers ``PolyChirp`` there). Slice 1
    registers only ``Gaussian``; ``FlatTop``/``Arb`` land in later slices.
  * **Gain (D23).** ``Pulse.gain`` is already a v2 fraction by the time it reaches the compiler
    (the cqed factory converted it), so the compiler passes ``gain`` straight through — no
    int→fraction step here.

Envelopes are deduped on ``(generator, envelope)``; the frozen envelope is the hash key (R6).
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass

from strumento.core.pulses import (
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
from strumento.core.wiring import Wiring

#: default readout window (µs) when the instance/Measure supplies none (refined in Task 16).
DEFAULT_RO_LENGTH_US = 1.0

#: default Gaussian total window as a multiple of sigma (slice-1 convention).
_GAUSS_LENGTH_SIGMAS = 4.0


# --------------------------------------------------------------------------------------
# Envelope lowering registry (D22)
# --------------------------------------------------------------------------------------

_LOWERINGS: dict[type, Callable[..., None]] = {}


def register_envelope(env_type: type, fn: Callable[..., None]) -> None:
    """Register a lowering for an envelope type (packs call this to extend the compiler)."""
    _LOWERINGS[env_type] = fn


def lowers(env_type: type):
    """Decorator form of :func:`register_envelope`."""

    def deco(fn):
        _LOWERINGS[env_type] = fn
        return fn

    return deco


def _lower_envelope(program, ch: int, name: str, envelope) -> None:
    fn = _LOWERINGS.get(type(envelope))
    if fn is None:
        raise TypeError(
            f"no lowering registered for envelope type {type(envelope).__name__!r}; "
            f"a pack must register_envelope() it before it can be compiled"
        )
    fn(program, ch, name, envelope)


@lowers(Gaussian)
def _lower_gaussian(program, ch, name, env: Gaussian) -> None:
    program.add_gauss(ch=ch, name=name, sigma=env.sigma_us, length=env.sigma_us * _GAUSS_LENGTH_SIGMAS)


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
        self._declared_gens: set[int] = set()
        self._env_names: dict[tuple[int, object], str] = {}
        self._pulse_names: dict[tuple, str] = {}
        self._declared_ros: set[int] = set()

    # -- declaration pass --------------------------------------------------------------

    def declare(self, seq: Seq) -> None:
        pulses = list(self._all_pulses(seq))

        for p in pulses:
            ch = self._gen_ch(p)
            if ch not in self._declared_gens:
                gen = self.wiring.resolve(p.line, p.freq_mhz)
                self.program.declare_gen(ch=ch, nqz=gen.nyquist)
                self._declared_gens.add(ch)

        for p in pulses:
            ch = self._gen_ch(p)
            env_key = (ch, p.envelope)
            if env_key not in self._env_names:
                name = f"env{ch}_{len(self._env_names)}"
                _lower_envelope(self.program, ch, name, p.envelope)
                self._env_names[env_key] = name

        for p in pulses:
            ch = self._gen_ch(p)
            pkey = self._pulse_key(p, ch)
            if pkey not in self._pulse_names:
                name = f"pulse{ch}_{len(self._pulse_names)}"
                self.program.add_pulse(
                    ch=ch, name=name, style="arb",
                    envelope=self._env_names[(ch, p.envelope)],
                    freq=p.freq_mhz, phase=p.phase_deg, gain=p.gain,
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
        for op in seq.ops:
            if isinstance(op, Pulse):
                self._play_pulse(op)
            elif isinstance(op, Delay):
                self.program.delay(op.duration_us)
            elif isinstance(op, Raw):
                op.fn(self.program)
            elif isinstance(op, Segment):
                for p in op.pulses:
                    self._play_pulse(p)
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

    def _gen_ch(self, p: Pulse) -> int:
        return self.wiring.resolve(p.line, p.freq_mhz).dac_ch

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
