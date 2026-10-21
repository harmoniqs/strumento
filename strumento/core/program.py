"""``StrumentoProgram`` — an ``AveragerProgramV2`` driven by the strumento IR + compiler.

A program is constructed from a :class:`~strumento.core.device.Device` and a
:class:`~strumento.core.pulses.Seq` (plus optional sweeps). ``_initialize`` builds a
:class:`~strumento.core.compiler.Compiler` and runs its declaration pass (declare generators /
envelopes / readouts + one ``add_loop`` per sweep); ``_body`` runs the timeline pass. ``acquire``
wraps ``AveragerProgramV2.acquire``'s raw IQ into a :class:`~strumento.core.result.MeasurementResult`.

**Escape hatch:** a ``StrumentoProgram`` *is* an ``AveragerProgramV2``, so any raw qick call is
reachable (``.raw(fn)`` in a Seq, ``self.<method>`` from a hand-written ``_body``, …).

**The off-board bridge (D14).** :meth:`StrumentoProgram.to_compiled_job` turns a *built* program
into the :class:`~strumento.core.transport.CompiledJob` that crosses the wire to a job server. This
is the one v2-gated half of the D14 contract — it needs a really-compiled ``AveragerProgramV2`` —
and it lives here rather than in ``core/transport.py`` precisely so the transport stays pure.

This module imports ``qick`` — one of the qick-facing core modules. The raw→result
wrapping is factored out as :func:`_to_measurement_result` so it can be unit-tested without a
tProc-v2 board.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
from qick import QickConfig
from qick.asm_v2 import AveragerProgramV2

from strumento.core.compiler import Compiler
from strumento.core.pulses import Pulse, Seq
from strumento.core.result import MeasurementResult

#: default relax delay (µs) when the device calibration supplies none.
_DEFAULT_FINAL_DELAY_US = 1.0

#: default readout trigger offset (µs) when neither calibration nor wiring supplies one (#98).
_DEFAULT_TRIG_OFFSET_US = 0.0


def _resolve_trig_offset(device) -> float:
    """Resolve the readout trigger offset (µs): calibration-first, wiring-fallback, else 0 (#98).

    Mirrors ``final_delay``'s own resolution a few lines below in
    :meth:`StrumentoProgram.__init__` — a generic, duck-typed reach into
    ``device.calib``/``device.wiring`` via ``getattr(..., None)``, so ``core`` never learns a
    pack's calibration shape exists. ``calib.readout.trig_offset`` (the pack-populated field the
    ``Measure`` lowering never used to read) is authoritative; ``wiring.readout.trig_offset``
    (core-level, so any pack can populate it) is the fallback; ``0.0`` if neither is set — the
    literal ``core/compiler.py`` used to hardcode, so an uncalibrated device sees no behavior
    change.

    Factored out (like :func:`_to_measurement_result`) so the three-way fallback is unit-testable
    with lightweight duck-typed devices, no board and no tProc-v2 assembly needed.
    """
    ro_calib = getattr(device.calib, "readout", None)
    trig_offset = getattr(ro_calib, "trig_offset", None)
    if trig_offset is None:
        ro_wiring = getattr(device.wiring, "readout", None)
        trig_offset = getattr(ro_wiring, "trig_offset", None)
    return trig_offset if trig_offset is not None else _DEFAULT_TRIG_OFFSET_US


@dataclass
class Realized:
    """What the tProc will actually play for a pulse, read back from the *compiled* program.

    Every value comes from qick's ``get_pulse_param`` — the assembled, register-rounded
    parameter in ``add_pulse`` units — never from a second client-side conversion. A field swept
    in the IR comes back as the per-step array the loop engine will actually write (register-
    quantized step size included): stepping in lab units and re-converting per point does not
    reproduce the compiled register steps, so a hand-rolled round-trip is not a substitute.
    """

    freq_mhz_actual: float | np.ndarray
    length_us_actual: float | np.ndarray
    gain_frac: float | np.ndarray


def compiled_payload(prog) -> dict:
    """A built v2 program's compiled payload, JSON-safe — what rides in ``CompiledJob.program``.

    ``AveragerProgramV2.dump_prog()`` is qick's own "everything needed to execute this" dict (ASM,
    waveforms, envelopes, generator/readout declarations, the acquisition dims) and ``load_prog()``
    is its inverse. Using that pair rather than inventing a format is the whole reason a board agent
    can be dumb: the payload is qick's, so the executor is ``load_prog`` + ``acquire`` and nothing
    else, and a qick upgrade moves both ends together.

    The one transformation here is serialization: ``dump_prog`` returns live numpy arrays and
    ``Waveform`` objects, so the dict is pushed through qick's ``NpEncoder`` (envelope arrays →
    base64 + shape + dtype, which ``load_prog``'s ``decode_array`` reverses) and back. The result is
    JSON primitives all the way down, which is what makes the wire form byte-exact under
    ``json.dumps`` → ``json.loads`` with no custom decoder on either side.
    """
    from qick.helpers import NpEncoder

    return json.loads(json.dumps(prog.dump_prog(), cls=NpEncoder))


def acquire_config(prog, *, soft_avgs: int = 1) -> dict:
    """The ``CompiledJob.acquire`` block for a built program: how to run it and what shape comes back.

    Derived from the compiled program, never from the caller's intent — ``reads_per_shot`` is the
    ``trigger`` macro count qick itself accumulated per readout channel, and ``expts`` is the product
    of the loop dimensions *other than* the averaged one (so it is right whether ``reps`` is the
    outermost or the innermost loop). The board agent can therefore validate the declared shape
    against the program it reconstructed, instead of trusting a hand-written number.
    """
    dims = list(prog.loop_dims)
    del dims[prog.avg_level]        # the averaged axis is not an experiment axis
    return {
        "reps": int(prog.reps),
        "soft_avgs": int(soft_avgs),
        "ro_chs": [int(ch) for ch in prog.ro_chs],
        "reads_per_shot": [int(ro["trigs"]) for ro in prog.ro_chs.values()],
        "expts": int(np.prod(dims)) if dims else None,
    }


def _to_measurement_result(raw, xpts=None) -> MeasurementResult:
    """Wrap ``AveragerProgramV2.acquire``'s output into a :class:`MeasurementResult`.

    ``raw`` has shape ``(n_ch, n_reads, 2)`` — or ``(n_ch, n_reads, n_expts, 2)`` with sweeps —
    where the last axis is ``(I, Q)``. This surfaces the first readout channel's I/Q; the
    full raw array is retained on ``.raw`` for downstream reducers (the readout kinds).
    """
    arr = np.asarray(raw[0])  # first readout channel
    return MeasurementResult(xpts=xpts, avgi=arr[..., 0], avgq=arr[..., 1], raw=raw)


class StrumentoProgram(AveragerProgramV2):
    """A compiled strumento sequence, runnable via ``acquire`` on a (mock or real) soc.

    ``axes`` is the program's set of declared :class:`~strumento.core.sweeps.Loop` axes (D36):
    loop structure is stated by hand — each declared axis becomes one ``add_loop`` — and sweeps
    in the sequence bind to an axis with ``Sweep(start, stop, on=...)``. A legacy per-name
    sweep still auto-declares its axis (deprecated).
    """

    def __init__(
        self,
        device,
        seq: Seq | None = None,
        *,
        sweeps=(),
        axes=(),
        reps: int = 100,
        final_delay: float | None = None,
        soc=None,
        trace: bool = False,
        **kwargs,
    ) -> None:
        self.device = device
        self.seq = seq if seq is not None else Seq()
        self.sweeps = tuple(sweeps)
        self.axes = tuple(axes)
        self._compiler: Compiler | None = None
        self._trace = None
        if trace:
            from strumento.core.tracing import CompilerTrace
            self._trace = CompilerTrace()

        soccfg = soc if soc is not None else QickConfig(device.soccfg_snapshot)
        if final_delay is None:
            ro = getattr(device.calib, "readout", None)
            final_delay = getattr(ro, "relax_delay", None) or _DEFAULT_FINAL_DELAY_US
        self._trig_offset = _resolve_trig_offset(device)

        super().__init__(soccfg, reps=reps, final_delay=final_delay, **kwargs)

    def _initialize(self, cfg):
        # the compiler discovers sweeps in the sequence and registers their add_loop (a swept
        # pulse field carries its own Sweep bound to a declared axis), so we don't
        # double-register self.sweeps here.
        self._compiler = Compiler(self, self.device.wiring, self.soccfg, axes=self.axes,
                                  trace=self._trace, trig_offset=self._trig_offset)
        self._compiler.declare(self.seq)

    def _body(self, cfg):
        self._compiler.play(self.seq)

    def sweep_value(self, name: str, start, stop):
        """The compiler's handle for a swept field: a live ``QickSweep1D`` over loop ``name``.

        The :class:`~strumento.core.compiler.Compiler` calls this for any ``Sweep``-valued
        pulse/delay field (keeping the compiler qick-free); the returned object rides into
        ``add_pulse``/``delay`` exactly as v2 expects.
        """
        from qick.asm_v2 import QickSweep1D

        return QickSweep1D(name, start, stop)

    def acquire(self, soc, **kwargs) -> MeasurementResult:  # type: ignore[override]
        raw = super().acquire(soc, **kwargs)
        result = _to_measurement_result(raw, xpts=self.xpts())
        requested = getattr(self, "_requested_xpts", None)
        if requested is not None:
            result.meta["requested_xpts"] = np.asarray(requested, dtype=float)
        return result

    def trace(self) -> dict | None:
        """The compiler trace (if tracing was enabled). Returns None if ``trace=False``."""
        if self._trace is None:
            return None
        return {"entries": [{"phase": e.phase, "op": e.op, **e.detail} for e in self._trace.entries]}

    def dump(self) -> dict:
        """Full compiled program dump — strumento compiler state + qick compiled output.

        Uses stock qick APIs (``dump_prog()``, ``gen_chs``, ``ro_chs``, ``pulses``, ``prog_list``).
        No modified qick needed. See :func:`~strumento.core.tracing.dump_program`.
        """
        from strumento.core.tracing import dump_program
        return dump_program(self)

    def print_trace(self, *, max_asm: int = 50) -> None:
        """Print a human-readable dump of the compiled program + compiler trace.

        Shows both the strumento compiler's decisions (gen resolution, envelope dedup, cursor
        advance) and qick's compiled output (ASM instructions, waveforms, register values).
        """
        from strumento.core.tracing import print_program_dump
        print_program_dump(self, max_asm=max_asm)

    def realized(self, pulse: Pulse) -> Realized:
        """Read back what the hardware will actually play for ``pulse`` (D37).

        The pulse must be part of this program's sequence: its declared values are read back from
        the compiled program via qick's ``get_pulse_param``, so the answer is the register-rounded
        truth — for a swept field, the per-step array with the loop engine's quantized step size,
        which is exactly what a requested-units round-trip cannot reproduce.
        """
        assert self._compiler is not None  # set by _initialize, which __init__ runs via compile()
        ch = self._compiler._gen_ch(pulse)  # resolves the generator (swept-freq band check incl.)
        name = self._compiler._pulse_names.get(Compiler._pulse_key(pulse, ch))
        if name is None:
            raise ValueError(
                "this pulse is not declared in the program — build the program from a Seq "
                "containing it (declare dedups on (channel, freq, phase, gain, envelope))"
            )
        if self.binprog is None:
            # get_pulse_param reads the *compiled* params (loop step sizes known only then)
            self.compile()
        return Realized(
            freq_mhz_actual=self._readback_param(name, "freq"),
            length_us_actual=self._readback_param(name, "total_length"),
            gain_frac=self._readback_param(name, "gain"),
        )

    def _readback_param(self, name: str, par: str):
        val = self.get_pulse_param(name, par)
        if hasattr(val, "is_sweep"):  # swept → the realized per-step axis, not a QickParam handle
            val = self.get_pulse_param(name, par, as_array=True)
        return val

    def to_compiled_job(self, *, overlay_id: str = "", soft_avgs: int = 1):
        """This built program as a :class:`~strumento.core.transport.CompiledJob` (D14).

        The off-board half of the job-server split: everything expensive (build, lower, compile,
        assemble) has already happened in this process, and what is left is a coarse payload — one
        job in, raw IQ out.

        ``overlay_id`` selects a pre-synthesized firmware personality (one overlay ⇔ one
        ``dump_cfg()`` snapshot, D25). The default ``""`` means **"whatever overlay is already
        loaded"** — the single-personality lab case, and what the reference agent in
        ``examples/jobserver/`` accepts; a multi-overlay deployment names them.

        ``soft_avgs`` is the ``rounds`` the board averages in software; it rides in the acquire block
        because the board runs the loop, not this process.
        """
        from strumento.core.transport import CompiledJob

        return CompiledJob(
            overlay_id=overlay_id,
            program=compiled_payload(self),
            acquire=acquire_config(self, soft_avgs=soft_avgs),
        )

    def xpts(self):
        """The result's swept axis — the **realized** lab-native axis (D37).

        The compiled per-step axis is read back from the program (``get_pulse_param`` for a
        swept pulse field, ``get_time_param`` for a swept delay) — the register-rounded truth
        with the loop engine's quantized step sizes, never a second client-side conversion —
        and mapped to lab units through the factory's inverse where one exists. What you plot
        and fit against is what the tProc actually steps. The requested axis is kept at
        ``MeasurementResult.meta["requested_xpts"]``. The point count comes from the sweep
        itself (legacy) or its declared axis (D36). We surface the first swept-field site's
        axis. Public because it is also the axis the experiment ``Runner`` arms a simulator
        with (D32) — the result's axis and the simulated axis are the same object by
        construction, never two derivations.
        """
        if self.sweeps:
            return self.sweeps[0].xpts  # legacy explicit form: the lab axis as given (pre-D37)
        sites = Compiler._sweep_sites(self.seq)
        if not sites:
            return None
        sw, kind, host = sites[0]
        return self._axis_of(sw, kind, host)

    #: swept-field site kind → the qick ``add_pulse`` parameter it lowers to
    _SITE_PARAM = {"gain_frac": "gain", "freq_mhz": "freq", "phase_deg": "phase", "time_us": "length"}

    def _axis_of(self, sw, kind: str, host):
        """The realized display axis for one sweep site: lab-native, ``points`` long (D36/D37)."""
        points = sw.points
        if sw.on is not None:
            points = next((a.points for a in self.axes if a.name == sw.on), None)
            if points is None:
                raise ValueError(f"sweep bound to undeclared axis {sw.on!r}")
        requested = self._lab_of(sw.axis_values(points), sw)
        realized = self._lab_of(self._compiled_axis(sw, kind, host), sw)
        self._requested_xpts = requested
        return realized

    def _compiled_axis(self, sw, kind: str, host):
        """The compiled per-step axis for one sweep site, read back from the program (D37).

        Pulse fields come from ``get_pulse_param`` and a swept delay from ``get_time_param``
        on the tag the compiler stamped at lowering time — both register-rounded after the
        loop macros are processed, which is exactly what a per-point client-side round-trip
        of the requested values cannot reproduce (the truncated register step sizes).
        """
        if self.binprog is None:
            # the rounded values exist only once the loops have been processed
            self.compile()
        assert self._compiler is not None  # set by _initialize, which __init__ runs via compile()
        if host is not None:
            ch = self._compiler._gen_ch(host)
            name = self._compiler._pulse_names.get(Compiler._pulse_key(host, ch))
            if name is None:
                raise ValueError(
                    "the swept pulse is not declared in this program — build the program "
                    "from a Seq containing it"
                )
            return np.atleast_1d(self._readback_param(name, self._SITE_PARAM[kind]))
        tag = self._compiler._delay_tags.get(Compiler._site_key(sw, "time_us"))
        if tag is None:
            raise ValueError("the swept delay was not lowered by this program's compiler")
        return np.atleast_1d(self.get_time_param(tag, "t", as_array=True))

    @staticmethod
    def _lab_of(values, sw):
        """Compiled-domain values → lab units: the factory's inverse map, else the pinned lab
        axis (a user-rolled ``mapped`` without an inverse — ≈ when quantization snaps), else the
        values unchanged (no unit mapping applies)."""
        if sw._lab_inverse is not None:
            return np.asarray([sw._lab_inverse(float(v)) for v in values])
        if sw._lab_axis is not None:
            return np.asarray(sw._lab_axis, dtype=float)
        return np.asarray(values, dtype=float)
