"""``StrumentoProgram`` — an ``AveragerProgramV2`` driven by the strumento IR + compiler.

A program is constructed from a :class:`~strumento.core.device.Device` and a
:class:`~strumento.core.pulses.Seq` (plus optional sweeps). ``_initialize`` builds a
:class:`~strumento.core.compiler.Compiler` and runs its declaration pass (declare generators /
envelopes / readouts + one ``add_loop`` per sweep); ``_body`` runs the timeline pass. ``acquire``
wraps ``AveragerProgramV2.acquire``'s raw IQ into a :class:`~strumento.core.result.MeasurementResult`.

**Escape hatch:** a ``StrumentoProgram`` *is* an ``AveragerProgramV2``, so any raw qick call is
reachable (``.raw(fn)`` in a Seq, ``self.<method>`` from a hand-written ``_body``, …).

This module imports ``qick`` — one of the three qick-facing core modules. The raw→result
wrapping is factored out as :func:`_to_measurement_result` so it can be unit-tested without a
tProc-v2 board.
"""

from __future__ import annotations

import numpy as np
from qick import QickConfig
from qick.asm_v2 import AveragerProgramV2

from strumento.core.compiler import Compiler
from strumento.core.pulses import Seq
from strumento.core.result import MeasurementResult

#: default relax delay (µs) when the device calibration supplies none.
_DEFAULT_FINAL_DELAY_US = 1.0


def _to_measurement_result(raw, xpts=None) -> MeasurementResult:
    """Wrap ``AveragerProgramV2.acquire``'s output into a :class:`MeasurementResult`.

    ``raw`` has shape ``(n_ch, n_reads, 2)`` — or ``(n_ch, n_reads, n_expts, 2)`` with sweeps —
    where the last axis is ``(I, Q)``. Slice 1 surfaces the first readout channel's I/Q; the
    full raw array is retained on ``.raw`` for downstream reducers (Task 16).
    """
    arr = np.asarray(raw[0])  # first readout channel
    return MeasurementResult(xpts=xpts, avgi=arr[..., 0], avgq=arr[..., 1], raw=raw)


class StrumentoProgram(AveragerProgramV2):
    """A compiled strumento sequence, runnable via ``acquire`` on a (mock or real) soc."""

    def __init__(
        self,
        device,
        seq: Seq | None = None,
        *,
        sweeps=(),
        reps: int = 100,
        final_delay: float | None = None,
        soc=None,
        **kwargs,
    ) -> None:
        self.device = device
        self.seq = seq if seq is not None else Seq()
        self.sweeps = tuple(sweeps)
        self._compiler: Compiler | None = None

        soccfg = soc if soc is not None else QickConfig(device.soccfg_snapshot)
        if final_delay is None:
            ro = getattr(device.calib, "readout", None)
            final_delay = getattr(ro, "relax_delay", None) or _DEFAULT_FINAL_DELAY_US

        super().__init__(soccfg, reps=reps, final_delay=final_delay, **kwargs)

    def _initialize(self, cfg):
        # the compiler discovers sweeps in the sequence and registers their add_loop (a swept
        # pulse field carries its own Sweep), so we don't double-register self.sweeps here.
        self._compiler = Compiler(self, self.device.wiring, self.soccfg)
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
        return _to_measurement_result(raw, xpts=self._xpts())

    def _xpts(self):
        """The lab-native swept axis, from the explicit ``sweeps=`` or discovered in the seq.

        A factory-built swept pulse carries a :class:`Sweep` whose ``xpts`` is the lab-native
        axis (D23); we surface the first sweep's axis as the result's ``xpts``.
        """
        sweeps = list(self.sweeps) or Compiler._sweeps_in(self.seq)
        return sweeps[0].xpts if sweeps else None
