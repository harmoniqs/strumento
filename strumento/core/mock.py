"""``MockQickSocV2`` — an offline (no-FPGA) soc for building and "running" v2 programs (D25).

The mock **is a real** :class:`qick.QickConfig`, reconstructed from a committed ``dump_cfg()``
snapshot. So an :class:`~strumento.core.program.StrumentoProgram` built against it genuinely
compiles — ``declare_gen`` / ``add_gauss`` / ``add_pulse`` and the v2 assembler all run and
validate — and only the *leaf* hardware calls that ``acquire`` makes on the soc are stubbed.
That is what makes the numeric-golden gate (Task 10) validate actual v2 assembly, not just
array shapes.

The stubbed surface (enumerated in ``docs/qick-v2-api-notes.md``): everything under
``config_all`` / ``load_pulses`` / ``config_gens`` / ``config_readouts`` / ``config_bufs`` /
the acquire loop is a no-op, except ``start_readout`` (captures the acquisition dims) and
``poll_data`` (returns correctly-shaped, int64, zero-valued IQ).

This module imports ``qick`` — it is one of the three qick-facing core modules (with
``compiler`` and ``program``).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from qick import QickConfig


class MockQickSocV2(QickConfig):
    """A real QickConfig with a stubbed hardware surface — build+compile real, acquire faked."""

    def __init__(self, soccfg_dict: dict) -> None:
        super().__init__(soccfg_dict)
        self._total_count: int = 0
        self._reads_per_shot: list[int] = [1]

    @classmethod
    def from_snapshot(cls, path: str | Path) -> "MockQickSocV2":
        """Build from a committed ``dump_cfg()`` JSON snapshot (no board)."""
        return cls(json.loads(Path(path).read_text()))

    # -- data path (the only methods that do something) --------------------------------

    def start_readout(self, total_count, counter_addr=None, ch_list=None, reads_per_shot=None):
        self._total_count = int(total_count)
        self._reads_per_shot = list(reads_per_shot) if reads_per_shot is not None else [1]

    def poll_data(self):
        """Return all acquisition data in one chunk: ``[(new_points, (d, stats))]``.

        ``d[ii]`` is int64 shaped ``(new_points * reads_per_shot[ii], 2)`` — the raw
        ``(I, Q)`` layout ``AveragerProgramV2.acquire`` reshapes into its ``d_buf``.
        """
        d = [np.zeros((self._total_count * nr, 2), dtype=np.int64) for nr in self._reads_per_shot]
        return [(self._total_count, (d, None))]

    # -- stubbed leaf hardware surface (no-ops) ----------------------------------------

    def load_bin_program(self, *args, **kwargs): pass
    def stop_tproc(self, *args, **kwargs): pass
    def start_src(self, *args, **kwargs): pass
    def start_tproc(self, *args, **kwargs): pass
    def reload_mem(self, *args, **kwargs): pass
    def load_pulse_data(self, *args, **kwargs): pass
    def set_nyquist(self, *args, **kwargs): pass
    def set_mixer_freq(self, *args, **kwargs): pass
    def config_mux_gen(self, *args, **kwargs): pass
    def configure_readout(self, *args, **kwargs): pass
    def config_mux_readout(self, *args, **kwargs): pass
    def config_avg(self, *args, **kwargs): pass
    def config_buf(self, *args, **kwargs): pass
    def set_tproc_counter(self, *args, **kwargs): pass

    def get_tproc_counter(self, *args, **kwargs):
        return 0
