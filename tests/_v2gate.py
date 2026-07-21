"""Shared gate for tests that need a real tProc-v2 soccfg.

The rescued Schuster snapshot is tProc-**v1** (``axis_tproc64x32_x8``); ``AveragerProgramV2``
requires a ``qick_processor`` (v2) config. Until we have one — Sho Uemura's ``dump_cfg()`` from a
v2 board — real-v2-assembly tests skip rather than fake it.

To enable them: drop a v2 ``dump_cfg()`` JSON at ``devices/_fixtures/soccfg_v2.json``.
"""

from pathlib import Path

import pytest

V2_SOCCFG = Path(__file__).parent.parent / "devices" / "_fixtures" / "soccfg_v2.json"

requires_v2_soccfg = pytest.mark.skipif(
    not V2_SOCCFG.exists(),
    reason=(
        "no tProc-v2 soccfg — drop Sho's dump_cfg() at devices/_fixtures/soccfg_v2.json "
        "to enable real-v2 assembly + numeric-golden tests"
    ),
)
