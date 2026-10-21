"""Shared gate for tests that need a real tProc-v2 soccfg.

The committed board snapshot (``devices/_fixtures/soccfg_v1_zcu216.json``) is tProc-**v1**
(``axis_tproc64x32_x8``); ``AveragerProgramV2`` requires a ``qick_processor`` (v2) config, so
real-v2-assembly tests skip rather than fake it when no v2 config is available.

Resolution order:

1. ``devices/_fixtures/soccfg_v2.json`` — a local ``dump_cfg()`` from a real v2 board
   (gitignored; drop one here and the whole v2 suite runs against *your* firmware), else
2. ``devices/_fixtures/soccfg_v2_testbench.json`` — the committed snapshot of the official qick
   testbench firmware config (single generator + single readout; see that directory's README).
"""

from pathlib import Path

import pytest

_FIXTURES = Path(__file__).parent.parent / "devices" / "_fixtures"
_LOCAL = _FIXTURES / "soccfg_v2.json"
_TESTBENCH = _FIXTURES / "soccfg_v2_testbench.json"

V2_SOCCFG = _LOCAL if _LOCAL.exists() else _TESTBENCH

requires_v2_soccfg = pytest.mark.skipif(
    not V2_SOCCFG.exists(),
    reason=(
        "no tProc-v2 soccfg — drop a v2 dump_cfg() at devices/_fixtures/soccfg_v2.json "
        "to enable real-v2 assembly + numeric-golden tests"
    ),
)
