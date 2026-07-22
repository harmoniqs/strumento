"""Tests for the AbstractSoc seam (spec §19.3) — the board-object protocol.

``AbstractSoc`` names the runtime-checkable surface that ``AveragerProgramV2.acquire`` drives: a
mock, a real ``QickSoc``, or a simulator all satisfy it. It's the seam that makes "which board"
(and, above it, "which transport") a swap rather than a rewrite — the property the resolved D14
(no-Pyro, job-server-native) rests on.
"""

from pathlib import Path

import pytest

from strumento.core.soc import AbstractSoc


class _DuckSoc:
    def start_readout(self, *a, **k): ...
    def poll_data(self): return []
    def reload_mem(self, *a, **k): ...
    def start_src(self, *a, **k): ...


def test_full_surface_satisfies_protocol():
    assert isinstance(_DuckSoc(), AbstractSoc)


def test_incomplete_soc_does_not_satisfy():
    class Partial:
        def start_readout(self, *a, **k): ...
        # missing poll_data / reload_mem / start_src

    assert not isinstance(Partial(), AbstractSoc)


@pytest.mark.qick
def test_mock_satisfies_abstract_soc():
    from strumento.core.mock import MockQickSocV2

    snap = Path(__file__).parents[2] / "devices" / "_fixtures" / "soccfg_v1_zcu216.json"
    assert isinstance(MockQickSocV2.from_snapshot(snap), AbstractSoc)
