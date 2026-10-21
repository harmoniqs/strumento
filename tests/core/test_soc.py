"""Tests for the AbstractSoc seam — the board-object protocol.

``AbstractSoc`` names the runtime-checkable surface that ``AveragerProgramV2.acquire`` drives: a
mock, a real ``QickSoc``, or a simulator all satisfy it. It's the seam that makes "which board"
(and, above it, "which transport") a swap rather than a rewrite — the property the resolved D14
(no-Pyro, job-server-native) rests on.
"""

import pytest

from strumento.core.soc import AbstractSoc
from tests._paths import DEVICES


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

    snap = DEVICES / "_fixtures" / "soccfg_v1_zcu216.json"
    assert isinstance(MockQickSocV2.from_snapshot(snap), AbstractSoc)
