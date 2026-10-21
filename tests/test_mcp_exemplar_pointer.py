"""LabSession's error messages point at the cqed exemplar, ``multimode_demo`` (issue #131).

``tests/test_mcp.py`` needs the ``mcp`` extra; :class:`LabSession` is importable without the SDK
(and without qick), so the pointers its errors carry are pinned here, in the pure lane.
``loopback_demo`` is the single-channel v2 testbench, not the exemplar.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from strumento.mcp import LabSession, NoDeviceError, SessionError


def _assert_points_at_multimode(msg: str) -> None:
    assert "multimode_demo" in msg and "loopback_demo" not in msg, msg


def test_no_device_loaded_points_at_multimode_demo():
    with pytest.raises(NoDeviceError) as err:
        LabSession()._require_device()
    assert "no device loaded" in str(err.value)
    _assert_points_at_multimode(str(err.value))


@pytest.mark.parametrize("kind", ["simulator", "mock"])
def test_a_snapshotless_instance_points_at_multimode_demo(kind):
    with pytest.raises(SessionError) as err:
        LabSession._open_soc(kind, SimpleNamespace(soccfg_snapshot=None), {})
    assert "no soccfg snapshot" in str(err.value)
    _assert_points_at_multimode(str(err.value))
