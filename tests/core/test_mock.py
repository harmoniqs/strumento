"""Tests for MockQickSocV2 (D25) — a real QickConfig + stubbed acquire surface.

Marked ``qick``: needs the (pure-Python, off-board) qick library installed.
"""

import numpy as np
import pytest

pytest.importorskip("qick")

from strumento.core.mock import MockQickSocV2  # noqa: E402
from tests._paths import DEVICES  # noqa: E402

SNAP = DEVICES / "_fixtures" / "soccfg_v1_zcu216.json"

pytestmark = pytest.mark.qick


def test_mock_is_a_real_qickconfig():
    soc = MockQickSocV2.from_snapshot(SNAP)
    # real config data + working unit conversions (so program build/compile validate for real)
    assert soc["gens"][2]["maxv"] == 32766
    assert soc.freq2reg(3562.6, gen_ch=2) != 0


def test_poll_data_shape_and_dtype_single_readout():
    soc = MockQickSocV2.from_snapshot(SNAP)
    soc.start_readout(total_count=51, counter_addr=1, ch_list=[0], reads_per_shot=[1])
    chunks = soc.poll_data()
    assert len(chunks) == 1
    new_points, (d, stats) = chunks[0]
    assert new_points == 51
    assert len(d) == 1
    assert d[0].shape == (51, 2)
    assert d[0].dtype == np.int64
    assert stats is None


def test_poll_data_multi_readout_and_reads():
    soc = MockQickSocV2.from_snapshot(SNAP)
    soc.start_readout(total_count=10, counter_addr=1, ch_list=[0, 1], reads_per_shot=[2, 1])
    new_points, (d, _) = soc.poll_data()[0]
    assert new_points == 10
    assert d[0].shape == (20, 2)  # 10 shots * 2 reads
    assert d[1].shape == (10, 2)  # 10 shots * 1 read


def test_leaf_hardware_methods_are_noops():
    soc = MockQickSocV2.from_snapshot(SNAP)
    for m in (
        "load_bin_program", "stop_tproc", "start_src", "load_pulse_data",
        "set_nyquist", "set_mixer_freq", "config_mux_gen", "configure_readout",
        "config_mux_readout", "config_avg", "config_buf", "reload_mem",
    ):
        assert getattr(soc, m)() is None  # callable, accepts no args, returns None


# -- decimated-mode leaf surface (#104: AveragerProgramV2.acquire_decimated) -----------------


def test_get_tproc_counter_reports_done_on_the_first_ask():
    """``acquire_decimated``'s ``finish_round`` polls ``while count < total_count: count =
    soc.get_tproc_counter(...)`` after ``start_tproc()`` — with no real tProc clock behind the
    mock, a value that never satisfies some caller's ``total_count`` would spin forever. Assert
    the actual contract (large enough for any realistic total_count), not just "no hang"."""
    soc = MockQickSocV2.from_snapshot(SNAP)
    count = soc.get_tproc_counter(addr=1)
    assert count > 1_000_000, "must be large enough to satisfy any reps*expts total_count at once"
    # Idempotent: asking again (as the poll loop does) reports the same "already done" answer.
    assert soc.get_tproc_counter(addr=1) == count


def test_get_decimated_returns_zero_valued_iq_of_the_requested_length():
    soc = MockQickSocV2.from_snapshot(SNAP)
    d = soc.get_decimated(ch=0, address=0, length=307)
    assert d.shape == (307, 2)
    assert d.dtype == np.int64
    assert np.all(d == 0)


def test_get_accumulated_returns_zero_valued_iq_of_the_requested_length():
    soc = MockQickSocV2.from_snapshot(SNAP)
    a = soc.get_accumulated(ch=0, address=0, length=12)
    assert a.shape == (12, 2)
    assert a.dtype == np.int64
    assert np.all(a == 0)
