"""Tests for MockQickSocV2 (plan Task 7, D25) — a real QickConfig + stubbed acquire surface.

Marked ``qick``: needs the (pure-Python, off-board) qick library installed.
"""

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("qick")

from strumento.core.mock import MockQickSocV2  # noqa: E402

SNAP = Path(__file__).parents[2] / "devices" / "_fixtures" / "soccfg_v1_zcu216.json"

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
