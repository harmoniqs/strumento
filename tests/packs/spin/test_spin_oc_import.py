"""Spin OC import — from_solution round-trip through the spin pack's load_oc_pulse.

Tests that:
1. SpinGainReference converts physical amplitudes to v2 fractions correctly.
2. load_oc_pulse builds Arb-envelope pulses with correct LineRef targets and gains.
3. The round-trip works for both EDSR drives (RF, at Larmor) and exchange drives (baseband).

Pure: no qick (uses only the IR and calibration layers).
"""

from __future__ import annotations

import numpy as np
import pytest

from strumento.core.pulses import Arb, Segment
from strumento.core.wiring import LineRef
from strumento.packs.spin.calibration import SpinGainReference
from strumento.packs.spin.library.composite import load_oc_pulse

# -- SpinGainReference --------------------------------------------------------


class TestSpinGainReference:
    """The linear ruler: physical Rabi amplitude → v2 gain fraction."""

    def test_conversion(self):
        ref = SpinGainReference(pi_gain_frac=0.4, pi_rabi_mhz=20.0)
        # At pi Rabi rate, gain should equal pi_gain_frac
        assert ref(20.0) == pytest.approx(0.4)
        # Half the Rabi rate → half the gain
        assert ref(10.0) == pytest.approx(0.2)
        # Zero amplitude → zero gain
        assert ref(0.0) == pytest.approx(0.0)

    def test_rejects_nonpositive_rabi(self):
        with pytest.raises(ValueError, match="positive"):
            SpinGainReference(pi_gain_frac=0.4, pi_rabi_mhz=0.0)
        with pytest.raises(ValueError, match="positive"):
            SpinGainReference(pi_gain_frac=0.4, pi_rabi_mhz=-1.0)


# -- load_oc_pulse round-trip ------------------------------------------------


def _toy_exchange_solution():
    """A synthetic 2-drive solution: EDSR drive (real) + exchange (real, baseband)."""
    t = np.linspace(0.0, 0.5, 51)  # 500 ns
    return {
        "times": t,
        "controls": {
            "edsr_q0": 15.0 * np.sin(np.pi * t / 0.5) ** 2,       # MHz Rabi, real
            "exchange_b0": 8.0 * np.exp(-((t - 0.25) / 0.1) ** 2),  # MHz, Gaussian pulse
        },
    }


def _spin_drive_map():
    return {
        "edsr_q0": (LineRef("q0", "drive"), 5000.0),     # EDSR at Larmor frequency
        "exchange_b0": (("b0", "fast"), 0.0),             # baseband exchange (tuple form)
    }


class TestLoadOCPulse:
    """load_oc_pulse builds a Seq of Arb-envelope pulses with correct targets and gains."""

    def test_builds_parallel_arb_segment(self):
        ref = SpinGainReference(pi_gain_frac=0.4, pi_rabi_mhz=20.0)
        seq = load_oc_pulse(_toy_exchange_solution(), drive_map=_spin_drive_map(), gain_calib=ref)

        # One Segment containing two parallel pulses
        assert len(seq.ops) == 1 and isinstance(seq.ops[0], Segment)
        seg = seq.ops[0]
        assert len(seg.pulses) == 2

        # Both lines present
        lines = {p.line for p in seg.pulses}
        assert lines == {LineRef("q0", "drive"), LineRef("b0", "fast")}

    def test_edsr_pulse_properties(self):
        ref = SpinGainReference(pi_gain_frac=0.4, pi_rabi_mhz=20.0)
        seq = load_oc_pulse(_toy_exchange_solution(), drive_map=_spin_drive_map(), gain_calib=ref)
        seg = seq.ops[0]
        by_line = {p.line.line: p for p in seg.pulses}

        edsr = by_line["q0"]
        assert isinstance(edsr.envelope, Arb)
        assert edsr.envelope.qdata is None  # real-valued EDSR drive
        assert edsr.freq_mhz == pytest.approx(5000.0)
        # Peak amplitude is 15 MHz → gain = ref(15) = 15 * 0.4/20 = 0.3
        assert edsr.gain == pytest.approx(ref(15.0))
        # Normalized samples: peak should be ±1
        assert np.max(np.abs(edsr.envelope.idata)) == pytest.approx(1.0)

    def test_exchange_pulse_properties(self):
        ref = SpinGainReference(pi_gain_frac=0.4, pi_rabi_mhz=20.0)
        seq = load_oc_pulse(_toy_exchange_solution(), drive_map=_spin_drive_map(), gain_calib=ref)
        seg = seq.ops[0]
        by_line = {p.line.line: p for p in seg.pulses}

        exch = by_line["b0"]
        assert isinstance(exch.envelope, Arb)
        assert exch.freq_mhz == pytest.approx(0.0)  # baseband
        assert exch.line == LineRef("b0", "fast")
        # Peak amplitude is 8 MHz → gain = ref(8) = 8 * 0.4/20 = 0.16
        assert exch.gain == pytest.approx(ref(8.0))

    def test_provenance_metadata(self):
        ref = SpinGainReference(pi_gain_frac=0.4, pi_rabi_mhz=20.0)
        seq = load_oc_pulse(_toy_exchange_solution(), drive_map=_spin_drive_map(), gain_calib=ref)
        assert seq.meta["source"] == "from_solution"
        assert sorted(seq.meta["drives"]) == ["edsr_q0", "exchange_b0"]
        assert seq.meta["n_samples"] == 51
        assert seq.meta["span_us"] == pytest.approx(0.5)

    def test_resamples_to_dac_rate(self):
        ref = SpinGainReference(pi_gain_frac=0.4, pi_rabi_mhz=20.0)
        seq = load_oc_pulse(
            _toy_exchange_solution(), drive_map=_spin_drive_map(),
            gain_calib=ref, dac_rate=400.0,
        )
        # 0.5 µs at 400 samples/µs → 201 samples
        assert seq.meta["n_samples"] == 201
        seg = seq.ops[0]
        assert all(len(p.envelope.idata) == 201 for p in seg.pulses)

    def test_unknown_drive_raises(self):
        ref = SpinGainReference(pi_gain_frac=0.4, pi_rabi_mhz=20.0)
        # Only map one of the two drives — the other should raise
        partial_map = {"edsr_q0": (("q0", "drive"), 5000.0)}
        with pytest.raises(KeyError, match="drive_map"):
            load_oc_pulse(_toy_exchange_solution(), drive_map=partial_map, gain_calib=ref)
