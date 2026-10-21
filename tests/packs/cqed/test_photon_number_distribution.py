"""Tests for PhotonNumberDistribution -- had zero coverage before this audit fix
(2026-09-14): a missing/zero chi calibration silently substituted an arbitrary 1.0 MHz
and proceeded to drive number-selective pulses at the wrong frequency with no error
signal, contradicting the class's own docstring ("Requires calib.modes[mode_name].chi").

Pure: no qick needed.
"""

from __future__ import annotations

import pytest

from strumento.packs.cqed.experiments.photon_number_distribution import (
    PhotonNumberDistribution,
)


def test_sequence_builds_with_calibrated_chi(dev):
    """cavity has chi=1.5 calibrated -- should build cleanly."""
    exp = PhotonNumberDistribution(dev, n_max=3)
    seq, _ = exp.sequence()
    # (selective pi + readout probe + measure) per photon number, n=0..3 -- #103: each block
    # now drives the readout line before its Measure, alongside the selective pi pulse.
    assert len(seq.ops) == 3 * 4


def test_selective_pulse_frequencies_use_the_calibrated_chi(dev):
    exp = PhotonNumberDistribution(dev, n_max=2)
    seq, _ = exp.sequence()
    f_ge = dev.calib.qubit.f_ge
    chi = dev.calib.modes["cavity"].chi
    # #103 added a readout probe per block, so `seq.ops` now interleaves two Pulses per
    # photon number (the selective pi, then the readout probe) -- select the selective pi
    # pulses specifically by their per-n label rather than grabbing every Pulse-shaped op.
    selective_pulses = [
        op for op in seq.ops
        if hasattr(op, "freq_mhz") and str(getattr(op, "label", "")).startswith("selective_pi_")
    ]
    assert len(selective_pulses) == 3  # n=0,1,2
    for n, pulse in enumerate(selective_pulses):
        assert pulse.freq_mhz == pytest.approx(f_ge - n * chi, rel=1e-9)


def test_missing_chi_calibration_raises(dev):
    dev.calib.modes["cavity"].chi = None
    exp = PhotonNumberDistribution(dev)
    with pytest.raises(KeyError, match="chi"):
        exp.sequence()


def test_unknown_mode_name_raises(dev):
    exp = PhotonNumberDistribution(dev, mode_name="does_not_exist")
    with pytest.raises(KeyError, match="chi"):
        exp.sequence()
