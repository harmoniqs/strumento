"""Contracts for direct Fock-|1> cavity lifetime measurement."""

from __future__ import annotations

import numpy as np
import pytest

from strumento.core.pulses import Delay, Measure, Pulse
from strumento.core.result import MeasurementResult
from strumento.core.sweeps import Sweep
from strumento.packs.cqed.readout import DispersiveIQ


def test_fock_cavity_t1_prepares_waits_retrieves_maps_and_reads(dev):
    from strumento.packs.cqed.experiments.fock_cavity_t1 import FockCavityT1

    exp = FockCavityT1(
        dev,
        delays=Sweep(start=0.0, stop=1200.0, on="t"),
        points=81,
    )
    seq, sweeps = exp.sequence()

    assert sweeps == ()
    assert isinstance(seq.ops[3], Delay)
    assert seq.ops[3].duration_us == exp.delays
    labels = [op.label for op in seq.ops if isinstance(op, Pulse)]
    assert labels == ["pi_ge", "pi_ef", "f0-g1_pi0", "f0-g1_pi0", "pi_ef", "ro_probe"]
    assert sum(isinstance(op, Measure) for op in seq.ops) == 1


def test_fock_cavity_t1_recovers_direct_photon_lifetime(dev):
    from strumento.packs.cqed.experiments.fock_cavity_t1 import FockCavityT1

    delays = np.linspace(0.0, 1200.0, 81)
    lifetime_us = 210.0
    signal = 0.08 + 0.82 * np.exp(-delays / lifetime_us)
    result = MeasurementResult(
        xpts=delays,
        avgi=signal[None, :],
        avgq=np.zeros((1, delays.size)),
    )
    exp = FockCavityT1(
        dev,
        delays=Sweep(start=0.0, stop=1200.0, on="t"),
        points=delays.size,
        readout=DispersiveIQ(threshold=0.0),
    )

    fit = exp.analyze(result)

    assert fit.success is True
    assert fit.derived["T1"] == pytest.approx(lifetime_us, rel=1.0e-4)


def test_fock_cavity_t1_reports_three_level_retrieval_diagnostics(dev):
    from strumento.packs.cqed.experiments.fock_cavity_t1 import FockCavityT1

    delays = np.linspace(0.0, 1200.0, 81)
    p_e = 0.04 + 0.80 * np.exp(-delays / 210.0)
    g = np.array([0.0, 0.0])
    e = np.array([2.0, 0.0])
    f = np.array([0.0, 2.0])
    raw = (1.0 - p_e)[:, None] * g + p_e[:, None] * e
    result = MeasurementResult(
        xpts=delays,
        avgi=raw[:, 0][None, :],
        avgq=raw[:, 1][None, :],
    )
    readout = DispersiveIQ(
        threshold=0.0,
        g_center=tuple(g),
        e_center=tuple(e),
        f_center=tuple(f),
    )
    exp = FockCavityT1(
        dev,
        delays=Sweep(start=0.0, stop=1200.0, on="t"),
        points=delays.size,
        readout=readout,
    )

    fit = exp.analyze(result)

    assert fit.derived["initial_mapped_e_population"] == pytest.approx(p_e[0])
    assert fit.derived["final_mapped_e_population"] == pytest.approx(p_e[-1])
    assert fit.derived["max_f_population"] == pytest.approx(0.0, abs=1.0e-10)
