"""Contracts for direct single-photon cavity Ramsey coherence measurement."""

from __future__ import annotations

import numpy as np
import pytest

from strumento.core.pulses import Delay, Measure, Pulse
from strumento.core.result import MeasurementResult
from strumento.core.sweeps import Sweep
from strumento.packs.cqed.readout import DispersiveIQ


def test_fock_cavity_t2_ramsey_prepares_waits_retrieves_and_closes(dev):
    from strumento.packs.cqed.experiments.fock_cavity_t2_ramsey import FockCavityT2Ramsey

    exp = FockCavityT2Ramsey(
        dev,
        delays=Sweep(start=0.0, stop=300.0, on="t"),
        points=101,
        reference_detuning_mhz=0.02,
    )
    seq, sweeps = exp.sequence()

    assert sweeps == ()
    assert isinstance(seq.ops[3], Delay)
    assert seq.ops[3].duration_us is exp.delays
    assert isinstance(seq.ops[4], Pulse)
    assert isinstance(seq.ops[4].phase_deg, Sweep)
    assert seq.ops[4].phase_deg.on == "t"
    assert seq.ops[4].phase_deg.start == pytest.approx(0.0)
    assert seq.ops[4].phase_deg.stop == pytest.approx(360.0 * 0.02 * 300.0)
    labels = [op.label for op in seq.ops if isinstance(op, Pulse)]
    assert labels == [
        "hpi_ge",
        "pi_ef",
        "f0-g1_pi0",
        "f0-g1_pi0",
        "pi_ef",
        "hpi_ge",
        "ro_probe",
    ]
    assert sum(isinstance(op, Measure) for op in seq.ops) == 1


def test_fock_cavity_t2_ramsey_recovers_the_coherence_envelope_and_fringe(dev):
    from strumento.packs.cqed.experiments.fock_cavity_t2_ramsey import FockCavityT2Ramsey

    delays = np.linspace(0.0, 300.0, 101)
    t2_us = 75.0
    detuning_mhz = 0.026
    signal = 0.45 + 0.42 * np.exp(-delays / t2_us) * np.cos(2.0 * np.pi * detuning_mhz * delays)
    result = MeasurementResult(
        xpts=delays,
        avgi=signal[None, :],
        avgq=np.zeros((1, delays.size)),
    )
    exp = FockCavityT2Ramsey(
        dev,
        delays=Sweep(start=0.0, stop=300.0, on="t"),
        points=delays.size,
        readout=DispersiveIQ(threshold=0.0),
    )

    fit = exp.analyze(result)

    assert fit.success is True
    assert fit.derived["T2"] == pytest.approx(t2_us, rel=1.0e-4)
    assert fit.derived["detuning"] == pytest.approx(detuning_mhz, rel=1.0e-4)


def test_fock_cavity_t2_ramsey_reports_three_level_mapping_diagnostics(dev):
    from strumento.packs.cqed.experiments.fock_cavity_t2_ramsey import FockCavityT2Ramsey

    delays = np.linspace(0.0, 300.0, 101)
    p_e = 0.50 + 0.40 * np.exp(-delays / 75.0) * np.cos(2.0 * np.pi * 0.02 * delays)
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
    exp = FockCavityT2Ramsey(
        dev,
        delays=Sweep(start=0.0, stop=300.0, on="t"),
        points=delays.size,
        readout=readout,
    )

    fit = exp.analyze(result)

    assert fit.derived["ramsey_population_contrast"] == pytest.approx(np.ptp(p_e))
    assert fit.derived["max_f_population"] == pytest.approx(0.0, abs=1.0e-10)
