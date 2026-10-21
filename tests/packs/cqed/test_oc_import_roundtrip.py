"""OC import round-trip on a shipped device: a calibrated gate, re-imported as if solved, comes back.

The factory π (and π/2) is written in Piccolo units — ``(u_x, u_y)`` in GHz on a ns grid — and
pushed through the WHOLE import path: ``piccolo_controls_to_solution`` → ``from_solution(dac_rate=fs,
gain_calib=PiPulseReference)``. A wrong factor of 2, a flipped Q, a coarse or missing ``dac_rate``,
or an off-by-one resample all fail here (#210).

Uses ``devices/multimode_demo`` (committed, with a soccfg snapshot) with DRAG switched on, so both
quadratures are exercised. Pure: no qick.
"""

from __future__ import annotations

import numpy as np
import pytest

from strumento.interop.piccolo import PICCOLO_GHZ_TO_RABI_MHZ

from ._oc_lowering import dac_rate, lower, pi_ruler, played_nu_mhz, unitary

#: A made-up DRAG coefficient: anything non-zero puts a real Q quadrature on the factory ge gates.
DRAG_ALPHA = -0.08


@pytest.fixture
def ddev(dev):
    dev.calib.qubit.drag_alpha = DRAG_ALPHA
    return dev


def _factory(dev, key):
    return dev.qubit.ge_pi() if key == "pi_ge" else dev.qubit.ge_hpi()


def _as_piccolo_solution(dev, key, ruler, fs):
    """The factory pulse for ``key``, written as Piccolo (2, N) GHz controls on a ns grid."""
    p = _factory(dev, key)
    nu_mhz = played_nu_mhz(p, ruler)                                      # physical Ω/2π, MHz
    times_ns = np.arange(len(nu_mhz)) / fs * 1e3                          # one knot per sample
    u = np.vstack([nu_mhz.real, -nu_mhz.imag]) / PICCOLO_GHZ_TO_RABI_MHZ   # invert ν = 2000(u_x − i u_y)
    return p, times_ns, u


def _played(pulse):
    return pulse.gain * (np.asarray(pulse.envelope.idata) + 1j * np.asarray(pulse.envelope.qdata))


@pytest.mark.parametrize("key", ["pi_ge", "hpi_ge"])
def test_calibrated_pulse_round_trips_through_the_import_path(ddev, key):
    ruler, fs = pi_ruler(ddev), dac_rate(ddev)
    ref, times_ns, u = _as_piccolo_solution(ddev, key, ruler, fs)
    assert np.any(u[1] != 0), "DRAG must put a Q quadrature on the factory gate"
    seq, imported = lower(ddev, times_ns, u, ruler=ruler)

    # What the qubit sees is gain × envelope. (from_solution normalizes by the peak of |I + iQ|,
    # the factory by the I peak; with DRAG's Q those differ slightly — a representation choice, so
    # the invariant is the product, not the split.)
    assert len(imported.envelope.idata) == len(ref.envelope.idata)       # no sample lost or added
    np.testing.assert_allclose(_played(imported).real, _played(ref).real, atol=1e-12)  # amplitude
    np.testing.assert_allclose(_played(imported).imag, _played(ref).imag, atol=1e-12)  # ... Q sign
    peak = np.max(np.abs(np.asarray(imported.envelope.idata) + 1j * np.asarray(imported.envelope.qdata)))
    assert peak == pytest.approx(1.0)
    ref_peak = np.max(np.abs(np.asarray(ref.envelope.idata) + 1j * np.asarray(ref.envelope.qdata)))
    assert imported.gain == pytest.approx(ref.gain * ref_peak, rel=1e-9)  # the gain ratio is exact
    assert imported.freq_mhz == ddev.calib.qubit.f_ge
    assert seq.meta["dac_rate"] == fs


def test_pi_ruler_anchor_is_the_physical_pi_rate(ddev):
    """Ω_peak · ∫Î dt = π at the π gain — the anchor that makes MHz → gain fraction physical."""
    ruler, fs = pi_ruler(ddev), dac_rate(ddev)
    pi = ddev.qubit.ge_pi()
    rotation = 2 * np.pi * ruler.pi_rabi_mhz * float(np.sum(pi.envelope.idata)) / fs
    assert rotation == pytest.approx(np.pi)


def test_the_played_duration_equals_the_solution_span(ddev):
    """N samples at fs play for N/fs. A solver grid of N knots spaced 1/fs imports to N samples
    (not N+1 from an inclusive-endpoint resample) — an extra sample shifts every later pulse."""
    ruler, fs = pi_ruler(ddev), dac_rate(ddev)
    ref, times_ns, u = _as_piccolo_solution(ddev, "pi_ge", ruler, fs)
    seq, imported = lower(ddev, times_ns, u, ruler=ruler)
    n = len(imported.envelope.idata)
    assert n == seq.meta["n_samples"] == len(ref.envelope.idata)
    span_us = (times_ns[-1] - times_ns[0]) * 1e-3
    assert seq.meta["span_us"] == pytest.approx(span_us)
    assert n / fs == pytest.approx(span_us + 1 / fs)    # each knot is one sample long: no stretch


def test_a_coarse_or_missing_dac_rate_changes_the_played_duration(ddev):
    """The Arb plays its N samples at the DAC's own fs — any other grid time-compresses it, and
    that is visible in the import's sample count and provenance, not silently corrected."""
    ruler, fs = pi_ruler(ddev), dac_rate(ddev)
    ref, times_ns, u = _as_piccolo_solution(ddev, "pi_ge", ruler, fs)
    good, p_good = lower(ddev, times_ns, u, ruler=ruler)
    coarse, p_coarse = lower(ddev, times_ns, u, ruler=ruler, fs=fs / 4)
    played_good = len(p_good.envelope.idata) / fs
    played_coarse = len(p_coarse.envelope.idata) / fs
    assert played_good == pytest.approx(len(ref.envelope.idata) / fs)
    assert played_coarse < 0.3 * played_good           # played ~4× shorter on the board
    assert coarse.meta["dac_rate"] == fs / 4

    # Missing: a solver's own ~1 ns grid, imported with dac_rate=None, keeps its knots — each
    # then plays for one DAC sample, ~10× shorter than the solve assumed. The provenance says so.
    t_ns = times_ns[::10]
    missing, p_missing = lower(ddev, t_ns, u[:, ::10], ruler=ruler, fs=None)
    assert missing.meta["dac_rate"] is None
    assert len(p_missing.envelope.idata) == len(t_ns)
    assert len(p_missing.envelope.idata) / fs < 0.2 * played_good


# ---------------------------------------------------------------------------------------------
# Gate equivalence: an OC-lowered pulse must DO the same gate as the factory gate it stands in
# for. A known-good waveform (the factory gate itself) round-tripped through the OC path must give
# the SAME unitary through a 3-level transmon Hamiltonian. No solver: this isolates the import.
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("key", ["pi_ge", "hpi_ge"])
def test_oc_lowered_pulse_does_the_same_gate_as_the_factory(ddev, key):
    ruler, fs = pi_ruler(ddev), dac_rate(ddev)
    ref, times_ns, u = _as_piccolo_solution(ddev, key, ruler, fs)
    _, oc = lower(ddev, times_ns, u, ruler=ruler)

    delta = ddev.calib.qubit.f_ef - ddev.calib.qubit.f_ge
    u_ref = unitary(played_nu_mhz(ref, ruler), fs, delta)
    u_oc = unitary(played_nu_mhz(oc, ruler), fs, delta)
    np.testing.assert_allclose(u_oc, u_ref, atol=1e-9)          # identical gate, element by element

    p_e = abs((u_oc @ np.array([1, 0, 0], complex))[1]) ** 2   # ... and a sane one
    assert p_e == pytest.approx(1.0 if key == "pi_ge" else 0.5, abs=0.05)


def test_oc_override_stands_in_for_ge_pi_through_transmon_ops(ddev):
    """The imported pulse, dropped in via overrides=, is what the ops factory then returns."""
    from strumento.packs.cqed.library.transmon import TransmonOps

    ruler, fs = pi_ruler(ddev), dac_rate(ddev)
    ref, times_ns, u = _as_piccolo_solution(ddev, "pi_ge", ruler, fs)
    _, oc = lower(ddev, times_ns, u, ruler=ruler)

    ops = TransmonOps(ddev, ddev.qubit.element, overrides={"pi_ge": oc})
    played = ops.ge_pi()
    assert played.envelope is oc.envelope                       # the override IS what ge_pi plays
    np.testing.assert_allclose(_played(played), _played(ref), atol=1e-12)
    assert played.start_us is None                              # ... sequenced like the factory gate
    assert ops.ge_pi(phase=90.0).phase_deg == 90.0              # phase still applies on top
    assert ops.ge_hpi().envelope is not oc.envelope             # scoped: hpi_ge untouched
