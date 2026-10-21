"""Regression: WignerTomography and StatePrepFidelity passed a physical
displacement amplitude beta straight into mode.displace(gain=...), which treats any Sweep
as a lab-native int-code sweep (D23) -- resolving to a fraction of ~beta/32766 (~1e-5 for
the 0-0.25 default), i.e. no real displacement on hardware. SimulatorSoc's response models
read the swept axis (beta) directly and never exercised the gain-unit boundary, so this was
invisible to every existing test. These tests check the actual resolved gain, not just that
the experiment runs.

Pure: no qick needed.
"""

from __future__ import annotations

import numpy as np
import pytest

from strumento.core.result import MeasurementResult
from strumento.core.sweeps import Sweep
from strumento.packs.cqed.experiments.state_prep_fidelity import StatePrepFidelity
from strumento.packs.cqed.experiments.wigner_tomography import WignerTomography

MAXV = 32766  # from the committed soccfg snapshot


def test_wigner_tomography_gain_uses_displacement_cal_not_raw_beta(dev):
    """cavity's displacement_cal is 3000 (int-code per unit |alpha|). A default beta
    sweep of 0->0.25 must resolve to a gain fraction of 3000*0.25/32766, not 0.25/32766
    (the bug: treating beta itself as the int code)."""
    exp = WignerTomography(dev, betas=Sweep(start=0.0, stop=0.25, on="beta_re"), points=5)
    seq, _ = exp.sequence()
    displace_pulse = seq.ops[0]
    expected_stop_frac = (3000 * 0.25) / MAXV
    buggy_stop_frac = 0.25 / MAXV
    assert displace_pulse.gain.stop == pytest.approx(expected_stop_frac, rel=1e-6)
    assert displace_pulse.gain.stop != pytest.approx(buggy_stop_frac, rel=1e-3)
    # sanity: the fixed value is ~3000x larger than the buggy one would have been
    assert displace_pulse.gain.stop / buggy_stop_frac == pytest.approx(3000.0, rel=1e-6)


def test_wigner_tomography_negative_beta_preserves_sign(dev):
    """A sweep spanning negative beta (the general Wigner-grid case) maps through the same
    slope without folding sign onto magnitude -- gain.start stays negative."""
    exp = WignerTomography(dev, betas=Sweep(start=-2.0, stop=2.0, on="beta_re"), points=5)
    seq, _ = exp.sequence()
    displace_pulse = seq.ops[0]
    assert displace_pulse.gain.start == pytest.approx((3000 * -2.0) / MAXV, rel=1e-6)
    assert displace_pulse.gain.stop == pytest.approx((3000 * 2.0) / MAXV, rel=1e-6)


def test_wigner_tomography_no_displacement_cal_raises(dev):
    """Without a calibrated displacement_cal, the experiment should raise -- not silently
    build a near-zero displacement (the original bug's failure mode)."""
    dev.calib.modes["cavity"].displacement_cal = None
    exp = WignerTomography(dev, points=5)
    with pytest.raises(KeyError, match="displacement_cal"):
        exp.sequence()


def test_state_prep_fidelity_gain_uses_displacement_cal_not_raw_beta(dev):
    exp = StatePrepFidelity(dev, betas=Sweep(start=0.0, stop=0.25, on="beta_re"), points=5)
    seq, _ = exp.sequence()
    displace_pulse = seq.ops[0]
    expected_stop_frac = (3000 * 0.25) / MAXV
    assert displace_pulse.gain.stop == pytest.approx(expected_stop_frac, rel=1e-6)


def test_state_prep_fidelity_prepared_state_bypasses_displacement_entirely(dev):
    """When prepared_state is given, no displacement_cal lookup happens at all -- an OC-
    imported pulse shouldn't need a cavity displacement calibration."""
    from strumento.core.pulses import Seq

    dev.calib.modes["cavity"].displacement_cal = None  # would raise if looked up
    oc_prep = Seq().play(dev.qubit.ge_pi())
    exp = StatePrepFidelity(dev, prepared_state=oc_prep, points=5)
    seq, _ = exp.sequence()  # should not raise
    assert len(seq.ops) > 0


# -- state prep + Kerr wait + complex-beta ray -----------------------------------------


def test_wigner_prep_kerr_wait_and_ray_phase(dev):
    """init_alpha prepares a coherent state, kerr_wait_us inserts a free-evolution Delay, and
    beta_phase_deg rotates the tomography displacement off the real axis."""
    exp = WignerTomography(
        dev, betas=Sweep(start=-2.0, stop=2.0, on="beta_re"), points=5,
        init_alpha=1.0, kerr_wait_us=5.0, beta_phase_deg=90.0,
    )
    seq, _ = exp.sequence()
    kinds = [type(op).__name__ for op in seq.ops]
    # prep displace, Kerr Delay, tomography displace, then parity (hpi, revival Delay, hpi, ro, meas)
    assert kinds[0] == "Pulse"                                  # state-prep displacement
    assert not isinstance(seq.ops[0].gain, Sweep)               # scalar gain (fixed |alpha|)
    assert kinds[1] == "Delay"                                  # Kerr free-evolution wait
    assert seq.ops[1].duration_us == pytest.approx(5.0)
    assert kinds[2] == "Pulse"                                  # tomography displacement
    assert seq.ops[2].phase_deg == pytest.approx(90.0)          # complex-beta ray (was always 0)
    assert isinstance(seq.ops[2].gain, Sweep)                   # swept magnitude via displacement_cal
    assert seq.ops[2].gain.start == pytest.approx((3000 * -2.0) / MAXV, rel=1e-6)


def test_wigner_prep_fock_builds_the_ladder_not_a_coherent_displace(dev):
    """prep_fock=n preps a Fock |n> via the analytic ge_pi·ef_pi·f0g1 ladder (the non-classical
    test) instead of a coherent displace_alpha, ahead of the swept tomography displacement."""
    exp = WignerTomography(dev, betas=Sweep(start=-1.0, stop=1.0, on="beta_re"), points=5,
                           prep_fock=1)
    seq, _ = exp.sequence()
    labels = [getattr(op, "label", type(op).__name__) for op in seq.ops]
    assert labels[:3] == ["pi_ge", "pi_ef", "f0-g1_pi0"]      # the Fock|1> ladder, first
    assert isinstance(seq.ops[3].gain, Sweep)                 # then the swept tomography displacement
    assert "Measure" in [type(op).__name__ for op in seq.ops]


def test_wigner_prep_fock_none_keeps_the_coherent_path(dev):
    """prep_fock=None (default) leaves the coherent |init_alpha> displacement prep — no Fock ladder."""
    exp = WignerTomography(dev, betas=Sweep(start=0.0, stop=0.25, on="beta_re"), points=5,
                           init_alpha=1.0)
    seq, _ = exp.sequence()
    labels = [getattr(op, "label", type(op).__name__) for op in seq.ops]
    assert "pi_ge" not in labels and "f0-g1_pi0" not in labels  # no Fock ladder on the coherent path


def test_wigner_default_is_unchanged_vacuum_ray(dev):
    """With init_alpha=0 and kerr_wait_us=0 (defaults), the sequence is the prior vacuum ray:
    ops[0] is the tomography displacement at zero phase (no prep pulse, no Delay). Guards the
    backward-compatible path the gain-unit tests above rely on."""
    exp = WignerTomography(dev, betas=Sweep(start=0.0, stop=0.25, on="beta_re"), points=5)
    seq, _ = exp.sequence()
    assert type(seq.ops[0]).__name__ == "Pulse"
    assert seq.ops[0].phase_deg == pytest.approx(0.0)
    assert isinstance(seq.ops[0].gain, Sweep)


def test_wigner_2d_star_samples_complex_plane(dev):
    """The polar-star driver reaches off-real-axis points -- the fix for the old 2-D driver that
    reran Re(beta) and dropped Im(beta). Pure: sample_points() takes no soc."""
    from strumento.packs.cqed.experiments.wigner_tomography import WignerTomography2D

    star = WignerTomography2D(dev, soc=None, beta_max=3.0, n_radial=5, n_angle=4)
    pts = star.sample_points()
    assert pts.size == 4 * 5                       # n_angle * n_radial
    assert np.any(np.abs(pts.imag) > 1e-6)         # reaches Im(beta) != 0 (the bug fix)
    assert np.any(np.abs(pts.imag) < 1e-9)         # still covers the real axis (theta=0 ray)


# -- hardware compile/acquire split (regression: 'Proxy' object is not subscriptable on --arm) ----


class _CompileCfg:
    """Stand-in for the compile-time soccfg (a real QickConfig): subscriptable; never acquired."""

    def __getitem__(self, key):
        return {}

    def acquire(self, *a, **kw):
        raise AssertionError("acquired from the compile-time soccfg — should never happen")


class _AcquireProxy:
    """Stand-in for the qick.pyro acquire proxy: NOT subscriptable (the actual --arm failure),
    acquires fine."""

    def __getitem__(self, key):
        raise AssertionError("compiled against the run-time soc — should never happen")

    def acquire(self, *a, **kw):
        return "ACQUIRED"


class _StubProgram:
    """Swaps ``StrumentoProgram``: reads its constructor ``soc`` once (a compile-time soccfg read)
    and its acquire-time ``soc`` once, so a subscript on the wrong object fails loudly — enough to
    prove which object played which role, no qick installed."""

    def __init__(self, device, seq=None, *, sweeps=(), axes=(), reps=100, final_delay=None,
                 soc=None, **kw):
        soc["probe"]  # compile-time config read

    def xpts(self):
        return ()

    def acquire(self, soc, **kw):
        marker = soc.acquire()
        n = 3
        return MeasurementResult(xpts=list(range(n)), avgi=[[0.0] * n], avgq=[[0.0] * n],
                                 meta={"acquired_via": marker})


@pytest.fixture
def _stub_program(monkeypatch):
    import sys
    import types

    module = types.ModuleType("strumento.core.program")
    module.StrumentoProgram = _StubProgram
    monkeypatch.setitem(sys.modules, "strumento.core.program", module)


def test_wigner_2d_star_compiles_against_soccfg_and_acquires_against_proxy(dev, _stub_program):
    """Regression for the first --arm Wigner: the acquire soc is a Pyro proxy that is NOT a
    compile-time soccfg ('Proxy' object is not subscriptable). With the D14 split (a
    LegacyPyroRunner injected by the harness), the per-ray programs compile against the soccfg and
    acquire against the proxy — the run completes."""
    from strumento.core.legacy_pyro_runner import LegacyPyroRunner
    from strumento.packs.cqed.experiments.wigner_tomography import WignerTomography2D

    star = WignerTomography2D(dev, _AcquireProxy(), beta_max=1.0, n_radial=3, n_angle=2,
                              mode_name="cavity", runner=LegacyPyroRunner(_CompileCfg()))
    res = star.run()
    assert len(res.meta["parity_i"]) == 3 * 2      # stashed raw IQ, acquired via the proxy
    assert len(res.meta["parity_q"]) == 3 * 2


def test_wigner_2d_star_without_split_compiles_against_the_proxy_and_raises(dev, _stub_program):
    """Pins the bug: with no split runner (the base Runner default), the per-ray program compiles
    against the acquire proxy, reproducing the 'not subscriptable' failure the first --arm shot
    hit (wrapped, as there, in CompilationError). The harness must inject the split runner."""
    from strumento.core.experiment import CompilationError
    from strumento.packs.cqed.experiments.wigner_tomography import WignerTomography2D

    star = WignerTomography2D(dev, _AcquireProxy(), beta_max=1.0, n_radial=3, n_angle=1,
                              mode_name="cavity")  # runner=None -> base Runner
    with pytest.raises(CompilationError, match="compiled against the run-time soc"):
        star.run()


# -- displaced-parity sign convention (regression: coherent |+a0> must reconstruct at +a0) --------


def test_wigner_2d_sample_points_are_negative_of_applied_displacement():
    """Displaced-parity samples W(α) by applying D(-α); the pulse applies D(+β), so the sampled
    point is α = -β. On the θ=0 ray the applied magnitudes are [-2, 0, +2], so the sampled points
    are [+2, 0, -2] — the negation is the fix for the reflected (−α₀) reconstruction."""
    from strumento.packs.cqed.experiments.wigner_tomography import WignerTomography2D

    star = WignerTomography2D(None, soc=None, beta_max=2.0, n_radial=3, n_angle=1)
    pts = star.sample_points()
    assert np.allclose(pts, [2.0, 0.0, -2.0])


def test_wigner_2d_run_labels_point_as_negative_applied_displacement(dev, _stub_program):
    """The same sign convention in the acquire path: run()'s meta labels each point α = -β. A θ=0
    ray with applied magnitudes [-2, 0, +2] lands alpha_re = [+2, 0, -2] (alpha_im ≈ 0), so a
    coherent state prepared at +α₀ reconstructs at +α₀, not its parity-reflection -α₀."""
    from strumento.core.legacy_pyro_runner import LegacyPyroRunner
    from strumento.packs.cqed.experiments.wigner_tomography import WignerTomography2D

    star = WignerTomography2D(dev, _AcquireProxy(), beta_max=2.0, n_radial=3, n_angle=1,
                              mode_name="cavity", runner=LegacyPyroRunner(_CompileCfg()))
    res = star.run()
    assert np.allclose(res.meta["alpha_re"], [2.0, 0.0, -2.0])
    assert np.allclose(res.meta["alpha_im"], 0.0)
