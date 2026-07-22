"""Oracle cross-check: the new lmfit-based strumento fitters vs the legacy v1
``multimode_expts/fitting/fitting.py`` (plan Task 18, spec §9.3 / D4).

The trusted old scipy fits confirm the new fits agree on the *same* data before the old code
is retired. Guarded: the v1 source is loaded from ``.refs-v1/multimode_expts`` (or the path in
env var ``STRUMENTO_V1_FITTING``); the whole module skips if it is unavailable.

Run with:  ``pytest -m oracle``
"""

import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import pytest

from strumento.core.fitters import fit_decaysin_detuned, fit_exp
from strumento.core.resonator_fitters import fit_lorentzian

pytestmark = pytest.mark.oracle


def _load_v1_fitting():
    """Import the legacy ``fitting.fitting`` module as a physics oracle, or return None."""
    env = os.environ.get("STRUMENTO_V1_FITTING")
    candidates = []
    if env:
        candidates.append(Path(env))
    repo_root = Path(__file__).resolve().parents[2]
    candidates.append(repo_root / ".refs-v1" / "multimode_expts")
    for mm_root in candidates:
        fitting_py = mm_root / "fitting" / "fitting.py"
        if fitting_py.exists():
            # `fitting` is an implicit namespace package once its parent is on sys.path;
            # `fitdecaysin` does `from fitting.fit_utils import ...`, so this must resolve.
            if str(mm_root) not in sys.path:
                sys.path.insert(0, str(mm_root))
            spec = importlib.util.spec_from_file_location("fitting.fitting", fitting_py)
            mod = importlib.util.module_from_spec(spec)
            sys.modules["fitting.fitting"] = mod
            spec.loader.exec_module(mod)
            return mod
    return None


v1 = _load_v1_fitting()

skip_no_v1 = pytest.mark.skipif(
    v1 is None,
    reason="v1 multimode_expts/fitting/fitting.py not found (.refs-v1 or $STRUMENTO_V1_FITTING)",
)


@skip_no_v1
def test_exp_agrees_with_v1_fitexp():
    rng = np.random.default_rng(100)
    t = np.linspace(0, 300.0, 121)
    true_T1 = 75.0
    y = 0.1 + 0.85 * np.exp(-t / true_T1) + rng.normal(0, 0.01, t.size)

    new = fit_exp(t, y).derived["T1"]
    # v1 expfunc: y0 + yscale*exp(-(x-x0)/decay); decay == T1
    pOpt, _ = v1.fitexp(t, y)
    old = pOpt[3]
    assert new == pytest.approx(old, rel=0.05)


@skip_no_v1
def test_decaysin_agrees_with_v1_fitdecaysin():
    rng = np.random.default_rng(101)
    t = np.linspace(0, 20.0, 401)
    true_det, true_T2 = 1.5, 8.0
    y = (
        0.5
        + 0.4 * np.exp(-t / true_T2) * np.cos(2 * np.pi * true_det * t + 0.3)
        + rng.normal(0, 0.008, t.size)
    )

    r = fit_decaysin_detuned(t, y)
    # v1 decaysin params: [yscale, freq, phase_deg, decay, y0, x0]; freq == detuning, decay == T2
    pOpt, _ = v1.fitdecaysin(t, y)
    old_freq, old_T2 = abs(pOpt[1]), abs(pOpt[3])
    assert r.derived["detuning"] == pytest.approx(old_freq, rel=0.05)
    assert r.derived["T2"] == pytest.approx(old_T2, rel=0.10)


@skip_no_v1
def test_lorentzian_agrees_with_v1_fitlor():
    rng = np.random.default_rng(102)
    fr = 7.0e9
    fwhm = 2.0e6
    hwhm = fwhm / 2
    f = np.linspace(fr - 20e6, fr + 20e6, 601)
    mag = 1.0 - 0.8 / (1 + ((f - fr) / hwhm) ** 2) + rng.normal(0, 0.005, f.size)

    r = fit_lorentzian(f, mag)
    # v1 lorfunc: y0 + yscale/(1+(x-x0)^2/xscale^2); x0 == f0, xscale == HWHM
    pOpt, _ = v1.fitlor(f, mag)
    old_f0, old_hwhm = pOpt[2], abs(pOpt[3])
    assert r.derived["f0"] == pytest.approx(old_f0, rel=1e-5)
    assert r.derived["fwhm"] == pytest.approx(2 * old_hwhm, rel=0.05)
