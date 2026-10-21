"""Tests for NumberSplitting -- the chi_cav (dispersive-shift) step of the cat-state path.

Regression coverage: NumberSplitting hardcoded
``getattr(self.device, "cavity")`` and exposed no ``mode_name`` parameter, so it
raised ``AttributeError`` on any device whose cavity mode is not literally named
``cavity`` (e.g. a real multimode instance with modes named ``M1``/``M2``). Every
other cqed mode experiment (DisplacementCalibration, CavityT1, WignerTomography, ...)
takes an overridable ``mode_name``; this one was the lone outlier.

Modeled on test_photon_number_distribution.py. Pure: no qick needed for the build tests;
the sim run is qick-marked.
"""

from __future__ import annotations

import inspect
from dataclasses import replace

import pytest

from strumento.core.device import Device
from strumento.packs.cqed.experiments.number_splitting import NumberSplitting
from tests._paths import FIXTURES, MULTIMODE_YAML


@pytest.fixture
def dev_m1():
    """A multimode device whose cavity mode is named 'M1', not 'cavity' -- the shape
    a real lab instance takes. Built by relabeling the demo's cavity mode + its calib
    so the only difference from the demo is the *name* (isolating the mode-name routing)."""
    d = Device.load(MULTIMODE_YAML)
    # same calibration, new mode name
    d.calib.modes["M1"] = d.calib.modes.pop("cavity")
    # relabel the element; bind dev.M1 and drop the stale dev.cavity binding
    elem = replace(d.cavity.element, name="M1")
    d.elements["M1"] = elem
    del d.elements["cavity"]
    delattr(d, "cavity")
    setattr(d, "M1", type(elem).ops_cls(d, elem))  # bind dev.M1 the way _bind_namespaces does
    return d


def test_mode_name_param_exists():
    """The constructor must accept a mode_name keyword (the fix's contract)."""
    params = inspect.signature(NumberSplitting.__init__).parameters
    assert "mode_name" in params, "NumberSplitting must expose an overridable mode_name"


def test_default_mode_builds(dev):
    """Default (mode_name='cavity') still builds on the demo device -- no regression."""
    exp = NumberSplitting(dev, points=41)
    seq, _ = exp.sequence()
    assert len(seq.ops) >= 2  # displace + probe + measure


def test_a_mode_named_otherwise_builds(dev_m1):
    """The bug: on a device with mode 'M1' (no 'cavity'), sequence() must build."""
    exp = NumberSplitting(dev_m1, mode_name="M1", points=41)
    seq, _ = exp.sequence()  # must NOT raise AttributeError
    assert len(seq.ops) >= 2


@pytest.mark.qick
def test_a_mode_named_otherwise_runs_on_sim(dev_m1):
    """End-to-end on the SimulatorSoc: the M1-targeted run returns a chi_estimate."""
    import strumento.packs.cqed.responses  # noqa: F401  (register response models)
    from strumento.core.simulator import SimulatorSoc

    soc = SimulatorSoc.from_snapshot(FIXTURES / "soccfg_v2_cqed.json")
    exp = NumberSplitting(dev_m1, mode_name="M1", points=41)
    outcome = exp.go(soc, reps=64, progress=False)
    assert outcome.fit is not None
    assert "chi_estimate" in outcome.fit.derived
