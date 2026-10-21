"""Bosonic experiments pick their mode, coupler pair and transition from the device.

No experiment or factory hardcodes an element name. Given ``None`` (the default), each resolves
its target from the device:

* a **mode**: the one bosonic mode whose role is ``"cavity"``; else the device's only bosonic
  mode; else a ``ValueError`` naming the candidates;
* a **swap pair**: the only calibrated coupler swap (or the only one touching a named mode).

An explicit name is used as given. Pure lane: no qick needed.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from strumento.core.config import apply_updates
from strumento.core.device import Device
from strumento.packs.cqed.library.targets import resolve_mode, resolve_swap_pair
from tests._paths import MULTIMODE_YAML


@pytest.fixture
def dev():
    return Device.load(MULTIMODE_YAML)


def _relabel(dev, old: str, new: str):
    """``dev`` with bosonic mode ``old`` renamed to ``new`` (element, namespace and calibration)."""
    d = Device.load(MULTIMODE_YAML)
    d.calib.modes[new] = d.calib.modes.pop(old)
    elem = replace(d.elements.pop(old), name=new)
    d.elements[new] = elem
    setattr(d, new, type(elem).ops_cls(d, elem))
    return d


# -- modes ----------------------------------------------------------------------------------


def test_the_cavity_role_picks_the_mode(dev):
    assert resolve_mode(dev, None, purpose="test") == "cavity"


def test_the_cavity_is_found_by_role_not_by_name(dev):
    assert resolve_mode(_relabel(dev, "cavity", "c1"), None, purpose="test") == "c1"


def test_an_explicit_mode_is_used_as_given(dev):
    assert resolve_mode(dev, "S2", purpose="test") == "S2"


def test_no_cavity_role_and_several_modes_is_refused_naming_them(dev):
    dev.elements["cavity"] = replace(dev.elements["cavity"], role="storage")
    with pytest.raises(ValueError) as err:
        resolve_mode(dev, None, purpose="CavityT1")
    msg = str(err.value)
    assert "CavityT1" in msg and "mode_name=" in msg
    assert all(name in msg for name in ("cavity", "S1", "S2"))


def test_a_single_bosonic_mode_is_picked_whatever_its_role():
    config, _ = apply_updates(Device.load(MULTIMODE_YAML).to_config(), {})
    config = replace(
        config,
        topology=[t for t in config.topology if t["name"] not in ("S1", "S2")],
    )
    d = Device.from_config(config)
    d.elements["cavity"] = replace(d.elements["cavity"], role="storage")
    assert resolve_mode(d, None, purpose="test") == "cavity"


def test_a_device_without_modes_is_refused():
    loopback = Device.load(MULTIMODE_YAML.parent.parent / "loopback_demo" / "device.yaml")
    with pytest.raises(ValueError, match="no bosonic mode"):
        resolve_mode(loopback, None, purpose="test")


# -- swap pairs -----------------------------------------------------------------------------


def test_several_swaps_are_refused_naming_them(dev):
    with pytest.raises(ValueError) as err:
        resolve_swap_pair(dev, None, purpose="SwapRabi")
    msg = str(err.value)
    assert "SwapRabi" in msg and "pair=" in msg and "cavity-S1" in msg and "cavity-S2" in msg


def test_the_only_swap_is_picked(dev):
    dev.calib.swaps.pop("cavity-S2")
    assert resolve_swap_pair(dev, None, purpose="test") == "cavity-S1"


def test_a_mode_narrows_the_swaps_to_the_one_touching_it(dev):
    assert resolve_swap_pair(dev, None, mode="S2", purpose="test") == "cavity-S2"


def test_an_explicit_pair_is_used_as_given(dev):
    assert resolve_swap_pair(dev, "cavity-S2", purpose="test") == "cavity-S2"


def test_a_pair_that_does_not_touch_the_named_mode_is_refused(dev):
    with pytest.raises(ValueError, match="does not involve"):
        resolve_swap_pair(dev, "cavity-S1", mode="S2", purpose="test")



# -- the experiments and factories use them -------------------------------------------------


def test_a_cavity_experiment_follows_the_cavity_role_not_its_name(dev):
    from strumento.packs.cqed.experiments.cavity_t1 import CavityT1

    assert CavityT1(dev).mode_name == "cavity"
    assert CavityT1(_relabel(dev, "cavity", "c1")).mode_name == "c1"


def test_the_parity_composite_follows_the_cavity_role(dev):
    from strumento.packs.cqed.library.composite import parity

    renamed = _relabel(dev, "cavity", "c1")
    assert parity(renamed, measure=False).ops  # builds on the renamed cavity, no name needed


def test_a_swap_experiment_must_name_its_pair_when_several_are_calibrated(dev):
    from strumento.packs.cqed.experiments.swap_rabi import SwapRabi

    with pytest.raises(ValueError, match="pass pair="):
        SwapRabi(dev)
    assert SwapRabi(dev, pair="cavity-S2").pair == "cavity-S2"


def test_storage_t1_measures_the_storage_mode_it_is_given(dev):
    """mode_name used to be accepted and ignored; it now selects the swap and is the mode measured."""
    from strumento.packs.cqed.experiments.storage_t1 import StorageT1

    exp = StorageT1(dev, mode_name="S2")
    assert (exp.pair, exp.mode_name) == ("cavity-S2", "S2")
    seq, _ = exp.sequence()
    assert any("swap_cavity-S2" in getattr(op, "label", "") for op in seq.ops)


def test_storage_t1_derives_the_storage_mode_from_an_explicit_pair(dev):
    from strumento.packs.cqed.experiments.storage_t1 import StorageT1

    assert StorageT1(dev, pair="cavity-S1").mode_name == "S1"


def test_storage_t1_refuses_the_swap_source_as_its_storage_mode(dev):
    from strumento.packs.cqed.experiments.storage_t1 import StorageT1

    with pytest.raises(ValueError, match="source"):
        StorageT1(dev, pair="cavity-S1", mode_name="cavity")


def test_coupler_swap_must_name_its_pair_when_several_are_calibrated(dev):
    with pytest.raises(ValueError, match="pass pair="):
        dev.coupler.swap()
    assert dev.coupler.swap("cavity-S2").label == "swap_cavity-S2_pi"
