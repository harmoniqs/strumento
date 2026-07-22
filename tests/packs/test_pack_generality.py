"""core is modality-general (spec §15) — the honesty check that `core` isn't cqed-shaped.

All three packs are discovered through the `strumento.packs` entry-point group, and `core.device`
composes a device instance from each **with no core code aware of any pack**. If `core` had leaked a
superconducting assumption, one of these loads would fail.
"""

from pathlib import Path

import pytest

from strumento.core.device import Device

DEVICES = Path(__file__).parents[2] / "devices"


def test_all_packs_are_discoverable():
    from importlib.metadata import entry_points

    names = {ep.name for ep in entry_points(group="strumento.packs")}
    assert {"cqed", "tweezer", "spin"} <= names


@pytest.mark.parametrize(
    "instance,pack",
    [
        ("multimode_demo", "cqed"),
        ("tweezer_demo", "tweezer"),
        ("spin_demo", "spin"),
    ],
)
def test_core_composes_every_pack(instance, pack):
    dev = Device.load(DEVICES / instance / "device.yaml")
    assert dev.pack.__name__.endswith(pack)
    assert len(dev.elements) >= 1
    # every composed element's declared lines resolve against the instance wiring
    for el in dev.elements.values():
        for ref in el.lines():
            assert ref.line in dev.wiring.lines
