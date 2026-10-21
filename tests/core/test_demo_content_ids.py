"""Every demo device's content id is pinned (#184).

A device's ``DeviceConfig.content_id()`` addresses a stored calibration forever, so documentation
changes to the calibration schema (field descriptions, units) must never move it. Two ids are
pinned per shipped device:

* ``loaded`` — ``Device.load(yaml).to_config().content_id()``, the id the store records;
* ``reserialized`` — the same config with ``calibration`` replaced by the pack model's
  ``model_dump``, so a changed default or field set in the calibration schema shows up here even
  though the loaded form is the raw YAML.

The values were computed before the unit descriptions were added. If a demo YAML's *data* changes
on purpose, recompute them; a schema-documentation change must leave them alone.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from strumento.core.device import Device
from tests._paths import SHIPPED_DEVICE_YAMLS

#: device folder → (loaded id, re-serialized id)
PINNED_CONTENT_IDS = {
    "loopback_demo": (
        "cfg-04b853992eea17bc1f132600e4c77770d1219f8613a304e98d95942d65a06d05",
        "cfg-79a2cc65fb0968db9a1ad24a7c8efb57685f5ac575bd1e84f42f9faa64caa27a",
    ),
    "multimode_demo": (
        "cfg-5d860116b1de0b1a1cebaf08fb6142f24032b2ae700a83909c9e6fa287f3f04b",
        "cfg-64ffab22499d22035ee3ff7a49da489c32c17a31561f1ea387fa54cb574f83a5",
    ),
    "spin_demo": (
        "cfg-73bd13defb21fe31e3917c57664122ed7f87e95f610a9f3f86d8e21109ef6d7a",
        "cfg-3a883ff2f022d6970a0bfa4e17f50da851a629c6a20f58f0a1890d43115fbedb",
    ),
    "spin_eo_demo": (
        "cfg-cda31ec80e2cdeb0e3a322a06bcd65842c5e5cec96f8c627215235fff896dd50",
        "cfg-1d5f1ea9daef21a2d2a816b239527029a1cd8c4729fbc2fb3f4983569462e7d9",
    ),
    "spin_hole_demo": (
        "cfg-6f1f64f387e6718db506d8d6402143626bcc368807fcddbf3c66d71209dfee04",
        "cfg-c74d540522b8a706d2f56522d5e7e65b2a683dfc0dd7deb4457431c57c3d2af2",
    ),
    "spin_st_demo": (
        "cfg-05186bfde6bbb83cc8f89e46b09208befefdd1b63f1928d31675f118238c48ce",
        "cfg-179d3738d7aa296ade4dd35b11b4d282af2fd62baf9ec859fb2d126da7f1444d",
    ),
    "tweezer_demo": (
        "cfg-a8777aa511faddf0fbb450b9972cec325cac044b5f301a57f63842118b0cb070",
        "cfg-a8777aa511faddf0fbb450b9972cec325cac044b5f301a57f63842118b0cb070",
    ),
}


def test_every_shipped_device_has_a_pinned_id():
    assert sorted(p.parent.name for p in SHIPPED_DEVICE_YAMLS) == sorted(PINNED_CONTENT_IDS)


@pytest.mark.parametrize("yaml_path", SHIPPED_DEVICE_YAMLS, ids=lambda p: p.parent.name)
def test_demo_device_content_id_is_unchanged(yaml_path):
    loaded, reserialized = PINNED_CONTENT_IDS[yaml_path.parent.name]
    dev = Device.load(yaml_path)
    cfg = dev.to_config()
    assert cfg.content_id() == loaded
    dumped = replace(cfg, calibration=dev.calib.model_dump(mode="json"))
    assert dumped.content_id() == reserialized
