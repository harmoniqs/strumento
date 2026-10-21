"""Absolute repo/device paths and the shipped-device list, shared by the whole suite.

Every path here is anchored on this file, never on the working directory, so ``pytest`` runs
the same from anywhere. Import as ``from tests._paths import ...``; the fixtures built on these
live in ``tests/conftest.py``.

The shipped-device list is what **git tracks** (``devices/*/device.yaml``), not what is on
disk: a local, untracked device folder is never discovered, so it can neither fail a test nor
leak its name into a test ID. Without git (e.g. an sdist) it falls back to an explicit list.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEVICES = ROOT / "devices"
FIXTURES = DEVICES / "_fixtures"

MULTIMODE_YAML = DEVICES / "multimode_demo" / "device.yaml"
LOOPBACK_YAML = DEVICES / "loopback_demo" / "device.yaml"
SPIN_YAML = DEVICES / "spin_demo" / "device.yaml"
SPIN_EO_YAML = DEVICES / "spin_eo_demo" / "device.yaml"
SPIN_HOLE_YAML = DEVICES / "spin_hole_demo" / "device.yaml"
SPIN_ST_YAML = DEVICES / "spin_st_demo" / "device.yaml"
TWEEZER_YAML = DEVICES / "tweezer_demo" / "device.yaml"

# Used only when git can't answer (no git binary, or not a checkout). Kept in sync with the
# tracked devices by tests/test_shared_setup.py.
SHIPPED_DEVICES_FALLBACK = (
    "loopback_demo",
    "multimode_demo",
    "spin_demo",
    "spin_eo_demo",
    "spin_hole_demo",
    "spin_st_demo",
    "tweezer_demo",
)


def _tracked_device_yamls(root: Path) -> list[Path] | None:
    try:
        out = subprocess.run(
            ["git", "ls-files", "-z", "--", "devices/*/device.yaml"],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return None
    rels = [r for r in out.split("\0") if r]
    if not rels:
        return None
    return sorted(root / r for r in rels)


def shipped_device_yamls(root: Path = ROOT) -> list[Path]:
    """The shipped ``device.yaml`` files under ``root``: git-tracked, else the fallback list."""
    tracked = _tracked_device_yamls(root)
    if tracked is not None:
        return tracked
    candidates = (root / "devices" / name / "device.yaml" for name in SHIPPED_DEVICES_FALLBACK)
    return [p for p in candidates if p.is_file()]


SHIPPED_DEVICE_YAMLS = shipped_device_yamls()
