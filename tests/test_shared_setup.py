"""The shared test setup (``tests/conftest.py`` + ``tests/_paths.py``) itself.

The suite must not depend on the working directory, nor on whatever happens to sit in the
local ``devices/`` folder: device discovery reads what git tracks, never what is on disk.
"""

from __future__ import annotations

import ast
import shutil
import subprocess
from pathlib import Path

import pytest

from strumento.core.device import Device
from tests import _paths
from tests._paths import (
    DEVICES,
    ROOT,
    SHIPPED_DEVICE_YAMLS,
    SHIPPED_DEVICES_FALLBACK,
    shipped_device_yamls,
)

_HAS_GIT = shutil.which("git") is not None


def _git(cwd: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True)


def _make_device(root: Path, name: str) -> Path:
    yaml_path = root / "devices" / name / "device.yaml"
    yaml_path.parent.mkdir(parents=True)
    yaml_path.write_text("name: junk\n")
    return yaml_path


def test_paths_are_absolute_and_point_into_the_repo():
    assert ROOT.is_absolute() and (ROOT / "pyproject.toml").is_file()
    assert DEVICES == ROOT / "devices" and DEVICES.is_dir()
    for name in ("MULTIMODE_YAML", "LOOPBACK_YAML", "SPIN_YAML", "TWEEZER_YAML"):
        path = getattr(_paths, name)
        assert path.is_absolute() and path.is_file(), name


@pytest.mark.skipif(not _HAS_GIT, reason="needs git to build a scratch repo")
def test_untracked_device_folder_is_not_discovered(tmp_path):
    _git(tmp_path, "init", "-q")
    shipped = _make_device(tmp_path, "shipped_demo")
    _git(tmp_path, "add", str(shipped))
    _make_device(tmp_path, "junk_local_device")  # present on disk, never added

    found = shipped_device_yamls(tmp_path)

    assert found == [shipped]
    assert all("junk_local_device" not in str(p) for p in found)


def test_fallback_list_is_used_when_git_is_unavailable(monkeypatch, tmp_path):
    def no_git(*args, **kwargs):
        raise FileNotFoundError("git")

    monkeypatch.setattr(_paths.subprocess, "run", no_git)
    _make_device(tmp_path, "junk_local_device")

    assert shipped_device_yamls(ROOT) == [DEVICES / n / "device.yaml" for n in SHIPPED_DEVICES_FALLBACK]
    assert shipped_device_yamls(tmp_path) == []  # an untracked folder is never in the fallback


@pytest.mark.skipif(not _HAS_GIT, reason="needs git to read the tracked device list")
def test_fallback_list_matches_what_git_tracks():
    # The fallback (sdist, no git) must not rot away from the tracked devices.
    assert [p.parent.name for p in SHIPPED_DEVICE_YAMLS] == list(SHIPPED_DEVICES_FALLBACK)


def test_shared_device_fixtures_are_fresh_per_test(multimode_dev, loopback_dev, spin_dev, dev):
    for d in (multimode_dev, loopback_dev, spin_dev, dev):
        assert isinstance(d, Device)
    assert dev is not multimode_dev  # tests mutate calibration: never share one instance


_RELATIVE_PREFIXES = ("devices" + "/", "./" + "devices")


def _relative_device_paths(path: Path) -> list[int]:
    tree = ast.parse(path.read_text(), filename=str(path))
    return [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and node.value.startswith(_RELATIVE_PREFIXES)
    ]


def test_no_test_module_uses_a_cwd_relative_device_path():
    # _paths.py is exempt: its git pathspec is resolved with cwd=ROOT, never the process cwd.
    offenders = {
        str(p.relative_to(ROOT)): lines
        for p in sorted((ROOT / "tests").rglob("*.py"))
        if p != ROOT / "tests" / "_paths.py" and (lines := _relative_device_paths(p))
    }
    assert offenders == {}
