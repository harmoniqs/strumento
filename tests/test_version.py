"""The package version has one source: ``pyproject.toml`` (#158).

``strumento.__version__`` reads the installed package metadata, so it can never disagree with
the version pip reports. Pure lane: no qick needed.
"""

from __future__ import annotations

import importlib.metadata
import sys

import strumento
from tests._paths import ROOT

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - exercised on 3.10 only
    import tomli as tomllib

EXPECTED = "0.1.0"


def test_pyproject_declares_the_release_version():
    with open(ROOT / "pyproject.toml", "rb") as f:
        assert tomllib.load(f)["project"]["version"] == EXPECTED


def test_runtime_version_is_the_installed_metadata():
    assert strumento.__version__ == importlib.metadata.version("strumento") == EXPECTED
