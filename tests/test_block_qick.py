"""The ``--block-qick`` option (issue #124): the pure CI lane, reproduced on a qick machine.

Each test runs a small inner pytest session (pytester) with the option's plugin loaded, so the
option is exercised exactly as a contributor would use it: ``pytest --block-qick -m "not qick"``.
"""

from __future__ import annotations

import sys

import pytest

pytest_plugins = ["pytester"]

_PLUGIN = ("-p", "tests._block_qick")

_IMPORTS_QICK_AT_MODULE_SCOPE = """
import qick

def test_uses_qick():
    assert qick is not None
"""


def test_module_importing_qick_fails_to_collect_under_the_option(pytester):
    pytester.makepyfile(test_inner=_IMPORTS_QICK_AT_MODULE_SCOPE)
    result = pytester.runpytest(*_PLUGIN, "--block-qick")
    result.assert_outcomes(errors=1)
    result.stdout.fnmatch_lines(["*ImportError*"])


def test_same_module_collects_without_the_option(pytester):
    pytest.importorskip("qick")
    pytester.makepyfile(test_inner=_IMPORTS_QICK_AT_MODULE_SCOPE)
    result = pytester.runpytest(*_PLUGIN)
    result.assert_outcomes(passed=1)


def test_importorskip_qick_skips_under_the_option(pytester):
    pytester.makepyfile(
        test_inner="""
        import pytest

        qick = pytest.importorskip("qick")

        def test_uses_qick():
            assert qick is not None
        """
    )
    result = pytester.runpytest(*_PLUGIN, "--block-qick")
    result.assert_outcomes(skipped=1)


def test_qick_and_its_submodules_raise_inside_a_test_under_the_option(pytester):
    pytester.makepyfile(
        test_inner="""
        import importlib

        import pytest

        @pytest.mark.parametrize("name", ["qick", "qick.asm_v2"])
        def test_import_fails(name):
            with pytest.raises(ImportError):
                importlib.import_module(name)
        """
    )
    result = pytester.runpytest(*_PLUGIN, "--block-qick")
    result.assert_outcomes(passed=2)


def test_an_already_imported_qick_is_evicted_and_restored_afterwards(pytester):
    """qick imported before configure (e.g. by another plugin) is still blocked for the session;
    the outer interpreter gets its modules back when the inner session ends."""
    pytest.importorskip("qick")
    import qick  # noqa: F401 — make sure it is in sys.modules before the inner session

    before = sys.modules["qick"]
    pytester.makepyfile(
        test_inner="""
        import pytest

        def test_import_fails():
            with pytest.raises(ImportError):
                import qick  # noqa: F401
        """
    )
    result = pytester.runpytest(*_PLUGIN, "--block-qick")
    result.assert_outcomes(passed=1)
    assert sys.modules["qick"] is before
