"""Meta-tests for the docs-snippet harness (issue #138, ``tests/_snippets.py``).

Each test writes a small fixture document plus a three-line docs test into a pytester directory
and runs an inner pytest session over it, so the harness is exercised exactly as
``tests/test_docs.py`` uses it. None of these need qick: the fixture snippets are plain Python.
"""

from __future__ import annotations

import textwrap

from tests._paths import ROOT

pytest_plugins = ["pytester"]

_INNER = """
from pathlib import Path

import pytest

from tests._snippets import document_params, run_snippet

DOCUMENTS = ["doc.md"]


@pytest.mark.parametrize("snippet", document_params(Path(__file__).parent, DOCUMENTS))
def test_snippet(snippet):
    run_snippet(snippet)
"""

_FENCE = "```"


def _doc(text: str) -> str:
    """Dedent a fixture document; ``'''`` stands in for a code fence (readability)."""
    return textwrap.dedent(text).lstrip("\n").replace("'''", _FENCE)


def _run(pytester, doc: str, *args: str):
    pytester.makefile(".md", doc=_doc(doc))
    pytester.makepyfile(test_inner=_INNER)
    return pytester.runpytest("-p", "no:cacheprovider", "-rA", *args)


# ---------------------------------------------------------------------------- extraction + ids


def test_each_python_block_is_one_test_named_by_document_and_line(pytester):
    result = _run(
        pytester,
        """
        # Title

        '''python
        x = 1
        '''

        Some prose.

        '''bash
        echo not python
        '''

        '''python
        assert x == 1
        '''
        """,
        "--collect-only",
        "-q",
    )
    result.stdout.fnmatch_lines(
        [
            "*test_snippet?doc.md:3?*",
            "*test_snippet?doc.md:13?*",
            "*test_snippet?doc.md:writes?*",
        ]
    )
    result.stdout.no_fnmatch_line("*doc.md:9*")


def test_a_passing_block_passes(pytester):
    result = _run(
        pytester,
        """
        '''python
        print("hello")
        '''
        """,
    )
    result.assert_outcomes(passed=2)


def test_blocks_share_one_namespace_in_document_order(pytester):
    result = _run(
        pytester,
        """
        '''python
        value = 41
        '''

        '''py
        value += 1
        assert value == 42
        '''
        """,
    )
    result.assert_outcomes(passed=3)


def test_selecting_one_block_still_runs_the_blocks_before_it(pytester):
    result = _run(
        pytester,
        """
        '''python
        value = 41
        '''

        '''python
        assert value == 41
        '''
        """,
        "-k",
        "doc.md:5",
    )
    result.assert_outcomes(passed=1, deselected=2)


# ---------------------------------------------------------------------------- failures


def test_a_raising_block_fails_naming_document_and_line(pytester):
    result = _run(
        pytester,
        """
        Intro.

        '''python
        x = 1
        raise ValueError("boom")
        '''
        """,
    )
    result.assert_outcomes(passed=1, failed=1)
    result.stdout.fnmatch_lines(["*doc.md:3*ValueError*boom*"])
    # the traceback points at the line in the document, not at harness internals
    result.stdout.fnmatch_lines(["*doc.md*line 5*"])


def test_a_syntax_error_fails_naming_document_and_line(pytester):
    result = _run(
        pytester,
        """
        '''python
        def broken(:
        '''
        """,
    )
    result.assert_outcomes(passed=1, failed=1)
    result.stdout.fnmatch_lines(["*doc.md:1*SyntaxError*"])


def test_a_block_after_a_failed_block_names_the_earlier_failure(pytester):
    result = _run(
        pytester,
        """
        '''python
        raise RuntimeError("first")
        undefined_later = 1
        '''

        '''python
        print(undefined_later)
        '''
        """,
    )
    result.assert_outcomes(passed=1, failed=2)
    result.stdout.fnmatch_lines(["*doc.md:6*NameError*", "*earlier*doc.md:1 (failed)*"])


# ---------------------------------------------------------------------------- opt-out markers


def test_skip_with_a_reason_is_not_executed_and_reports_the_reason(pytester):
    result = _run(
        pytester,
        """
        <!-- snippet: skip reason="hardware only" -->
        '''python
        raise RuntimeError("must not run")
        '''
        """,
    )
    result.assert_outcomes(passed=1, skipped=1)
    result.stdout.fnmatch_lines(["*hardware only*"])


def test_skip_without_a_reason_fails(pytester):
    result = _run(
        pytester,
        """
        <!-- snippet: skip -->
        '''python
        x = 1
        '''
        """,
    )
    result.assert_outcomes(passed=1, failed=1)
    result.stdout.fnmatch_lines(["*doc.md:2*reason*"])


def test_skip_with_an_empty_reason_fails(pytester):
    result = _run(
        pytester,
        """
        <!-- snippet: skip reason="" -->
        '''python
        x = 1
        '''
        """,
    )
    result.assert_outcomes(passed=1, failed=1)


def test_an_unknown_marker_or_attribute_fails(pytester):
    result = _run(
        pytester,
        """
        <!-- snippet: skipp reason="typo" -->
        '''python
        x = 1
        '''

        <!-- snippet: skip reaosn="typo" -->
        '''python
        y = 1
        '''
        """,
    )
    result.assert_outcomes(passed=1, failed=2)
    result.stdout.fnmatch_lines(["*doc.md:2*skipp*"])
    result.stdout.fnmatch_lines(["*doc.md:7*reaosn*"])


def test_a_marker_not_followed_by_a_block_fails_at_its_line(pytester):
    result = _run(
        pytester,
        """
        <!-- snippet: skip reason="dangling" -->
        Prose between the marker and the block.

        '''python
        x = 1
        '''
        """,
    )
    result.assert_outcomes(passed=2, failed=1)
    result.stdout.fnmatch_lines(["*doc.md:1*marker*"])


def test_a_malformed_marker_comment_fails(pytester):
    result = _run(
        pytester,
        """
        <!-- snippet skip reason="no colon" -->
        '''python
        x = 1
        '''
        """,
    )
    # the malformed marker still binds to the block below it, which then fails without running
    result.assert_outcomes(passed=1, failed=1)
    result.stdout.fnmatch_lines(["*doc.md:2*doc.md:1*malformed*"])


# ---------------------------------------------------------------------------- strict xfail


def test_xfail_with_an_issue_reason_that_raises_is_an_expected_failure(pytester):
    result = _run(
        pytester,
        """
        <!-- snippet: xfail reason="undefined name, fixed by #139" -->
        '''python
        print(not_defined)
        '''
        """,
    )
    result.assert_outcomes(passed=1, xfailed=1)
    result.stdout.fnmatch_lines(["*#139*"])


def test_xfail_that_unexpectedly_passes_fails_strictly(pytester):
    result = _run(
        pytester,
        """
        <!-- snippet: xfail reason="was broken, see #140" -->
        '''python
        x = 1
        '''
        """,
    )
    result.assert_outcomes(passed=1, failed=1)
    result.stdout.fnmatch_lines(["*XPASS(strict)*"])


def test_xfail_without_a_reason_or_without_an_issue_fails(pytester):
    result = _run(
        pytester,
        """
        <!-- snippet: xfail -->
        '''python
        raise RuntimeError
        '''

        <!-- snippet: xfail reason="broken, no issue named" -->
        '''python
        raise RuntimeError
        '''
        """,
    )
    result.assert_outcomes(passed=1, failed=2)
    result.stdout.fnmatch_lines(["*doc.md:7*issue*"])


# ---------------------------------------------------------------------------- expected output


def test_expected_output_that_matches_passes(pytester):
    result = _run(
        pytester,
        """
        '''python
        print("pi_gain =", 9350)
        print("ok")
        '''

        <!-- snippet: output -->
        '''text
        pi_gain = 9350
        ok
        '''
        """,
    )
    result.assert_outcomes(passed=2)


def test_expected_output_that_differs_fails_with_a_diff(pytester):
    result = _run(
        pytester,
        """
        '''python
        print("pi_gain =", 9351)
        '''

        <!-- snippet: output -->
        '''text
        pi_gain = 9350
        '''
        """,
    )
    result.assert_outcomes(passed=1, failed=1)
    result.stdout.fnmatch_lines(["*doc.md:1*output*", "*-pi_gain = 9350*", "*+pi_gain = 9351*"])


def test_expected_output_is_exact_for_floats_by_default(pytester):
    result = _run(
        pytester,
        """
        '''python
        print(0.30000000000000004)
        '''

        <!-- snippet: output -->
        '''text
        0.3
        '''
        """,
    )
    result.assert_outcomes(passed=1, failed=1)


def test_expected_output_tolerance_accepts_close_numbers(pytester):
    result = _run(
        pytester,
        """
        '''python
        print("pi_gain = 9351.2 +/- 12.1, ok")
        '''

        <!-- snippet: output rtol=1e-2 -->
        '''text
        pi_gain = 9350.0 +/- 12.0, ok
        '''
        """,
    )
    result.assert_outcomes(passed=2)


def test_expected_output_tolerance_rejects_far_numbers_and_changed_text(pytester):
    result = _run(
        pytester,
        """
        '''python
        print("pi_gain = 9500")
        '''

        <!-- snippet: output rtol=1e-3 -->
        '''text
        pi_gain = 9350
        '''

        '''python
        print("pi_gain = 9350, FAILED")
        '''

        <!-- snippet: output atol=1 -->
        '''text
        pi_gain = 9350, ok
        '''
        """,
    )
    result.assert_outcomes(passed=1, failed=2)


def test_an_output_marker_not_after_a_python_block_fails(pytester):
    result = _run(
        pytester,
        """
        Prose.

        <!-- snippet: output -->
        '''text
        hello
        '''
        """,
    )
    result.assert_outcomes(passed=1, failed=1)
    result.stdout.fnmatch_lines(["*doc.md:3*output*"])


# ---------------------------------------------------------------------------- optional deps


def test_a_missing_optional_dependency_skips_with_a_reason(pytester):
    result = _run(
        pytester,
        """
        '''python
        raise ModuleNotFoundError("No module named 'matplotlib'", name="matplotlib")
        '''
        """,
    )
    result.assert_outcomes(passed=1, skipped=1)
    result.stdout.fnmatch_lines(["*matplotlib*"])


def test_a_missing_required_module_fails(pytester):
    result = _run(
        pytester,
        """
        '''python
        import strumento_no_such_module
        '''
        """,
    )
    result.assert_outcomes(passed=1, failed=1)


# ---------------------------------------------------------------------------- sandbox


def test_relative_writes_and_home_land_in_a_temp_dir_and_reads_see_the_repo(pytester):
    result = _run(
        pytester,
        """
        '''python
        import os
        import tempfile
        from pathlib import Path

        assert Path("doc.md").is_file()          # the document's root is readable, relatively
        Path("scratch.txt").write_text("x")      # ...but relative writes land in the temp cwd
        home, tmp = Path(os.environ["HOME"]), Path(tempfile.gettempdir())
        assert home.parent == tmp.parent and home.parent.name.startswith("strumento-docs-")
        assert os.environ.get("MPLBACKEND") == "Agg"
        '''
        """,
    )
    result.assert_outcomes(passed=2)
    assert not (pytester.path / "scratch.txt").exists()


def test_a_block_that_writes_into_the_root_fails_the_writes_check(pytester):
    result = _run(
        pytester,
        """
        '''python
        from pathlib import Path

        (Path("doc.md").resolve().parent / "stray.txt").write_text("leak")
        '''
        """,
    )
    result.assert_outcomes(passed=1, failed=1)
    result.stdout.fnmatch_lines(["*doc.md:writes*stray.txt*"])


def test_a_block_cannot_open_a_network_connection(pytester):
    result = _run(
        pytester,
        """
        '''python
        import socket

        socket.create_connection(("192.0.2.1", 80), timeout=0.1)
        '''
        """,
    )
    result.assert_outcomes(passed=1, failed=1)
    result.stdout.fnmatch_lines(["*network*"])


def test_the_process_state_is_restored_after_a_document(pytester):
    pytester.makefile(".md", doc=_doc("'''python\nimport os\nos.environ['LEAK'] = '1'\n'''\n"))
    pytester.makepyfile(
        test_inner=_INNER
        + """

def test_zz_after():
    import os
    assert "LEAK" not in os.environ
    assert os.getcwd() == str(Path(__file__).parent)
"""
    )
    result = pytester.runpytest("-p", "no:cacheprovider")
    result.assert_outcomes(passed=3)


# ---------------------------------------------------------------------------- the qick lane


def test_the_docs_test_skips_with_a_reason_without_qick(pytester):
    # a fresh copy, so the module-level qick check runs again inside the inner session
    pytester.makepyfile(test_docs_copy=(ROOT / "tests" / "test_docs.py").read_text())
    result = pytester.runpytest(
        "-p", "tests._block_qick", "--block-qick", "-p", "no:cacheprovider", "-rs"
    )
    result.assert_outcomes(skipped=1)
    result.stdout.fnmatch_lines(["*qick*simulator*"])
