"""The MCP session's fit summary keeps a fit's JSON values (issue #178, D38).

``FitResult.derived`` holds numbers *and* the JSON values an analyzer reports beside them — a
protocol name, a confusion matrix, a refusal's reason (D38). The MCP payload must carry those as
they are, turn NaN/inf into ``null`` at any depth (NaN is not JSON), and leave the numeric
``params``/``stderr`` exactly as before. These are unit tests of the summary itself: no soc, no
qick, no MCP SDK — the end-to-end check through a real session lives in ``tests/test_mcp.py``.
"""

from __future__ import annotations

import json
import math

import numpy as np

from strumento.core.fitters import FitResult
from strumento.mcp import _fit_summary

NAN, INF = float("nan"), float("inf")


def _reference_finite(value) -> float | None:
    """The summary's number rule as it stood before #178, frozen here as the reference."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _reference_fit_summary(fit) -> dict | None:
    """``_fit_summary`` as it stood before #178 — the byte-identity reference for numbers."""
    if fit is None:
        return None
    return {
        "success": bool(fit.success),
        "params": {key: _reference_finite(value) for key, value in fit.params.items()},
        "stderr": {key: _reference_finite(value) for key, value in fit.stderr.items()},
        "derived": {key: _reference_finite(value) for key, value in fit.derived.items()},
    }


def _strict_dumps(payload) -> str:
    """JSON with no ``default=`` hook and no NaN/Infinity tokens — what a strict client parses."""
    return json.dumps(payload, allow_nan=False)


def test_non_numeric_derived_values_arrive_as_given_and_nan_becomes_null_at_any_depth():
    fit = FitResult(
        params={"a": 1.0},
        stderr={"a": NAN},
        derived={
            "refused": "fixed-period fit needs a blob-anchored population",
            "protocol": "bb1",
            "confusion_matrix": [[0.9, 0.1], [0.2, 0.8]],
            "T1": 42.0,
            "nested": {"l": [1.0, NAN, {"x": INF}], "neg": -INF},
            "flag": True,
            "off": False,
            "missing": None,
            "pair": (1.5, NAN),
            "np_float": np.float64(2.5),
            "np_nan": np.float64("nan"),
            "np_bool": np.bool_(True),
            "np_int": np.int64(3),
            "array": np.array([[1.0, np.nan], [np.inf, 4.0]]),
        },
        success=False,
    )

    summary = _fit_summary(fit)
    derived = summary["derived"]

    assert derived["refused"] == "fixed-period fit needs a blob-anchored population"
    assert derived["protocol"] == "bb1"
    assert derived["confusion_matrix"] == [[0.9, 0.1], [0.2, 0.8]]
    assert derived["T1"] == 42.0
    assert derived["nested"] == {"l": [1.0, None, {"x": None}], "neg": None}
    assert derived["flag"] is True and derived["off"] is False      # a bool is not the number 1
    assert derived["missing"] is None
    assert derived["pair"] == [1.5, None]                          # a tuple arrives as a list
    assert derived["np_float"] == 2.5 and type(derived["np_float"]) is float
    assert derived["np_nan"] is None
    assert derived["np_bool"] is True
    assert derived["np_int"] == 3.0 and type(derived["np_int"]) is float  # numbers stay floats
    assert derived["array"] == [[1.0, None], [None, 4.0]]
    assert summary["success"] is False
    assert summary["stderr"] == {"a": None}

    # valid, strict JSON: no default= hook needed, no NaN/Infinity tokens
    text = _strict_dumps(summary)
    assert json.loads(text) == summary


def test_numeric_params_stderr_and_derived_are_byte_identical_to_the_old_summary():
    """Numbers keep their exact JSON form: ints and numpy scalars as floats, NaN/inf as null."""
    fit = FitResult(
        params={"a": 1.0, "b": -2.5e-7, "c": 3, "d": np.float64(0.1), "e": np.int64(7),
                "f": np.float32(1.25), "g": 0.0},
        stderr={"a": NAN, "b": INF, "c": -INF, "d": np.float64("nan"), "e": 1e-12,
                "f": np.float32("inf"), "g": 2},
        derived={"T1": 42.0, "T1_stderr": NAN, "count": 5, "pi_gain": np.float64(9350.25),
                 "big": 1e300, "tiny": 5e-324, "inf": -INF},
        success=True,
    )

    new, old = _fit_summary(fit), _reference_fit_summary(fit)
    for key in ("success", "params", "stderr", "derived"):
        assert _strict_dumps(new[key]) == _strict_dumps(old[key]), key
    assert _strict_dumps(new) == _strict_dumps(old)
    assert _fit_summary(None) is None
