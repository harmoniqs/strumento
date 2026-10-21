"""Every real experiment fit validates against the published ``fit`` contract (issue #174).

The contract ``strumento schema fit`` publishes is generated from
:class:`~strumento.core.fitters.FitResult`; this cross-check pins it to what the experiments
actually return, so contract and code cannot drift apart again. Each registered cqed experiment
(on ``multimode_demo``) and spin experiment (on ``spin_demo``) is driven through
``SimulatorSoc`` with ``go(..., analyze=True)`` by the sim-smoke runners themselves
(``examples/sim_smoke/run_{cqed,spin}.py`` — same device, truth and reps), which hand back every
fit they produced; each must pass :func:`~strumento.core.schemas.validate`, and (issue #175) each
must serialise with the standard ``json.dumps`` and no ``default=`` hook — plain Python values all
the way down, so a fit can go into a task record or over MCP as-is.

Marked ``qick``: ``SimulatorSoc`` is a real v2 ``QickConfig``.
"""

from __future__ import annotations

import importlib.util
import json
from dataclasses import asdict

import pytest

from strumento.core.schemas import SchemaError, validate
from tests._paths import ROOT

pytest.importorskip("qick")

pytestmark = pytest.mark.qick


_JSON_NATIVE = (type(None), bool, int, float, str)


def _non_json_leaves(value, path: str = "") -> list[str]:
    """Every leaf that is not a plain JSON value, as ``path: type`` (``np.float64`` included)."""
    if isinstance(value, dict):
        return [leaf for k, v in value.items() for leaf in _non_json_leaves(v, f"{path}.{k}")]
    if isinstance(value, list):
        return [leaf for i, v in enumerate(value) for leaf in _non_json_leaves(v, f"{path}[{i}]")]
    return [] if type(value) in _JSON_NATIVE else [f"{path.lstrip('.')}: {type(value).__name__}"]


def _run(pack: str) -> tuple[list[dict], dict]:
    script = ROOT / "examples" / "sim_smoke" / f"run_{pack}.py"
    spec = importlib.util.spec_from_file_location(f"sim_smoke_fit_contract_{pack}", script)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    fits: dict = {}
    results = mod.run_all(fits=fits)
    return results, fits


@pytest.mark.parametrize("pack", ["cqed", "spin"])
def test_every_simulated_fit_validates_against_the_fit_contract(pack):
    results, fits = _run(pack)
    # Every experiment that ran its analysis handed its fit back (none silently dropped).
    analysed = {r["experiment"] for r in results if r.get("fit") in ("OK", "FAIL")}
    assert analysed and analysed == set(fits)
    rejected = {}
    for name, fit in sorted(fits.items()):
        try:
            validate("fit", asdict(fit))
        except SchemaError as err:
            rejected[name] = str(err).splitlines()[1:3]
    assert sorted(rejected) == [], rejected


@pytest.mark.parametrize("pack", ["cqed", "spin"])
def test_every_simulated_fit_serialises_with_plain_json_dumps(pack):
    _, fits = _run(pack)
    assert fits
    offenders = {}
    for name, fit in sorted(fits.items()):
        wire = asdict(fit)
        leaves = _non_json_leaves(wire)
        try:
            json.dumps(wire)
        except TypeError as err:
            leaves = leaves or [str(err)]
        if leaves:
            offenders[name] = leaves
    assert offenders == {}, offenders
