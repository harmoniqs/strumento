"""Experiments that need per-shot data say so, and refuse out loud without it (#199, option B).

Per-shot data reaches a result only through the ``ShotSource`` protocol. The simulator implements
it; a board soc does not yet, and the job-server wire form carries averaged IQ only. Before this
guard, a shot-based experiment on such a soc failed silently (``active_reset``: an empty fit with
``success=False`` and no reason) or, worse, reported a shot quantity computed from averages
(``single_shot_readout``'s fidelity proxy, ``readout_state``'s histogram of two averaged points).

The contract this file pins, for every pack's registry:

1. **Declaration.** An experiment whose module reads ``.shots`` off a result declares
   ``needs_shots = True`` (a class attribute on :class:`~strumento.core.experiment.Experiment`,
   ``False`` by default). The readers are found from source, by an AST walk — not a hand list — so
   a new reader that forgets to declare fails here. Non-vacuity: the known readers must be found,
   and the detector is proven against synthetic sources.
2. **Refusal.** On a result **without** shots, each declared experiment's ``analyze`` returns
   ``success=False`` with ``derived["refused"]`` naming the missing per-shot data. It may keep
   quantities that are honest from averages alone (the averaged Ramsey of ``validation_check``,
   the centre angle of ``readout_state``); every quantity it withholds is then named in
   ``derived["refused_parts"]``. It never reports a shot quantity computed from averages.
3. **Only the result decides.** A result that carries shots analyses exactly as before, whatever
   soc (or no soc) produced it — a runner that reads per-shot IQ itself and hands ``analyze`` a
   result with ``shots`` set keeps working.
"""

from __future__ import annotations

import ast
import inspect
import sys
import textwrap
from dataclasses import replace

import numpy as np
import pytest

from strumento.core.experiment import SHOTS_REFUSAL, Experiment
from tests._v2gate import V2_SOCCFG
from tests.packs.test_experiment_contract import (
    _CQED_EXPANSION_NAMES,
    _CQED_SOCCFG,
    _SPIN_EXPERIMENT_NAMES,
    _SPIN_SOCCFG,
    CONTRACT_KWARGS,
    CONTRACT_PERTURBATION,
    CONTRACT_SEED,
    CQED_EXPANSION_TRUTH,
    REGISTRIES,
    SPIN_CONTRACT_TRUTH,
    _loopback_device,
    _multimode_device,
    _spin_device,
)

#: ``pack.name`` for every registered experiment.
ALL = {f"{pack}.{name}": cls for pack, reg in REGISTRIES.items() for name, cls in reg.items()}

#: The experiments known (on the tree #199 was filed against) to read ``result.shots``. The
#: detector below must find at least these — the proof that it has not gone blind.
KNOWN_READERS = {
    "cqed.active_reset",
    "cqed.readout_state",
    "cqed.validation_check",
    "spin.heralded_init",  # not in the issue's list: found by the source scan
}

#: Declared without reading ``.shots`` for any analysis — each with its written reason. A
#: declared experiment still checks ``result.shots`` to refuse, so it may also show up as a
#: reader; this table records WHY it is declared at all.
DECLARED_FOR_PURPOSE: dict[str, str] = {
    "cqed.single_shot_readout": (
        "its purpose is single-shot assignment fidelity; computed from the two averaged reads, "
        "the fidelity is a proxy (separation / 2) presented as a fidelity — the silent "
        "misleading result #199 exists to stop"
    ),
}

#: The experiments that need shots, as #199 resolves them. Pinned so the set cannot drift
#: silently either way: a new declaration (or a dropped one) must be written down here.
EXPECTED_DECLARED = KNOWN_READERS | set(DECLARED_FOR_PURPOSE)


def _reads_shots(source: str) -> bool:
    """Whether ``source`` reads a ``shots`` attribute off anything but ``self``.

    ``result.shots``, ``r.shots``, ``getattr(result, "shots")`` count; ``self.shots`` (an
    experiment's own shot-count knob) does not.
    """
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Attribute) and node.attr == "shots":
            if not (isinstance(node.value, ast.Name) and node.value.id == "self"):
                return True
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "getattr"
            and len(node.args) >= 2
            and isinstance(node.args[1], ast.Constant)
            and node.args[1].value == "shots"
        ):
            return True
    return False


def _readers() -> set[str]:
    return {
        key for key, cls in ALL.items()
        if _reads_shots(inspect.getsource(sys.modules[cls.__module__]))
    }


def _declared() -> set[str]:
    return {key for key, cls in ALL.items() if cls.needs_shots}


# -- the declaration (pure) ------------------------------------------------------------------


def test_the_base_experiment_needs_no_shots():
    assert Experiment.needs_shots is False


def test_the_refusal_reason_names_the_missing_per_shot_data():
    assert "needs per-shot data" in SHOTS_REFUSAL


def test_every_experiment_that_reads_shots_declares_it():
    """AC2: a module that reads ``.shots`` off a result belongs to an experiment that says so."""
    undeclared = sorted(_readers() - _declared())
    assert not undeclared, (
        f"these experiments read result.shots but do not declare needs_shots = True: "
        f"{undeclared}. Declare it, and refuse with SHOTS_REFUSAL on a result without shots."
    )


def test_the_reader_scan_is_not_vacuous():
    found = _readers()
    assert KNOWN_READERS <= found, f"the source scan lost known readers: {KNOWN_READERS - found}"


@pytest.mark.parametrize(
    ("source", "reads"),
    [
        ("def analyze(self, result):\n    return result.shots\n", True),
        ("def analyze(self, r):\n    x = np.asarray(r.shots)\n", True),
        ("def analyze(self, result):\n    return getattr(result, 'shots', None)\n", True),
        ("def __init__(self, shots):\n    self.shots = shots\n", False),
        ("def analyze(self, result):\n    return result.avgi  # no shots here\n", False),
    ],
)
def test_the_reader_detector_on_synthetic_sources(source, reads):
    assert _reads_shots(textwrap.dedent(source)) is reads


def test_the_declared_set_is_the_one_199_resolves():
    declared = _declared()
    assert declared == EXPECTED_DECLARED, (
        f"undocumented declarations: {sorted(declared - EXPECTED_DECLARED)}; "
        f"missing declarations: {sorted(EXPECTED_DECLARED - declared)}"
    )
    for key, reason in DECLARED_FOR_PURPOSE.items():
        assert reason.strip(), f"{key}: an empty reason"


# -- the refusal, through the real mock path (qick) -------------------------------------------


def _setup(key: str, *, simulated: bool):
    """(experiment, soc) for a registered ``pack.name``, on the contract's device dispatch."""
    from strumento.core.mock import MockQickSocV2
    from strumento.core.simulator import SimulatorSoc
    from strumento.packs.cqed.responses import truth_from_device
    from strumento.packs.spin.responses import truth_from_device as spin_truth_from_device

    _pack, name = key.split(".", 1)
    if name in _SPIN_EXPERIMENT_NAMES:
        dev, soccfg = _spin_device(), _SPIN_SOCCFG
        truth = spin_truth_from_device(dev, **SPIN_CONTRACT_TRUTH)
    elif name in _CQED_EXPANSION_NAMES:
        dev, soccfg = _multimode_device(), _CQED_SOCCFG
        truth = truth_from_device(dev, **CQED_EXPANSION_TRUTH)
    else:
        dev, soccfg = _loopback_device(), V2_SOCCFG
        truth = truth_from_device(dev, **CONTRACT_PERTURBATION)
    if simulated:
        soc = SimulatorSoc.from_snapshot(soccfg, truth=truth, seed=CONTRACT_SEED)
    else:
        soc = MockQickSocV2.from_snapshot(soccfg)
    return ALL[key](dev, **CONTRACT_KWARGS[name]), soc


def _assert_refused(fit, key: str) -> None:
    assert fit is not None, f"{key}: no fit"
    assert fit.success is False, f"{key}: succeeded without per-shot data: {fit.derived}"
    assert fit.derived.get("refused") == SHOTS_REFUSAL, f"{key}: no explicit reason: {fit.derived}"
    withheld = fit.derived.get("refused_parts", [])
    assert isinstance(withheld, list) and all(isinstance(k, str) for k in withheld)
    kept = set(fit.derived) - {"refused", "refused_parts"}
    assert not kept & set(withheld), f"{key}: reports quantities it says it withheld"
    if kept:  # a partial refusal names what it withheld
        assert withheld, f"{key}: keeps {sorted(kept)} but names nothing withheld"


@pytest.mark.qick
@pytest.mark.parametrize("key", sorted(EXPECTED_DECLARED))
def test_a_shot_experiment_on_the_mock_refuses_with_the_reason(key):
    """AC1: an averages-only soc (the mock) gives ``success=False`` and the explicit reason."""
    exp, soc = _setup(key, simulated=False)
    outcome = exp.go(soc, reps=8, progress=False)
    assert outcome.result.shots is None, f"{key}: the mock is supposed to return averages only"
    _assert_refused(outcome.fit, key)


@pytest.mark.qick
@pytest.mark.parametrize("key", sorted(EXPECTED_DECLARED))
def test_a_result_that_carries_shots_analyses_as_before_whatever_produced_it(key):
    """Only the result in hand decides. The simulator's result, copied off any soc, analyses to
    the same fit; and a mock (averages-only) acquisition with per-shot data attached by the
    caller — a runner that reads the board's per-shot IQ itself — analyses normally."""
    exp, sim = _setup(key, simulated=True)
    outcome = exp.go(sim, reps=64, progress=False)
    assert outcome.result.shots is not None, f"{key}: the simulator should provide shots"
    assert outcome.fit.success, f"{key}: the shots path no longer succeeds: {outcome.fit.derived}"
    assert "refused" not in outcome.fit.derived

    detached = replace(outcome.result, heralds={}, herald_lanes=())
    again, _ = _setup(key, simulated=True)
    refit = again.analyze(detached)
    assert refit.success and "refused" not in refit.derived
    assert refit.derived.keys() == outcome.fit.derived.keys()
    for k, v in outcome.fit.derived.items():
        np.testing.assert_allclose(np.asarray(refit.derived[k], dtype=float),
                                   np.asarray(v, dtype=float), equal_nan=True, err_msg=k)

    mocked, mock = _setup(key, simulated=False)
    averaged = mocked.go(mock, reps=8, progress=False, analyze=False).result
    assert averaged.shots is None
    grafted = replace(averaged, xpts=outcome.result.xpts, avgi=outcome.result.avgi,
                      avgq=outcome.result.avgq, shots=outcome.result.shots,
                      heralds={}, herald_lanes=())
    fit = mocked.analyze(grafted)
    assert fit.success and "refused" not in fit.derived, f"{key}: {fit.derived}"
