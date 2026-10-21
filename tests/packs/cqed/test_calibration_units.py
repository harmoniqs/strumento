"""Every cqed calibration field states its unit in the schema (#184).

The field names stay as they are (renaming would break every device YAML and every stored
calibration), so the unit lives in each field's ``description``. A description reads
``"<unit> — <meaning>"``: the part before the dash names a unit (``MHz``, ``µs``, ``deg``,
``DAC integer code``, ...), or says ``dimensionless`` or ``name or label``. A field whose type is
another calibration model (or a dict or list of them) is a nested section: its first part says
what the section holds instead of a unit.

The units table in ``docs/packs-and-devices.md`` is generated from the same descriptions and
checked here, so the docs and the schema can't drift. Pure lane: no qick needed.
"""

from __future__ import annotations

import inspect
import re
import typing

import pytest
from pydantic import BaseModel

from strumento.packs.cqed import calibration as cqed_calibration
from strumento.packs.cqed.calibration import Calibration
from tests._paths import ROOT

#: Phrases that count as a unit in the part of a description before the dash.
UNIT_TOKENS = (
    "GHz",
    "MHz",
    "kHz",
    "Hz",
    "µs",
    "us",
    "ns",
    "ms",
    "deg",
    "rad",
    "DAC integer code",
    "fraction of full scale",
    "ADC units",
    "dimensionless",
    "name or label",
)

_UNIT_RE = re.compile(r"(?<![\w])(?:" + "|".join(re.escape(t) for t in UNIT_TOKENS) + r")(?![\w])")
_SEP = " — "

DOC = ROOT / "docs" / "packs-and-devices.md"
TABLE_BEGIN = "<!-- cqed-calibration-units: begin (generated from the schema; checked by tests) -->"
TABLE_END = "<!-- cqed-calibration-units: end -->"


def _models_in(annotation) -> list[type[BaseModel]]:
    """The pydantic models an annotation mentions (``X``, ``X | None``, ``dict[str, X]``, ...)."""
    if inspect.isclass(annotation) and issubclass(annotation, BaseModel):
        return [annotation]
    found: list[type[BaseModel]] = []
    for arg in typing.get_args(annotation):
        found.extend(_models_in(arg))
    return found


def _reachable(root: type[BaseModel]) -> list[type[BaseModel]]:
    """``root`` and every model nested under it, in first-seen order."""
    seen: list[type[BaseModel]] = []
    todo = [root]
    while todo:
        model = todo.pop(0)
        if model in seen:
            continue
        seen.append(model)
        for field in model.model_fields.values():
            todo.extend(_models_in(field.annotation))
    return seen


def _calibration_models() -> list[type[BaseModel]]:
    """Every model defined in the calibration module, plus every model reachable from it."""
    defined = [
        obj
        for _, obj in inspect.getmembers(cqed_calibration, inspect.isclass)
        if issubclass(obj, BaseModel) and obj.__module__ == cqed_calibration.__name__
    ]
    models = _reachable(Calibration)
    for model in defined:
        for m in _reachable(model):
            if m not in models:
                models.append(m)
    return models


def _fields() -> list[tuple[type[BaseModel], str]]:
    return [(m, name) for m in _calibration_models() for name in m.model_fields]


def _id(param) -> str:
    model, name = param
    return f"{model.__name__}.{name}"


def test_the_enumeration_finds_every_calibration_model_and_field():
    names = {m.__name__ for m in _calibration_models()}
    assert {
        "Calibration", "QubitCal", "ReadoutCal", "PulseCal", "ModeCal",
        "SidebandPulseCal", "SwapCal", "MultiphotonCal",
    } <= names
    assert len(_fields()) == 60


@pytest.mark.parametrize("field", _fields(), ids=_id)
def test_every_calibration_field_states_its_unit(field):
    model, name = field
    info = model.model_fields[name]
    desc = (info.description or "").strip()
    assert desc, f"{model.__name__}.{name} has no description; say its unit and what it is"
    assert _SEP in desc, (
        f"{model.__name__}.{name}: write the description as '<unit>{_SEP}<meaning>', got {desc!r}"
    )
    unit, meaning = desc.split(_SEP, 1)
    assert unit.strip() and meaning.strip(), f"{model.__name__}.{name}: empty unit or meaning"
    if _models_in(info.annotation):
        return  # a nested section: its first part says what it holds
    assert _UNIT_RE.search(unit), (
        f"{model.__name__}.{name}: {unit!r} names no unit; use one of {', '.join(UNIT_TOKENS)}"
    )


def test_a_field_without_a_unit_is_caught():
    """The unit pattern is not vacuous: plain words don't pass for a unit."""
    assert not _UNIT_RE.search("the qubit frequency")
    assert not _UNIT_RE.search("status")  # 'us' inside a word is not microseconds
    assert _UNIT_RE.search("MHz")
    assert _UNIT_RE.search("µs")
    assert _UNIT_RE.search("DAC integer code per |α|")


def units_table() -> str:
    """The docs table, built from the schema's descriptions: one row per field."""
    rows = ["| Model | Field | Unit | Meaning |", "|---|---|---|---|"]
    for model, name in _fields():
        desc = model.model_fields[name].description or ""
        unit, _, meaning = desc.partition(_SEP)
        meaning = meaning.replace("|", "\\|")
        unit = unit.replace("|", "\\|")
        rows.append(f"| `{model.__name__}` | `{name}` | {unit} | {meaning} |")
    return "\n".join(rows)


def test_the_docs_units_table_matches_the_schema():
    text = DOC.read_text(encoding="utf-8")
    assert TABLE_BEGIN in text and TABLE_END in text, (
        f"{DOC.name} needs the generated units table between its two marker comments"
    )
    documented = text.split(TABLE_BEGIN, 1)[1].split(TABLE_END, 1)[0].strip()
    expected = units_table()
    assert documented == expected, (
        f"the units table in {DOC.name} is out of date; replace it with:\n\n{expected}\n"
    )
