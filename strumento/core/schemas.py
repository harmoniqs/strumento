"""The typed contracts, machine-readable (D35) — JSON Schema for every wire form.

An agent (or a CI job, or another language's client) needs the *shape* of what strumento accepts
and returns without reading Python. Four pack-agnostic contracts cover the boundary:

===================  ====================================================================
``device``           a device instance's source form — the YAML on disk and
                     :class:`~strumento.core.config.DeviceConfig` (config is data, D10)
``fit``              :class:`~strumento.core.fitters.FitResult` — every fitter's return
``result``           :meth:`~strumento.core.result.MeasurementResult.to_wire`
``operator-model``   :meth:`~strumento.core.operators.OperatorModel.to_dict` — the codesign
                     export a solver consumes
===================  ====================================================================

plus one **pack-scoped** contract, ``calibration``: a pack's ``CALIBRATION`` model, whose field
descriptions carry the units. The ``device`` contract types ``calibration`` as an open dict
because its content is pack-owned, so this contract is how the pack's shape is published:
:func:`calibration_schema` resolves the pack by name through the same ``strumento.packs``
entry-point resolver ``Device.load`` uses and asks it for ``CALIBRATION``. Core imports no pack
by name.

**One object per contract, never two.** Each contract is a Pydantic model (or, for a stdlib
dataclass that already *is* the wire form, a ``TypeAdapter`` over it), so :func:`schema_for` and
:func:`validate` are two views of the same definition and cannot drift — no hand-maintained
schema dict to keep in step, and no extra runtime dependency (Pydantic is already required).
The two models here that restate a dataclass (:class:`DeviceSource`, :class:`MeasurementWire`)
are pinned to it by tests that validate real instances and compare field sets.

This module is **pure**: it never imports ``qick``.
"""

from __future__ import annotations

from functools import cache
from typing import Any

from pydantic import BaseModel, ConfigDict, TypeAdapter, ValidationError

from strumento.core.fitters import FitResult
from strumento.core.wiring import Wiring

#: The pack-agnostic contract names (:func:`schema_for` answers each with no further input).
CONTRACTS: tuple[str, ...] = ("device", "fit", "result", "operator-model")

#: The pack-scoped contract names: each needs a pack name (see :func:`calibration_schema`).
PACK_CONTRACTS: tuple[str, ...] = ("calibration",)


class SchemaError(ValueError):
    """Unknown contract name, or a payload that does not match its contract."""


class DeviceSource(BaseModel):
    """A device instance's source form: the instance YAML *and* ``DeviceConfig.to_dict()``.

    The two differ in exactly one place: on disk ``soccfg_snapshot`` is a **path** relative to the
    YAML (``Device.load`` reads it), while in a :class:`~strumento.core.config.DeviceConfig` it is
    the resolved snapshot **object**. Both are accepted so one schema describes the contract end
    to end. ``wiring`` is validated structurally through :class:`~strumento.core.wiring.Wiring`;
    ``calibration`` and ``topology`` stay open because their content is pack-owned (core is
    physics-free — hard rule 2).
    """

    model_config = ConfigDict(extra="forbid")

    pack: str
    topology: list[dict]
    wiring: Wiring
    calibration: dict
    couplings: list[dict] = []
    aliases: dict = {}
    soccfg_snapshot: str | dict | None = None


class MeasurementWire(BaseModel):
    """The JSON form of a :class:`~strumento.core.result.MeasurementResult` (``to_wire``).

    Shapes stay loose on purpose: ``shots`` nests as deep as the acquire did (leading axes index
    sweep points, the last axis is shots) and ``meta`` is free-form provenance. What the schema
    *does* pin is the field set, the required data fields, and that herald lanes are boolean.
    """

    model_config = ConfigDict(extra="forbid")

    xpts: list | float | None = None
    avgi: list | float
    avgq: list | float
    raw: Any = None
    shots: list | None = None
    heralds: dict[str, list[bool]] = {}
    herald_lanes: list[str] = []
    meta: dict = {}


class ComplexMatrixWire(BaseModel):
    """A complex matrix split into real/imaginary parts (the framework-neutral convention)."""

    model_config = ConfigDict(extra="forbid")

    real: list[list[float]]
    imag: list[list[float]]


class OperatorModelWire(BaseModel):
    """The JSON form of :meth:`~strumento.core.operators.OperatorModel.to_dict`."""

    model_config = ConfigDict(extra="forbid")

    H0: ComplexMatrixWire
    drives: dict[str, ComplexMatrixWire]
    basis_labels: list[list]
    drive_map: dict[str, str]
    dims: dict[str, int]
    frame: str
    provenance: dict = {}


#: contract name → the type whose Pydantic schema *is* the contract.
_CONTRACT_TYPES: dict[str, Any] = {
    "device": DeviceSource,
    "fit": FitResult,  # a stdlib dataclass that is already its own wire form
    "result": MeasurementWire,
    "operator-model": OperatorModelWire,
}


def _packs_hint() -> str:
    from strumento.core.device import installed_packs

    return f"installed packs: {', '.join(installed_packs()) or '(none)'}"


@cache
def _adapter(name: str) -> TypeAdapter:
    if name in PACK_CONTRACTS:
        raise SchemaError(
            f"the {name!r} contract is pack-scoped, so it needs a pack name "
            f"(strumento schema {name} --pack <name>); {_packs_hint()}"
        )
    try:
        target = _CONTRACT_TYPES[name]
    except KeyError:
        raise SchemaError(
            f"unknown contract {name!r}; valid names: {', '.join(CONTRACTS)}"
            f" (and, with --pack, {', '.join(PACK_CONTRACTS)})"
        ) from None
    return TypeAdapter(target)


def schema_for(name: str) -> dict:
    """JSON Schema for one contract. Unknown name → :class:`SchemaError` listing valid names."""
    return _adapter(name).json_schema()


def calibration_schema(pack: str | None) -> dict:
    """JSON Schema for a pack's calibration model (its ``CALIBRATION``), units in the descriptions.

    The pack is resolved by name through the ``strumento.packs`` entry points. A missing or unknown
    pack, or a pack that exports no ``CALIBRATION``, raises :class:`SchemaError`; the first two
    name the installed packs.
    """
    from strumento.core.device import _load_pack, installed_packs

    if not pack:
        raise SchemaError(
            "the 'calibration' contract is pack-scoped, so it needs a pack name "
            f"(strumento schema calibration --pack <name>); {_packs_hint()}"
        )
    if pack not in installed_packs():
        raise SchemaError(f"no pack {pack!r} for the 'calibration' contract; {_packs_hint()}")
    model = getattr(_load_pack(pack), "CALIBRATION", None)
    if model is None:
        raise SchemaError(
            f"pack {pack!r} exports no CALIBRATION model, so it has no calibration contract"
        )
    return model.model_json_schema()


def contract_schema(name: str, pack: str | None = None) -> dict:
    """The one dispatch both doors use: a pack-scoped contract needs ``pack``, the rest refuse it.

    ``strumento schema <name> [--pack <p>]`` and the MCP ``schema`` tool call this, so a pack given
    to a pack-agnostic contract is an error on both rather than silently ignored on one.
    """
    if name in PACK_CONTRACTS:
        return calibration_schema(pack)
    if pack is not None:
        raise SchemaError(
            f"a pack applies only to the pack-scoped contracts ({', '.join(PACK_CONTRACTS)}), "
            f"not {name!r}"
        )
    return schema_for(name)


def validate(name: str, payload: Any) -> Any:
    """Validate ``payload`` against a contract; returns the parsed model, raises on mismatch.

    The error text is Pydantic's (which locates the offending field), prefixed with the contract
    name — so a CLI or MCP caller can surface it as data rather than a traceback.
    """
    try:
        return _adapter(name).validate_python(payload)
    except ValidationError as err:
        raise SchemaError(f"payload does not match the {name!r} contract: {err}") from err
