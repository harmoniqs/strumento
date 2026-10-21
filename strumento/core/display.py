"""Display registry — type-keyed plot-function dispatch (the D13 display layer).

The same MRO-walk registry pattern as the response-model registry in ``core/response.py``:
packs register ``(ExperimentType, plot_function)`` at import; ``display_for`` resolves
the plot function for an experiment instance by walking its MRO.

This module is **pure**: no ``qick``, no matplotlib (the plot functions that produce
figures live in the packs and import matplotlib lazily).
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any

_DISPLAYS: dict[type, Callable] = {}


def register_display(experiment_type: type, plot_fn: Callable) -> None:
    """Register ``plot_fn`` as the display function for ``experiment_type``.

    Packs call this at import (inside their ``plot.py``). Registering a type twice
    replaces the entry — same last-registration-wins rule as the response registry.
    """
    _DISPLAYS[experiment_type] = plot_fn


def displays(experiment_type: type) -> Callable[[Callable], Callable]:
    """Decorator form: ``@displays(MyExperiment)`` registers the decorated function."""
    def decorator(fn: Callable) -> Callable:
        register_display(experiment_type, fn)
        return fn
    return decorator


def _accepts_readout(fn: Callable) -> bool:
    """Whether ``fn`` opts into the optional ``readout=`` projection argument.

    Population-experiment display functions project averaged IQ onto the discrimination axis
    (``DispersiveIQ.project``), so they must plot the *same* signal the fitter consumed — they
    declare a ``readout`` parameter to receive it. Shape functions that don't (magnitude / bar /
    scalar, and the spin pack's two-argument functions) keep the ``(result, fit)`` signature and
    are called unchanged. Detected by signature so packs opt in per function, not globally.
    """
    params = inspect.signature(fn).parameters
    return "readout" in params or any(p.kind is p.VAR_KEYWORD for p in params.values())


def display_for(experiment: Any, result: Any, fit: Any) -> Any:
    """Resolve and call the registered plot function for ``experiment``.

    Walks the experiment's MRO (same as ``response_for`` in ``core/response.py``).
    A plot function that declares a ``readout`` parameter is handed ``experiment.readout`` so it
    can project onto the discrimination axis the fit used; otherwise it is called with
    ``(result, fit)``. Returns ``None`` if no plot function is registered.
    """
    cls = type(experiment)
    for base in cls.__mro__:
        fn = _DISPLAYS.get(base)
        if fn is not None:
            if _accepts_readout(fn):
                return fn(result, fit, readout=getattr(experiment, "readout", None))
            return fn(result, fit)
    return None
