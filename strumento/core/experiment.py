"""Experiment data-lifecycle scaffold (spec §9.1, D13).

The v1 code conflated acquire / fit / plot in one `Experiment.go`. Here the high-level contract
stays (`go`) but the internals are **separated collaborators**:

* **Runner** — builds a ``StrumentoProgram`` and acquires → :class:`MeasurementResult`. The only
  qick-facing step; imported lazily so this module stays pure at import and the Runner is
  injectable (a fake Runner drives the orchestration in tests without a board).
* **Analyzer** — the subclass's :meth:`Experiment.analyze` (fitters → :class:`FitResult`); **no
  plotting or printing**.
* **Display** — the subclass's :meth:`Experiment.display` (a figure from result + fit); default
  no-op so ``core`` needs no plotting dependency.

A concrete experiment subclasses :class:`Experiment`, implementing :meth:`sequence` (what to run)
and :meth:`analyze` (how to fit). ``go`` orchestrates and returns a typed
:class:`ExperimentOutcome` (raw result + fit + optional figure) — never stdout.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from strumento.core.fitters import FitResult
from strumento.core.pulses import Seq
from strumento.core.result import MeasurementResult


@dataclass
class ExperimentOutcome:
    """What an experiment run produces: the raw data, the fit, and an optional figure."""

    result: MeasurementResult
    fit: FitResult | None = None
    figure: Any = None


class Runner:
    """Builds a program from a sequence + sweeps and acquires it on a soc.

    The ``StrumentoProgram`` import is local so this module (and any experiment) imports without
    ``qick``; only an actual :meth:`run` touches it.
    """

    def run(self, device, seq: Seq, sweeps, soc, *, reps: int = 100, final_delay=None, **acquire_kw) -> MeasurementResult:
        from strumento.core.program import StrumentoProgram

        prog = StrumentoProgram(device, seq=seq, sweeps=sweeps, reps=reps, final_delay=final_delay, soc=soc)
        return prog.acquire(soc, **acquire_kw)


class Experiment(ABC):
    """Base for a characterization experiment (Rabi, T1, T2, …).

    Subclasses implement :meth:`sequence` and :meth:`analyze`; :meth:`display` is optional.
    """

    def __init__(self, device, *, runner: Runner | None = None) -> None:
        self.device = device
        self.runner = runner if runner is not None else Runner()

    @abstractmethod
    def sequence(self) -> tuple[Seq, tuple]:
        """Return ``(seq, sweeps)`` to run."""

    @abstractmethod
    def analyze(self, result: MeasurementResult) -> FitResult:
        """Fit the result → a typed :class:`FitResult` (no plotting, no printing)."""

    def display(self, result: MeasurementResult, fit: FitResult | None):
        """Build a figure from result + fit. Default: no-op (override to plot)."""
        return None

    def go(self, soc, *, analyze: bool = True, display: bool = False, **run_kw) -> ExperimentOutcome:
        seq, sweeps = self.sequence()
        result = self.runner.run(self.device, seq, sweeps, soc, **run_kw)
        fit = self.analyze(result) if analyze else None
        figure = self.display(result, fit) if display else None
        return ExperimentOutcome(result=result, fit=fit, figure=figure)
