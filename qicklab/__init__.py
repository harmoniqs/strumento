"""qicklab — a first-principles QICK tProc-v2 experiment framework.

Device model + pulse IR + compiler + experiment/fitting lifecycle, with a
raw-qick escape hatch at every level.

Status: design phase. The design of record lives in ``docs/spec.md`` and the
implementation plan in ``docs/plan.md``. Public API is defined incrementally
as the tracer-bullet slices land (see the plan); this module will re-export
the stable surface (``Device``, ``Seq``, ``Sweep``, ``QickLabProgram``,
``MeasurementResult``, the envelope types, ``MockQickSocV2``) once slice 1 is
green.
"""

__version__ = "0.0.0"
