"""strumento — a first-principles QICK tProc-v2 experiment framework.

Device model + pulse IR + compiler + experiment/fitting lifecycle, with a raw-qick escape hatch
at every level. See ``docs/spec.md`` (design of record), ``docs/plan.md`` (slices), and
``docs/getting-started.md`` (walkthrough with examples).

The names re-exported here are the **pure** public surface — importable with no ``qick`` present.
The qick-facing layer (``StrumentoProgram``, ``MockQickSocV2``, the compiler) is imported from its
submodules explicitly, e.g.::

    from strumento import Device, Seq, Sweep, Gaussian     # pure — no board needed
    from strumento.core.program import StrumentoProgram    # qick-facing
    from strumento.core.mock import MockQickSocV2          # qick-facing

so that ``import strumento`` never pulls in ``qick``.
"""

from strumento.core.device import Device
from strumento.core.pulses import Arb, Const, FlatTop, Gaussian, Pulse, Seq
from strumento.core.result import MeasurementResult
from strumento.core.sweeps import Sweep
from strumento.core.wiring import LineRef

__version__ = "0.0.0"

__all__ = [
    "Device",
    "Seq",
    "Pulse",
    "Sweep",
    "Gaussian",
    "FlatTop",
    "Const",
    "Arb",
    "MeasurementResult",
    "LineRef",
    "__version__",
]
