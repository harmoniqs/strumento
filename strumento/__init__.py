"""strumento — a first-principles QICK tProc-v2 experiment framework.

Device model + pulse IR + compiler + experiment/fitting lifecycle, with a raw-qick escape hatch
at every level. See ``docs/architecture.md`` (design + decision index) and
``docs/getting-started.md`` (walkthrough with examples).

The names re-exported here are the **pure** public surface — importable with no ``qick`` present.
The qick-facing layer (``StrumentoProgram``, ``MockQickSocV2``, the compiler) is imported from its
submodules explicitly, e.g.::

    from strumento import Device, Seq, Sweep, Gaussian     # pure — no board needed
    from strumento.core.program import StrumentoProgram    # qick-facing
    from strumento.core.mock import MockQickSocV2          # qick-facing

so that ``import strumento`` never pulls in ``qick``.
"""

from importlib import metadata as _metadata

from strumento.core.config import DeviceConfig, apply_updates
from strumento.core.device import Device
from strumento.core.pulses import Arb, Const, FlatTop, Gaussian, Pulse, Seq
from strumento.core.result import MeasurementResult
from strumento.core.store import FileConfigStore, record_calibration
from strumento.core.sweeps import Loop, Sweep
from strumento.core.wiring import LineRef

# The one version source is pyproject.toml; read it back from the installed metadata so the two
# can never disagree. Without metadata (e.g. a bare source tree on sys.path) say so honestly.
try:
    __version__ = _metadata.version("strumento")
except _metadata.PackageNotFoundError:  # pragma: no cover - only without an install
    __version__ = "0+unknown"

__all__ = [
    "Device",
    "DeviceConfig",
    "apply_updates",
    "FileConfigStore",
    "record_calibration",
    "Seq",
    "Pulse",
    "Loop",
    "Sweep",
    "Gaussian",
    "FlatTop",
    "Const",
    "Arb",
    "MeasurementResult",
    "LineRef",
    "__version__",
]
