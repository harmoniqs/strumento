"""Averaged acquisition → the 1-D trace a fitter fits (Analyzer-side input conditioning, D13).

``MeasurementResult.avgi``/``avgq`` carry the acquisition's shape faithfully: for a swept program
that is ``(n_reads, n_points)``, with a **leading axis over the reads in one shot**, because a
sequence may measure more than once — a herald read, then the data read. A fitter wants one trace
against ``xpts``, so picking the data read is the Analyzer's job, done here once instead of in
five ``analyze`` methods.

By convention the **data read is the last read** in a shot; earlier reads are heralds/verification
(the per-shot post-selection lanes). Skipping this step leaves a single-read program's signal shaped
``(1, n_points)``, where the fitters' scalar seeds (``float(y[-1])``) raise and the resonator
fitters' ``column_stack`` silently mis-shapes.

Pure: no ``qick`` import.
"""

from __future__ import annotations

import numpy as np


def data_read(values) -> np.ndarray:
    """Reduce an averaged-acquisition array to the **data read's** 1-D trace over the sweep."""
    arr = np.asarray(values, dtype=float)
    while arr.ndim > 1:
        arr = arr[-1]
    return arr


def readout_signal(result, readout) -> np.ndarray:
    """Project the averaged IQ onto the readout's discrimination axis → the fitted 1-D signal."""
    iq = np.stack(
        [np.asarray(result.avgi, dtype=float), np.asarray(result.avgq, dtype=float)], axis=-1
    )
    return data_read(readout.project(iq))


def readout_population(result, readout) -> np.ndarray:
    """Convert the averaged IQ to excited-state population ``P_e`` via the readout's g/e centres.

    The population counterpart of :func:`readout_signal`: instead of the raw projected signal
    (arbitrary units) it returns ``P_e`` anchored to the recorded blob centres. Raises (through
    ``DispersiveIQ.to_population``) when the readout has no centres — the population chokepoint for
    parity/Wigner normalization.
    """
    iq = np.stack(
        [np.asarray(result.avgi, dtype=float), np.asarray(result.avgq, dtype=float)], axis=-1
    )
    return data_read(readout.to_population(iq))


def readout_iq(result) -> tuple[np.ndarray, np.ndarray]:
    """The data read's averaged ``(I, Q)`` traces — for fitters that want the complex response."""
    return data_read(result.avgi), data_read(result.avgq)
