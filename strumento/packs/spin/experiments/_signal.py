"""Spin readout signal projection — averaged acquisition -> 1-D trace for fitters.

Spin readout is typically single-channel (charge sensor or gate reflectometry), so the
projection is simpler than cQED's IQ rotation: we just extract the sensor signal from the
data read (last read in a multi-read sequence) and return the 1-D trace over the sweep.

Pure: no ``qick`` import.
"""

from __future__ import annotations

import numpy as np


def data_read(values) -> np.ndarray:
    """Reduce an averaged-acquisition array to the **data read's** 1-D trace over the sweep.

    Convention: the data read is the last read in a shot; earlier reads are heralds or
    initialization verification. A single-read program has shape ``(1, n_points)`` — this
    collapses it to ``(n_points,)`` so fitters get a clean 1-D trace against ``xpts``.
    """
    arr = np.asarray(values, dtype=float)
    while arr.ndim > 1:
        arr = arr[-1]
    return arr


def readout_signal(result, readout) -> np.ndarray:
    """Project the acquisition onto the spin readout's discrimination axis.

    For PSB/Elzerman/latched-PSB: the signal is threshold-based on a single ADC channel
    (charge sensor or gate reflectometry). We return the reduced I-channel trace — the
    sensor signal that gets thresholded for state discrimination.
    """
    return data_read(result.avgi)
