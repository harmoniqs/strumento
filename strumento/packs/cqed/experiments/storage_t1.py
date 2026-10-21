"""Storage T1 experiment.

Measure the lifetime of excitations in a storage mode by swapping a photon into
storage, waiting a swept delay, swapping back, and measuring.

Pure at import: no ``qick`` (the Runner does the qick-facing acquire, lazily).
"""

from __future__ import annotations

from strumento.core.experiment import Experiment
from strumento.core.fitters import FitResult, fit_exp
from strumento.core.pulses import Seq
from strumento.core.result import MeasurementResult
from strumento.core.sweeps import Loop, Sweep
from strumento.packs.cqed.experiments._signal import readout_signal
from strumento.packs.cqed.library.composite import Fock, parity, prepare
from strumento.packs.cqed.library.targets import resolve_swap_pair
from strumento.packs.cqed.readout import DispersiveIQ
from strumento.packs.cqed.safety import MAX_TIME_SWEEP, MIN_TIME_SWEEP, validate_sweep_range


class StorageT1(Experiment):
    """Measure the excitation lifetime of a storage mode via swap-in, wait, swap-out.

    Sequence: prepare Fock |1> in the source mode -- ``swap_in`` (pi swap into
    storage) -- wait(t) -- ``swap_out`` (pi swap back) -- parity mapping on the
    source mode -- measure, with the idle delay ``t`` swept.  The storage
    excitation decays as exp(-t/T1_storage).
    Fitter: ``fit_exp`` (exponential decay ``y0 + amp * exp(-x/tau)``).

    Derived quantities:
        T1  : storage-mode lifetime in the same units as the delay sweep (typically us)
        tau : decay constant (same value as T1)

    Parameters:
        delays    : delay sweep; defaults to 0-10000 us on axis ``"t"``
        points    : number of sweep points (default 101)
        pair      : coupler pair for the swap, ``"<source>-<storage>"``; by default the only
                    calibrated swap involving ``mode_name`` (or the only one at all)
        mode_name : the storage mode to measure; selects the swap when ``pair`` is not given,
                    and must be one end of ``pair`` when it is. Defaults to the pair's second
                    element.
        readout   : ``DispersiveIQ`` readout config; defaults to the recorded calibration
                    (``DispersiveIQ.from_calib``)

    Usage::

        exp = StorageT1(dev, mode_name="S1")       # the swap cavity-S1, found from the mode
        outcome = exp.go(soc, reps=300)
        outcome.fit.derived["T1"]
    """

    def __init__(self, device, *, delays: Sweep | None = None, points: int = 101,
                 pair: str | None = None, mode_name: str | None = None,
                 readout: DispersiveIQ | None = None, runner=None):
        super().__init__(device, runner=runner)
        self.delays = delays if delays is not None else Sweep(start=0.0, stop=10000.0, on="t")
        validate_sweep_range(self.delays, min_val=MIN_TIME_SWEEP,
                             max_val=MAX_TIME_SWEEP, label="delay")
        self.points = points
        self.pair = resolve_swap_pair(device, pair, mode=mode_name, purpose=type(self).__name__)
        source, storage = self.pair.split("-", 1)
        if mode_name == source:
            raise ValueError(
                f"StorageT1: mode_name={mode_name!r} is the swap's source; name the storage mode "
                f"(the other end of {self.pair!r})"
            )
        self.mode_name = storage
        self.readout = readout if readout is not None else DispersiveIQ.from_calib(device.calib.readout)

    def axes(self) -> tuple:
        if isinstance(self.delays, Sweep) and self.delays.on is not None:
            return (Loop(self.delays.on, self.points),)
        return ()

    def sequence(self) -> tuple[Seq, tuple]:
        dev = self.device
        source_mode = self.pair.split("-", 1)[0]
        swap_in = dev.coupler.swap(self.pair, "pi")
        swap_out = dev.coupler.swap(self.pair, "pi")
        seq = prepare(dev, Fock(1))
        seq += Seq().play(swap_in).wait(self.delays).play(swap_out)
        # #100: parity() drives the readout and measures internally by default now -- chaining
        # our own .measure() here would double-measure.
        seq += parity(dev, source_mode)
        return seq, ()

    def analyze(self, result: MeasurementResult) -> FitResult:
        return fit_exp(result.xpts, readout_signal(result, self.readout))
