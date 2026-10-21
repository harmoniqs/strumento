"""Framework-native two-dimensional resonator validation map.

The map is deliberately distinct from :class:`Punchout`: it preserves the full readout-frequency
and gain grid for validation, while Punchout remains the one-dimensional operating-ceiling
calibration. Each gain row is an ordinary dynamic-readout frequency sweep; the host merges those
rows so :meth:`Experiment.go` still owns the result, fit, display, and gate lifecycle.
"""

from __future__ import annotations

import numpy as np

from strumento.core.experiment import Experiment
from strumento.core.fitters import FitResult
from strumento.core.pulses import Seq
from strumento.core.resonator_fitters import fit_lorentzian
from strumento.core.result import MeasurementResult
from strumento.core.sweeps import Loop, Sweep
from strumento.core.wiring import LineRef
from strumento.packs.cqed.experiments._signal import data_read
from strumento.packs.cqed.library._gain import maxv_for

DEFAULT_GAIN_CODES: tuple[int, ...] = tuple(range(0, 1201, 100))
DEFAULT_FREQUENCY_POINTS = 61
DEFAULT_REPS = 50
DEFAULT_FINAL_DELAY_US = 5.0


class ResonatorPowerSweep(Experiment):
    """Map the readout response over frequency and lab-native probe gain codes.

    The default grid matches the direct-QICK validation script: 61 frequency points across
    ``f_ro +/- 5 MHz`` and gain codes from 0 through 1200 in increments of 100. The calibrated
    gain row is fitted as a normal resonator spectrum; zero gain remains a null-control row.
    """

    def __init__(
        self,
        device,
        *,
        freqs: Sweep | None = None,
        points: int = DEFAULT_FREQUENCY_POINTS,
        gain_codes=DEFAULT_GAIN_CODES,
        length_us: float | None = None,
        line=("readout", "probe"),
        runner=None,
    ):
        super().__init__(device, runner=runner)
        if points < 3:
            raise ValueError("resonator power sweep requires at least three frequency points")

        gains = tuple(int(code) for code in gain_codes)
        if not gains:
            raise ValueError("resonator power sweep requires at least one gain code")
        if gains != tuple(sorted(gains)) or len(set(gains)) != len(gains):
            raise ValueError("gain codes must be strictly increasing")

        ro = device.calib.readout
        self.points = int(points)
        self.freqs = freqs if freqs is not None else Sweep(
            start=float(ro.frequency) - 5.0,
            stop=float(ro.frequency) + 5.0,
            on="frequency",
        )
        self.gain_codes = gains
        self.reference_gain_code = int(ro.gain)
        if self.reference_gain_code not in gains:
            raise ValueError(
                f"gain codes must include calibrated readout gain {self.reference_gain_code}"
            )
        self.length_us = float(ro.readout_length if length_us is None else length_us)
        self.line = line if isinstance(line, LineRef) else LineRef(*line)
        frequency_for_gain = (
            float(self.freqs.start) if isinstance(self.freqs, Sweep) else float(ro.frequency)
        )
        self._gain_maxv = float(maxv_for(device, self.line, frequency_for_gain))
        self._active_gain_code = self.reference_gain_code

    def axes(self) -> tuple:
        return (Loop("frequency", self.points),)

    @property
    def active_gain_code(self) -> int:
        """The scalar gain being acquired, exposed to the simulator response model."""
        return self._active_gain_code

    def _row_sequence(self, gain_code: int) -> tuple[Seq, tuple]:
        return (
            self.device.readout.dispersive_measure(
                freq=self.freqs,
                gain=gain_code / self._gain_maxv,
                length=self.length_us,
                line=self.line,
            ),
            (),
        )

    def sequence(self) -> tuple[Seq, tuple]:
        """The calibrated row, used for ordinary structural compilation contracts."""
        return self._row_sequence(self.reference_gain_code)

    def _validate_buffer(self, reps: int) -> None:
        adc_ch = self.device.wiring.readout.adc_chs[0]
        readouts = self.device.soccfg_snapshot.get("readouts", ())
        try:
            capacity = int(readouts[adc_ch]["avg_maxlen"])
        except (IndexError, KeyError, TypeError) as exc:
            raise ValueError(f"ADC{adc_ch} has no avg_maxlen in the device snapshot") from exc
        required = int(reps) * self.points
        if required > capacity:
            raise ValueError(
                f"one gain row needs {required} accumulated points, exceeding ADC{adc_ch} "
                f"avg_maxlen={capacity}"
            )

    def acquire(self, soc, *, record=None, **run_kw) -> MeasurementResult:
        """Acquire one ordinary frequency sweep per gain, then merge the ordered rows."""
        reps = int(run_kw.get("reps", DEFAULT_REPS))
        self._validate_buffer(reps)
        rows: list[MeasurementResult] = []
        try:
            for gain_code in self.gain_codes:
                self._active_gain_code = gain_code
                seq, sweeps = self._row_sequence(gain_code)
                try:
                    row = self.runner.run(
                        self.device,
                        seq,
                        sweeps,
                        soc,
                        axes=self.axes(),
                        experiment=self,
                        # The merged map emits one aggregate record after all rows complete.
                        record=None,
                        **run_kw,
                    )
                except Exception as exc:
                    raise RuntimeError(f"resonator power sweep failed at gain code {gain_code}") from exc
                rows.append(row)
        finally:
            self._active_gain_code = self.reference_gain_code

        xpts = np.asarray(rows[0].xpts, dtype=float)
        if xpts.shape != (self.points,):
            raise RuntimeError(f"gain code {self.gain_codes[0]} returned frequency axis {xpts.shape}")
        for gain_code, row in zip(self.gain_codes[1:], rows[1:], strict=True):
            if not np.array_equal(np.asarray(row.xpts, dtype=float), xpts):
                raise RuntimeError(f"realized frequency grid changed at gain code {gain_code}")

        avgi = np.stack([data_read(row.avgi) for row in rows])
        avgq = np.stack([data_read(row.avgq) for row in rows])
        if avgi.shape != (len(self.gain_codes), self.points) or avgq.shape != avgi.shape:
            raise RuntimeError(
                f"expected map shape {(len(self.gain_codes), self.points)}, got I{avgi.shape} Q{avgq.shape}"
            )

        row_environment = [row.meta.get("environment") for row in rows]
        first_environment = row_environment[0]
        meta = {
            "gain_codes": list(self.gain_codes),
            "ypts": list(self.gain_codes),
            "x_axis": "readout_frequency_mhz",
            "y_axis": "readout_gain_code",
            "reference_gain_code": self.reference_gain_code,
            "acquisition_points": len(self.gain_codes) * self.points,
            "reps_per_row": reps,
            "final_delay_us": run_kw.get("final_delay", DEFAULT_FINAL_DELAY_US),
            "row_environment": row_environment,
            "requested_xpts": rows[0].meta.get("requested_xpts"),
        }
        if first_environment is not None:
            meta["environment"] = first_environment
        result = MeasurementResult(
            xpts=xpts,
            avgi=avgi,
            avgq=avgq,
            raw=[row.raw for row in rows],
            meta=meta,
        )
        if record is not None:
            from strumento.core.experiment import _emit_acquired

            _emit_acquired(record, result, experiment=self, reps=reps)
        return result

    def go(self, soc, *, reps: int = DEFAULT_REPS, final_delay: float = DEFAULT_FINAL_DELAY_US, **run_kw):
        """Run the map with the direct-reference repetition and final-delay defaults."""
        return super().go(soc, reps=reps, final_delay=final_delay, **run_kw)

    def recovery_map(self) -> dict[str, str]:
        return {"f0": "resonator_f0"}

    def analyze(self, result: MeasurementResult) -> FitResult:
        reference_row = self.gain_codes.index(self.reference_gain_code)
        magnitude = np.hypot(np.asarray(result.avgi, dtype=float), np.asarray(result.avgq, dtype=float))
        fit = fit_lorentzian(np.asarray(result.xpts, dtype=float), magnitude[reference_row])
        zero_row = self.gain_codes.index(0) if 0 in self.gain_codes else None

        params = dict(fit.params)
        params["reference_gain_code"] = float(self.reference_gain_code)
        derived = dict(fit.derived)
        derived.update(
            {
                "reference_gain_code": float(self.reference_gain_code),
                "reference_gain_max_magnitude": float(np.max(magnitude[reference_row])),
                "driven_max_magnitude": float(np.max(magnitude[1:] if zero_row == 0 else magnitude)),
                "zero_gain_max_magnitude": (
                    float(np.max(magnitude[zero_row])) if zero_row is not None else float("nan")
                ),
            }
        )
        return FitResult(params=params, stderr=dict(fit.stderr), derived=derived, success=bool(fit.success))
