"""cqed dispersive-IQ readout kind.

``DispersiveIQ`` rotates the IQ cloud so the g/e separation lies along I, thresholds each shot to
a discrete state, and reduces to the excited-state population. It is the ``ReadoutKind`` the
population fitters consume so they fit *populations* (matching the QILC measurement-vector
convention), not raw IQ.

This module is **pure**: no ``qick`` import.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from strumento.core.readout import DataSource


@dataclass
class DispersiveIQ:
    """Threshold-and-angle dispersive discriminator.

    ``angle`` (radians) rotates ``(I, Q)`` so the state separation is along the rotated I axis;
    a shot is ``|e⟩`` (1) when rotated-I exceeds ``threshold``, else ``|g⟩`` (0). Raw arrays are
    ``(..., 2)`` with the last axis ``(I, Q)``; ``reduce`` averages over the trailing shot axis.
    """

    threshold: float
    angle: float = 0.0
    data_source: DataSource = DataSource.QICK_ADC
    #: g/e blob centres in RAW (I, Q), threaded from ``ReadoutCal.g_center_iq``/``e_center_iq``.
    #: When both are set, :meth:`to_population` maps averaged IQ to excited-state population by
    #: linear interpolation between them; ``None`` leaves that conversion unavailable.
    g_center: tuple[float, float] | None = None
    e_center: tuple[float, float] | None = None
    #: Optional |f> centre used only by :meth:`to_level_populations`. A three-level conversion must
    #: fail rather than guess when the three response centres are nearly collinear.
    f_center: tuple[float, float] | None = None

    @classmethod
    def from_calib(cls, readout_cal) -> DispersiveIQ:
        """The discriminator a device has recorded: its ``ReadoutCal``'s threshold, angle and centres.

        ``threshold`` (ADC units, on the rotated I axis) and ``angle`` (rad — the same unit
        ``ReadoutCal.angle`` records, so no conversion) default to ``0.0`` when unset. Each of
        ``g_center_iq`` / ``e_center_iq`` / ``f_center_iq`` is taken independently: a recorded
        centre becomes a ``(float, float)`` tuple, an unset one stays ``None``. This is the default
        every cqed experiment reads with when no ``readout=`` is passed (#196).
        """

        def centre(value):
            if value is None:
                return None
            i, q = value
            return (float(i), float(q))

        return cls(
            threshold=float(readout_cal.threshold or 0.0),
            angle=float(readout_cal.angle or 0.0),
            g_center=centre(readout_cal.g_center_iq),
            e_center=centre(readout_cal.e_center_iq),
            f_center=centre(readout_cal.f_center_iq),
        )

    def project(self, raw):
        """Rotate ``(I, Q)`` onto the discrimination axis and return the signal (rotated I).

        Works on *averaged* IQ (no thresholding), so fitters can fit the readout signal directly
        when per-shot data isn't kept — the common case for a swept averaging program.
        """
        raw = np.asarray(raw, dtype=float)
        i, q = raw[..., 0], raw[..., 1]
        return i * np.cos(self.angle) + q * np.sin(self.angle)

    def to_population(self, raw):
        """Averaged IQ -> excited-state population ``P_e`` by linear interpolation of the g/e centres.

        Projects ``raw`` and the two blob centres onto the discrimination axis and returns
        ``(project(raw) - project(g)) / (project(e) - project(g))``, so averaged IQ at the g centre
        maps to 0 and at the e centre to 1. This is the hardware-path fix for parity/Wigner: the
        averaged rotated-I is in arbitrary units, not a population, until anchored to the blobs.

        **Raises** ``ValueError`` when the centres are unset (record ``g_center_iq``/``e_center_iq``
        from ``ge-readout-blobs``) or coincide (degenerate separation).
        """
        if self.g_center is None or self.e_center is None:
            raise ValueError(
                "DispersiveIQ.to_population requires g_center and e_center — record "
                "readout.g_center_iq/e_center_iq from a ge-readout-blobs run"
            )
        g = self.project(np.asarray(self.g_center, dtype=float))
        e = self.project(np.asarray(self.e_center, dtype=float))
        denom = e - g
        if denom == 0.0:
            raise ValueError("DispersiveIQ.to_population: g_center and e_center project to the same point")
        return (self.project(raw) - g) / denom

    def level_population_condition(self) -> float:
        """Return the condition number of the |g>, |e>, |f> IQ simplex.

        The affine conversion uses ``g`` as its origin and the ``g→e`` and ``g→f`` IQ vectors as a
        two-dimensional basis. A large condition number means those vectors are nearly collinear,
        so small IQ noise would become large, fictitious level populations.
        """
        if self.g_center is None or self.e_center is None or self.f_center is None:
            raise ValueError(
                "DispersiveIQ.level_population_condition requires g_center, e_center, and f_center "
                "from a g/e/f blob diagnostic"
            )
        g = np.asarray(self.g_center, dtype=float)
        e = np.asarray(self.e_center, dtype=float)
        f = np.asarray(self.f_center, dtype=float)
        return float(np.linalg.cond(np.column_stack((e - g, f - g))))

    def to_level_populations(self, raw, *, max_condition: float = 10.0):
        """Averaged IQ -> ``(P_g, P_e, P_f)`` from calibrated g/e/f response centres.

        The centres define an affine triangle in raw IQ space. Solving its barycentric coordinates
        gives the three level populations without clipping: out-of-simplex values remain visible as
        a calibration or state-preparation failure. The inversion refuses an ill-conditioned
        triangle rather than amplifying readout noise into a false transfer result.
        """
        if self.g_center is None or self.e_center is None or self.f_center is None:
            raise ValueError(
                "DispersiveIQ.to_level_populations requires g_center, e_center, and f_center — "
                "run the g/e/f blob diagnostic and record all three centres"
            )
        condition = self.level_population_condition()
        if not np.isfinite(condition) or condition > max_condition:
            raise ValueError(
                "DispersiveIQ.to_level_populations: g/e/f response is ill-conditioned "
                f"(condition={condition:.3g}, limit={max_condition:.3g})"
            )

        values = np.asarray(raw, dtype=float)
        if values.shape[-1:] != (2,):
            raise ValueError(
                "DispersiveIQ.to_level_populations expects raw IQ with a trailing (I, Q) axis"
            )
        g = np.asarray(self.g_center, dtype=float)
        e = np.asarray(self.e_center, dtype=float)
        f = np.asarray(self.f_center, dtype=float)
        basis = np.column_stack((e - g, f - g))
        flat = values.reshape(-1, 2)
        pe_pf = np.linalg.solve(basis, (flat - g).T).T
        populations = np.empty((flat.shape[0], 3), dtype=float)
        populations[:, 1:] = pe_pf
        populations[:, 0] = 1.0 - np.sum(pe_pf, axis=1)
        return populations.reshape(values.shape[:-1] + (3,))

    def _rotated_i(self, raw):
        return self.project(raw)

    def discriminate(self, raw):
        """Per-shot state labels (0 = |g⟩, 1 = |e⟩), shape = raw without the trailing IQ axis."""
        return (self._rotated_i(raw) > self.threshold).astype(int)

    def reduce(self, raw):
        """Excited-state population, averaged over the trailing shot axis."""
        return np.mean(self.discriminate(raw), axis=-1)

    def herald(self, raw, *, keep_state: int = 0):
        """Per-shot **pass mask** for a herald readout event.

        Discriminates each shot and returns ``True`` where the outcome equals ``keep_state`` —
        the shots to keep in post-selection (default: keep ``|g⟩``, i.e. a bare-ground herald).
        Feed the result into ``MeasurementResult.heralds[<lane>]``; ``postselect`` ANDs the lanes.
        """
        return self.discriminate(raw) == keep_state
