"""Wigner tomography experiment.

Full phase-space reconstruction of an arbitrary cavity state via displaced parity
measurements. Sweeps displacement beta over a grid in phase space.

Pure at import: no ``qick`` (the Runner does the qick-facing acquire, lazily).
"""

from __future__ import annotations

import numpy as np

from strumento.core.experiment import Experiment
from strumento.core.fitters import FitResult
from strumento.core.pulses import Seq
from strumento.core.result import MeasurementResult
from strumento.core.sweeps import Loop, Sweep
from strumento.packs.cqed.experiments._signal import readout_iq, readout_signal
from strumento.packs.cqed.library.composite import Fock, parity, prepare
from strumento.packs.cqed.library.targets import resolve_mode
from strumento.packs.cqed.readout import DispersiveIQ


class WignerTomography(Experiment):
    """Displaced-parity Wigner sampling of a prepared, Kerr-evolved cavity state.

    Sequence: **prepare** a coherent state ``|init_alpha⟩`` (a calibrated displacement; skipped when
    ``init_alpha == 0``) → **free-evolve** under the mode's self-Kerr for ``kerr_wait_us`` (skipped
    when ``0``) → **tomography displacement** by ``beta`` (magnitude swept along the ray at
    ``beta_phase_deg``) → **parity map** and readout. This is the ``displace → wait at the Kerr point
    → Wigner`` protocol: one radial cut through phase space at a fixed ray angle. The companion
    ``WignerTomography2D`` host-iterates the ray angle to tile the whole plane (each point then
    carries its own phase — the fix for the old real-axis-only 2-D bug).

    The tomography ``beta`` sweep is a PHYSICAL amplitude routed through the calibrated
    ``displacement_cal`` gain-per-|alpha| (D23), so ``|beta|`` on hardware matches the swept value;
    ``beta_phase_deg`` sets the displacement phase so off-real-axis (complex) beta is reachable.
    Fitter: manual -- reports extrema of the displaced-parity signal (no curve fit); full
    density-matrix reconstruction is the reconstruction layer's job.

    Derived quantities:
        w_max : maximum value of the displaced-parity signal.
        w_min : minimum value of the displaced-parity signal.

    Parameters:
        betas         : displacement-magnitude sweep (``Sweep``); defaults to 0--beta_range on axis
                        ``"beta_re"``. Use a signed span (e.g. -beta_range..beta_range) for a full
                        line through the origin at ``beta_phase_deg``.
        points        : number of sweep points along the ray.
        mode_name     : cavity mode attribute name on the device (default: the device's cavity).
        beta_range    : upper bound of the default displacement sweep.
        beta_phase_deg: phase (deg) of the tomography displacement ray. 0 = real axis (the prior
                        behaviour); 90 = imaginary axis. This is what makes complex beta reachable.
        init_alpha    : coherent-state prep amplitude (calibrated ``displace_alpha``); 0 = vacuum
                        (no prep pulse — the prior behaviour).
        kerr_wait_us  : fixed free-evolution wait (µs) under the self-Kerr after prep; 0 = none.
        grid_size     : number of grid points (kept for the 2-D wrapper's convenience).

    Usage::

        # a cat at the Kerr point: prep |alpha=2>, wait, then a real-axis Wigner cut
        exp = WignerTomography(dev, betas=Sweep(-3, 3, on="beta_re"), points=61,
                               init_alpha=2.0, kerr_wait_us=5.0)
        outcome = exp.go(soc, reps=300)
        outcome.fit.derived["w_max"]
    """

    def __init__(self, device, *, betas: Sweep | None = None, points: int = 21,
                 mode_name: str | None = None, beta_range: float = 0.25,
                 beta_phase_deg: float = 0.0, init_alpha: float = 0.0,
                 kerr_wait_us: float = 0.0, grid_size: int = 21, prep_fock: int | None = None,
                 readout: DispersiveIQ | None = None, runner=None):
        super().__init__(device, runner=runner)
        mode_name = resolve_mode(device, mode_name, purpose=type(self).__name__)
        self.betas = betas if betas is not None else Sweep(start=0.0, stop=beta_range, on="beta_re")
        self.points = points
        self.mode_name = mode_name
        self.beta_range = beta_range
        self.beta_phase_deg = beta_phase_deg
        self.init_alpha = init_alpha
        self.kerr_wait_us = kerr_wait_us
        self.grid_size = grid_size
        #: Prepare a Fock state |prep_fock> (via the analytic ge/ef/f0g1 ladder) instead of a
        #: coherent |init_alpha>. ``None`` = coherent path (init_alpha). A Fock prep is the
        #: non-classical test: Fock |1> has W(0) = -2/pi (a genuine negative dip), unlike any
        #: coherent/vacuum state. Uses ``prepare(dev, Fock(n))``, which needs the f0g1 sideband
        #: calibration and (for this device) puts the photon in the cavity mode.
        self.prep_fock = prep_fock
        self.readout = readout if readout is not None else DispersiveIQ.from_calib(device.calib.readout)

    def axes(self) -> tuple:
        if isinstance(self.betas, Sweep) and self.betas.on is not None:
            return (Loop(self.betas.on, self.points),)
        return ()

    def sequence(self) -> tuple[Seq, tuple]:
        dev = self.device
        mode = getattr(dev, self.mode_name)
        # betas is a PHYSICAL displacement amplitude beta (dimensionless,
        # ~0-a few), not a lab-native int code. mode.displace(gain=...) treats any Sweep as the
        # latter (D23) -- passing betas directly resolved to a fraction of ~betas/32766 (~1e-5
        # for the 0-0.25 default), i.e. no real displacement on hardware, while SimulatorSoc's
        # response model reads betas as beta directly and never exercised the mismatch. Route
        # through the calibrated gain-per-alpha (the same displacement_cal ModeOps.displace_alpha
        # already uses) so the gain sweep is in the units mode.displace() actually expects.
        mode_cal = dev.calib.modes.get(self.mode_name)
        if mode_cal is None or mode_cal.displacement_cal is None:
            raise KeyError(
                f"mode {self.mode_name!r} has no displacement_cal — "
                "run DisplacementCalibration to calibrate the gain-per-alpha"
            )
        slope = mode_cal.displacement_cal
        gains = self.betas.mapped(lambda beta: slope * beta, inverse=lambda gain: gain / slope)

        seq = Seq()
        # 1. State prep. prep_fock -> a Fock |n> via the analytic ladder (ge_pi·ef_pi·f0g1 per
        #    photon), the non-classical test; else displace to a coherent |init_alpha> (init_alpha==0
        #    leaves the vacuum, keeping seq.ops[0] the tomography displacement for the gain-unit tests).
        if self.prep_fock is not None:
            seq = seq + prepare(dev, Fock(self.prep_fock))
        elif self.init_alpha != 0.0:
            seq.play(mode.displace_alpha(self.init_alpha))
        # 2. Free evolution under the mode self-Kerr for a fixed wait ("the Kerr point"). No Kerr
        #    coefficient is needed -- the state simply evolves; the wait is the knob.
        if self.kerr_wait_us > 0.0:
            seq.wait(self.kerr_wait_us)
        # 3. Tomography displacement (|beta| swept along the ray at beta_phase_deg), then parity-map.
        # #100: parity() drives the readout and measures internally by default now -- chaining
        # our own .measure() here would double-measure.
        seq.play(mode.displace(gain=gains, phase=self.beta_phase_deg))
        seq = seq + parity(dev, self.mode_name)
        return seq, ()

    def analyze(self, result: MeasurementResult) -> FitResult:
        signal = readout_signal(result, self.readout)
        w_max = float(np.max(signal))
        w_min = float(np.min(signal))
        return FitResult(
            params={},
            stderr={},
            derived={"w_max": w_max, "w_min": w_min},
            success=True,
        )


class WignerTomography2D:
    """Full phase-space Wigner sampling via a polar star of radial cuts.

    Fixes the earlier bug where the 2-D driver reran the real axis and dropped Im(β): the host
    iterates the ray angle over ``[0, 180)`` and each ray is a :class:`WignerTomography`
    **signed-magnitude** sweep, so a ray at angle ``θ`` with signed magnitude ``r`` samples the
    complex point ``r·e^{iθ}`` — both half-lines, tiling the whole plane. Every sampled point
    therefore carries its own phase (the fix), and the state prep + Kerr wait are applied once per
    ray before the tomography displacement, so the same prepared, Kerr-evolved state is sampled
    across the star.

    ``run()`` returns a :class:`MeasurementResult` whose ``meta`` carries the scattered complex
    displacement points (``alpha_re``/``alpha_im``), the projected parity signal at each
    (``parity_signal``, for the plot), and the **raw averaged IQ** per point (``parity_i``/
    ``parity_q``) — the reconstruction interpolates that against the readout's g/e blob centres to
    recover ``P_e`` before inversion (the hardware-path normalization; averaged rotated-I alone is
    in arbitrary units, not a population).
    """

    def __init__(
        self,
        device,
        soc,
        *,
        beta_max: float = 3.0,
        n_radial: int = 21,
        n_angle: int = 12,
        init_alpha: float = 0.0,
        kerr_wait_us: float = 0.0,
        mode_name: str | None = None,
        reps: int = 300,
        prep_fock: int | None = None,
        readout: DispersiveIQ | None = None,
        runner=None,
    ) -> None:
        self.device = device
        self.soc = soc
        self.beta_max = beta_max
        self.n_radial = n_radial
        self.n_angle = n_angle
        self.init_alpha = init_alpha
        self.kerr_wait_us = kerr_wait_us
        # A device-less star (sample-point geometry only) has no mode to resolve.
        self.mode_name = (
            mode_name if device is None
            else resolve_mode(device, mode_name, purpose=type(self).__name__)
        )
        self.reps = reps
        #: forwarded to each per-ray WignerTomography — prep a Fock |prep_fock> instead of a
        #: coherent |init_alpha> (the non-classical test; None = coherent path).
        self.prep_fock = prep_fock
        # the per-ray experiments read with this; a device-less star (sample_points only) has none
        if readout is None and device is not None:
            readout = DispersiveIQ.from_calib(device.calib.readout)
        self.readout = readout
        # The per-ray experiments run through this runner. ``None`` uses the base ``Runner`` (which
        # compiles against ``soc`` itself — valid when ``soc`` is a soccfg, i.e. mock/sim). On real
        # hardware ``soc`` is a Pyro proxy that is NOT a compile-time soccfg, so the harness injects
        # a ``LegacyPyroRunner(soccfg)`` here to split compile (against the QickConfig) from acquire
        # (against the proxy) — the D14 compile/acquire split. Without it, StrumentoProgram would try
        # to subscript the proxy as a soccfg and raise 'Proxy' object is not subscriptable.
        self.runner = runner

    def _angles_deg(self) -> np.ndarray:
        # [0, 180): a signed-magnitude radial sweep at θ also covers θ+180, so half the circle
        # tiles the whole plane without double-sampling the origin's opposite ray.
        return np.linspace(0.0, 180.0, self.n_angle, endpoint=False)

    def _radial_magnitudes(self) -> np.ndarray:
        return np.linspace(-self.beta_max, self.beta_max, self.n_radial)

    def sample_points(self) -> np.ndarray:
        """The Wigner phase-space points ``{α}`` the star samples (pure -- no acquire).

        Displaced-parity convention: sampling ``W(α)`` applies the displacement ``D(-α)`` before the
        parity map, so the point sampled is the **negative** of the applied displacement. These are
        the ``α`` the reconstruction/plot are indexed by, not the raw applied displacements.
        """
        mags = self._radial_magnitudes()
        return np.concatenate([
            -mags * np.exp(1j * np.radians(theta)) for theta in self._angles_deg()
        ])

    def run(self, *, progress: bool = False) -> MeasurementResult:
        """Acquire the whole polar star: one radial :class:`WignerTomography` sweep per ray angle.

        Every point carries its own phase (``β = r·e^{iθ}``), unlike the earlier real-axis-only
        driver — the scattered ``(α, parity)`` pairs land in ``meta`` for reconstruction.

        **Sign convention.** A ray applies the displacement ``D(+β)`` then the parity map, which
        samples the Wigner function at ``α = -β`` (displaced-parity measures ``W(α)`` by displacing
        ``D(-α)``). So ``meta["alpha_re"]``/``["alpha_im"]`` store ``-β`` — the actual phase-space
        point — so a coherent state prepared at ``+α₀`` reconstructs and plots at ``+α₀``, not its
        parity-reflection ``-α₀``.
        """
        mags = self._radial_magnitudes()
        alpha_re: list[float] = []
        alpha_im: list[float] = []
        signal: list[float] = []
        parity_i: list[float] = []
        parity_q: list[float] = []
        for theta in self._angles_deg():
            exp = WignerTomography(
                self.device,
                betas=Sweep(start=float(mags[0]), stop=float(mags[-1]), on="beta_re"),
                points=self.n_radial,
                mode_name=self.mode_name,
                beta_phase_deg=float(theta),
                init_alpha=self.init_alpha,
                kerr_wait_us=self.kerr_wait_us,
                prep_fock=self.prep_fock,
                readout=self.readout,
                runner=self.runner,
            )
            out = exp.go(self.soc, reps=self.reps, progress=progress)
            sig = np.asarray(readout_signal(out.result, self.readout), dtype=float).ravel()
            # Raw averaged (I, Q) per point — the reconstruction anchors these to the blob centres
            # (to_population) instead of trusting the projected scalar as a population.
            raw_i, raw_q = readout_iq(out.result)
            raw_i = np.asarray(raw_i, dtype=float).ravel()
            raw_q = np.asarray(raw_q, dtype=float).ravel()
            phase = np.radians(theta)
            for r, s, ii, qq in zip(mags, sig[: len(mags)], raw_i[: len(mags)], raw_q[: len(mags)]):
                # α = -β (displaced-parity: applying D(+β) samples W at -β) — see the docstring.
                alpha_re.append(float(-r * np.cos(phase)))
                alpha_im.append(float(-r * np.sin(phase)))
                signal.append(float(s))
                parity_i.append(float(ii))
                parity_q.append(float(qq))

        signal_arr = np.asarray(signal, dtype=float)
        return MeasurementResult(
            xpts=np.asarray(alpha_re, dtype=float),
            avgi=signal_arr,
            avgq=np.zeros_like(signal_arr),
            meta={
                "alpha_re": alpha_re,
                "alpha_im": alpha_im,
                "parity_signal": signal,
                "parity_i": parity_i,
                "parity_q": parity_q,
                "mode_name": self.mode_name,
                "init_alpha": self.init_alpha,
                "kerr_wait_us": self.kerr_wait_us,
            },
        )
