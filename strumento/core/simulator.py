"""``SimulatorSoc`` — real tProc-v2 assembly, simulated acquire (D32; realizes D29).

D29 named a ``SimulatorSoc`` ("the same acquire protocol against a physics simulation") and left
its mechanism open. This is that mechanism, split exactly on the qick boundary:

* :mod:`strumento.core.response` (**pure**) owns the response **registry**, the arming payload
  (:class:`~strumento.core.response.SimContext`) and the reusable simulation machinery.
* this module (**qick-facing**, joining ``compiler`` / ``program`` / ``mock`` in hard rule 1) owns
  the soc: a real :class:`~qick.QickConfig` built from a tProc-**v2** snapshot, reusing the mock's
  stubbed hardware surface so ``declare_gen`` / ``add_gauss`` / ``add_pulse`` and the whole v2
  assembler genuinely run — only ``poll_data`` is replaced, by data generated from the armed
  response context.

So a simulated experiment exercises **the real path**: the same IR, the same compiler, the same
``AveragerProgramV2``, the same reshape into a :class:`~strumento.core.result.MeasurementResult`.
The only fiction is the IQ the board would have returned.

**Ground truth is data.** ``SimulatorSoc(truth=...)`` takes an ordinary calibration dict (hard
rule 4 / D10) and hands it to the response model through the context, *separately* from the
device's calibrated belief — a perturbed truth is what makes a closed-loop calibration demo
non-trivial. No physics constant lives in this module: what the populations *are* comes from a
pack-owned response model (``strumento/packs/cqed/responses.py`` for cqed).

**Emission is per-shot.** Populations become per-shot draws placed at the readout's IQ blobs, so
the accumulated buffer the v2 acquire loop averages is genuinely shot-resolved and the per-shot
``shots`` / ``heralds`` / ``keep_mask`` post-selection path (and ``DispersiveIQ.discriminate``)
runs on simulated data exactly as it runs on the board's. The shots stay reachable via
:meth:`SimulatorSoc.collect_shots` (the :class:`~strumento.core.soc.ShotSource` capability).

**Escape hatch (hard rule 5).** Nothing here walls off raw qick. The soc *is* a ``QickConfig``, so
a hand-written ``AveragerProgramV2`` runs against it; arm it with an explicit model
(``SimContext(experiment=..., model=fn)``, or ``SimulatorSoc(model=fn)`` for a soc-level default)
and any program at all gets data. What is *not* provided is silent zeros — those remain the
:class:`~strumento.core.mock.MockQickSocV2` contract, which this class leaves untouched.

**Buffer layout.** ``poll_data`` fills the accumulated buffer in the default v2 loop order — the
``reps`` loop outermost, the swept loop inside it (``AveragerProgramV2(reps_innermost=False)``,
which is what :class:`~strumento.core.program.StrumentoProgram` builds). A program that inverts
that order would scramble the sweep axis; the layout assumption is stated here rather than
guessed at, because ``start_readout`` reports only the flat count.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from strumento.core.mock import MockQickSocV2
from strumento.core.response import (
    ResponseError,
    ResponseModel,
    SimContext,
    SimResponse,
    place_shots,
    populations_to_iq,
    response_for,
)

#: tProc-v2 firmware reports this processor type in a ``dump_cfg()`` snapshot (v1 is
#: ``axis_tproc64x32_x8``, which ``AveragerProgramV2`` cannot target at all).
_V2_TPROC_TYPE = "qick_processor"

#: ADC counts per unit of response signal. The accumulated buffer is ``int64``, so the response
#: (populations in [0, 1], transmission of order 1) is scaled before rounding; the acquire path's
#: own normalization (divide by the readout window, subtract the readout's IQ offset) makes the
#: returned average an affine image of the response, which is all any fitter here needs.
DEFAULT_ADC_SCALE = 1.0e4

#: seed for the per-shot draws when none is given — every stochastic step goes through this one
#: generator, so a pinned seed pins the data (no module-level random state).
DEFAULT_SEED = 1234


class NonV2SnapshotError(ValueError):
    """The soccfg snapshot is not a tProc-v2 config, so no v2 program can be assembled on it."""


def _require_v2(soccfg_dict: dict) -> None:
    """Reject a non-v2 snapshot at construction, naming the cqed exemplar (``multimode_demo``)."""
    tprocs = soccfg_dict.get("tprocs") or []
    kind = tprocs[0].get("type") if tprocs else None
    if kind != _V2_TPROC_TYPE:
        raise NonV2SnapshotError(
            f"device snapshot is not tProc v2 (tprocs[0].type={kind!r}, need "
            f"{_V2_TPROC_TYPE!r}): SimulatorSoc assembles a real AveragerProgramV2, which a v1 "
            "snapshot cannot target. Use a device with a v2 snapshot, e.g. the cqed exemplar "
            "devices/multimode_demo/device.yaml (devices/_fixtures/soccfg_v2_cqed.json)."
        )


def _as_response(value) -> SimResponse:
    """Coerce a model's return into a :class:`SimResponse` (bare arrays take the defaults)."""
    if isinstance(value, SimResponse):
        return value
    arr = np.asarray(value)
    if np.iscomplexobj(arr):
        return SimResponse(iq=arr)
    return SimResponse(populations=arr)


class SimulatorSoc(MockQickSocV2):
    """A real v2 ``QickConfig`` whose ``poll_data`` returns the armed response's per-shot data."""

    def __init__(
        self,
        soccfg_dict: dict,
        *,
        truth: dict | None = None,
        model=None,
        seed: int = DEFAULT_SEED,
        adc_scale: float = DEFAULT_ADC_SCALE,
    ) -> None:
        """
        Args:
            soccfg_dict: a tProc-v2 ``dump_cfg()`` snapshot (a v1 snapshot is a typed error).
            truth: the ground-truth calibration dict handed to the response model (see the module
                docstring). Plain data — this module never reads a key of it.
            model: a soc-level default response model, used when an armed context supplies none.
                This is the raw-program escape hatch.
            seed: seeds the one :class:`numpy.random.Generator` every draw comes from.
            adc_scale: counts per unit of response signal (see :data:`DEFAULT_ADC_SCALE`).
        """
        _require_v2(soccfg_dict)
        super().__init__(soccfg_dict)
        self.truth: dict = dict(truth or {})
        self.default_model = model
        self.adc_scale = float(adc_scale)
        self.seed = int(seed)
        self.rng = np.random.default_rng(seed)
        self._ctx: SimContext | None = None
        self._model: ResponseModel | None = None
        self._sim_shots: list | None = None

    # -- construction ------------------------------------------------------------------

    @classmethod
    def from_snapshot(cls, path: str | Path, **kwargs) -> SimulatorSoc:
        """Build from a committed ``dump_cfg()`` JSON snapshot (no board); must be tProc v2."""
        return cls(json.loads(Path(path).read_text()), **kwargs)

    @classmethod
    def from_device(cls, device, **kwargs) -> SimulatorSoc:
        """Build from a loaded device instance's own snapshot — the usual offline-run entry point."""
        snapshot = getattr(device, "soccfg_snapshot", None)
        if not snapshot:
            raise NonV2SnapshotError(
                "this device instance carries no soccfg snapshot, so no program can be assembled "
                "against it; use a device with a v2 snapshot, e.g. the cqed exemplar "
                "devices/multimode_demo/device.yaml"
            )
        return cls(snapshot, **kwargs)

    # -- the arming seam (ArmableSoc) ---------------------------------------------------

    def arm(self, ctx: SimContext) -> None:
        """Accept the run context and resolve its response model (D32 resolution order).

        Resolution happens **here**, not at ``poll_data``, so "no response model registered for
        ``Cls``" surfaces from the experiment path before a program is even run. The soc's own
        ``truth`` / default ``model`` fill in the context's empty slots; a context that carries its
        own truth or model wins (an explicit override is an override).
        """
        if not isinstance(ctx, SimContext):
            raise TypeError(f"arm() takes a SimContext, got {type(ctx).__name__}")
        if not ctx.truth and self.truth:
            ctx = replace(ctx, truth=dict(self.truth))
        if ctx.model is None and self.default_model is not None:
            ctx = replace(ctx, model=self.default_model)
        self._model = response_for(ctx)  # typed ResponseError when nothing is registered
        self._ctx = ctx
        self._sim_shots = None

    @property
    def context(self) -> SimContext | None:
        """The armed context (``None`` until :meth:`arm` is called)."""
        return self._ctx

    @property
    def armed(self) -> bool:
        return self._ctx is not None and self._model is not None

    # -- the environment seam (EnvironmentReporter, D34) ---------------------------------

    def environment(self) -> dict:
        """A **synthetic** environment stamp: simulated data says so, in its own provenance.

        The one non-obvious choice: the truth's *keys* are recorded, never its values. Recording
        the values would put the answer a closed-loop demo is supposed to *measure* into the
        result's provenance, one dict lookup away from any agent driving the session.
        """
        return {
            "kind": "simulator",
            "simulated": True,
            "seed": self.seed,
            "adc_scale": self.adc_scale,
            "truth_keys": sorted(str(key) for key in self.truth),
            "response_model": getattr(self._model, "__name__", None) if self.armed else None,
        }

    # -- the data path (ShotSource + poll_data) -----------------------------------------

    def collect_shots(self, ch: int = 0):
        """Per-shot IQ behind the last ``poll_data``, or ``None`` before any acquire.

        Shape follows the ``MeasurementResult.shots`` convention — leading axes index sweep points
        (dropped when the program has no sweep), then the reads in a shot (dropped when there is
        one), then the shot axis, then ``(I, Q)``. With ``rounds > 1`` this is the last round's
        draws (each round is an independent draw, as on hardware).
        """
        if not self._sim_shots:
            return None
        return self._sim_shots[ch]

    def poll_data(self):
        """Generate the acquisition data from the armed response (replaces the mock's zeros).

        Returns the ``[(new_points, (d, stats))]`` chunk shape ``AveragerProgramV2.acquire``
        consumes, with ``d[ii]`` int64 of shape ``(total_count * reads_per_shot[ii], 2)`` laid out
        rep-major (see the module docstring).
        """
        if not self.armed:
            raise ResponseError(
                "SimulatorSoc has not been armed: the Runner arms an ArmableSoc with a "
                "SimContext before acquire. For a raw qick program, arm it yourself "
                "(soc.arm(SimContext(experiment=..., model=fn))) or pass SimulatorSoc(model=fn); "
                "for zero-valued data use MockQickSocV2."
            )
        resp = _as_response(self._model(self._ctx))
        n_pts = len(self._ctx.xpts) or 1
        total = int(self._total_count)
        if total % n_pts:
            raise ResponseError(
                f"the armed sweep axis ({n_pts} points) does not divide the acquisition count "
                f"({total}): the simulator maps the flat count onto reps x sweep points, so the "
                f"armed xpts must be the program's swept axis"
            )
        shots = total // n_pts

        chunks: list[np.ndarray] = []
        per_channel: list[np.ndarray] = []
        for nreads in self._reads_per_shot:
            iq = self._draw(resp, n_pts, int(nreads), shots)  # (n_pts, nreads, shots, 2)
            per_channel.append(self._presented(iq, int(nreads)))
            flat = np.moveaxis(iq, -2, 0).reshape(-1, 2)  # rep-major: (shot, pt, read) -> flat
            chunks.append(np.rint(flat * self.adc_scale).astype(np.int64))

        self._sim_shots = per_channel
        return [(total, (chunks, None))]

    # -- internals ---------------------------------------------------------------------

    def _draw(self, resp: SimResponse, n_pts: int, nreads: int, shots: int) -> np.ndarray:
        """Per-shot IQ of shape ``(n_pts, nreads, shots, 2)`` for one readout channel."""
        if resp.machine is not None:
            draws = self._walk(resp.machine, n_pts, nreads, shots)
            return place_shots(draws, resp.blob0, resp.blob1, float(resp.sigma), self.rng)
        if resp.populations is not None:
            pops = self._grid(resp.populations, n_pts, nreads)
            return populations_to_iq(
                pops, resp.blob0, resp.blob1, float(resp.sigma), shots, self.rng
            )
        mean = self._grid(resp.iq, n_pts, nreads, dtype=complex)
        centers = np.stack([mean.real, mean.imag], axis=-1)[..., None, :]
        out = np.repeat(centers, shots, axis=-2)
        if resp.sigma:
            out = out + self.rng.normal(0.0, float(resp.sigma), size=out.shape)
        return out

    def _walk(self, machine, n_pts: int, nreads: int, shots: int) -> np.ndarray:
        """Per-shot outcomes ``(n_pts, nreads, shots)`` from walking the armed sequence.

        Every sweep point gets an independent set of shots: the machine models *control flow*, not
        the swept parameter, so its outcome distribution is the same at every point (a response
        whose populations depend on the sweep returns ``populations=`` instead — that is the choice
        between the two forms).
        """
        seq = getattr(self._ctx, "seq", None)
        if seq is None:
            raise ResponseError(
                "a per-shot machine response needs the sequence being run: the Runner arms it "
                "(SimContext(..., seq=seq)); a hand-armed raw program must pass seq= itself"
            )
        rounds = []
        for _ in range(n_pts):
            out = machine.outcomes(seq, shots, self.rng)
            if out.shape[0] != nreads:
                raise ResponseError(
                    f"the sequence measures {out.shape[0]} time(s) per shot but the acquisition "
                    f"declared {nreads} reads per shot. A Measure inside a Repeat is the usual "
                    f"cause: qick counts trigger *macros* at assembly, so a hardware loop's read "
                    f"is counted once however many times the tProc performs it. Unroll the rounds "
                    f"in the sequence (the cqed active-reset composite does this by default) so "
                    f"the static read count matches the dynamic one."
                )
            rounds.append(out)
        return np.stack(rounds)

    @staticmethod
    def _grid(value, n_pts: int, nreads: int, *, dtype=float) -> np.ndarray:
        """Broadcast a response array to ``(n_pts, nreads)``, or raise a typed shape error."""
        arr = np.asarray(value, dtype=dtype)
        if arr.ndim == 0:
            return np.full((n_pts, nreads), arr)
        if arr.shape == (n_pts,):
            return np.repeat(arr[:, None], nreads, axis=1)
        if arr.shape == (n_pts, nreads):
            return arr
        raise ResponseError(
            f"response shape {arr.shape} does not match the acquisition: expected a scalar, "
            f"({n_pts},) over the swept axis, or ({n_pts}, {nreads}) over (sweep, reads-per-shot)"
        )

    def _presented(self, iq: np.ndarray, nreads: int) -> np.ndarray:
        """Squeeze the degenerate axes so ``collect_shots`` matches the per-shot ``shots`` slot."""
        assert self._ctx is not None  # only poll_data calls this, after its armed check
        out = iq if nreads > 1 else iq[:, 0]
        return out if self._ctx.xpts else out[0]
