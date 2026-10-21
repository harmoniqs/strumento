"""Response registry + simulation mechanism (D32) — pure layer, no qick.

The registry is the pack seam (physics is registered *into* core, never imported by it); the
mechanism helpers are the shared machinery every response model reuses: shot sampling, IQ blob
placement, and the per-shot conditional walk over ``Branch``/``Repeat``.
"""

from __future__ import annotations

import numpy as np
import pytest

from strumento.core.pulses import Seq
from strumento.core.response import (
    ResponseError,
    ShotMachine,
    SimContext,
    average_iq,
    populations_to_iq,
    register_response,
    resolve_response,
    response_for,
    sample_shots,
)

# -- registry -------------------------------------------------------------------------------


def test_registry_resolves_by_mro():
    class Base: ...

    class Sub(Base): ...

    register_response(Base, lambda ctx: [0.5])
    assert resolve_response(Sub) is not None
    assert resolve_response(Sub)(None) == [0.5]


def test_missing_response_is_typed_error():
    with pytest.raises(ResponseError, match="no response model registered"):
        resolve_response(type("Orphan", (), {}))


def test_typed_error_names_the_class():
    with pytest.raises(ResponseError, match="WidgetExperiment"):
        resolve_response(type("WidgetExperiment", (), {}))


def test_nearest_registration_wins():
    class Base: ...

    class Sub(Base): ...

    register_response(Base, lambda ctx: ["base"])
    register_response(Sub, lambda ctx: ["sub"])
    assert resolve_response(Sub)(None) == ["sub"]
    assert resolve_response(Base)(None) == ["base"]


def test_responds_decorator_registers():
    from strumento.core.response import responds

    class Deco: ...

    @responds(Deco)
    def _model(ctx):
        return [1.0]

    assert resolve_response(Deco) is _model


def test_resolve_accepts_an_instance():
    class Inst: ...

    register_response(Inst, lambda ctx: [0.25])
    assert resolve_response(Inst())(None) == [0.25]


# -- SimContext + resolution order ---------------------------------------------------------


def test_simcontext_is_frozen_and_pins_xpts():
    ctx = SimContext(experiment=object(), device=None, xpts=[1.0, 2.0])
    assert ctx.xpts == (1.0, 2.0)  # coerced to a tuple — immutable, hashable axis
    with pytest.raises(Exception):
        ctx.device = 1  # frozen


def test_explicit_model_beats_registry():
    class Exp: ...

    register_response(Exp, lambda ctx: ["registry"])
    ctx = SimContext(experiment=Exp(), device=None, xpts=(), model=lambda c: ["explicit"])
    assert response_for(ctx)(ctx) == ["explicit"]


def test_experiment_hook_beats_registry():
    class Exp:
        def response_model(self, device, xpts):
            return lambda ctx: ["hook"]

    register_response(Exp, lambda ctx: ["registry"])
    ctx = SimContext(experiment=Exp(), device=None, xpts=())
    assert response_for(ctx)(ctx) == ["hook"]


def test_registry_used_when_no_hook():
    class Exp: ...

    register_response(Exp, lambda ctx: ["registry"])
    ctx = SimContext(experiment=Exp(), device=None, xpts=())
    assert response_for(ctx)(ctx) == ["registry"]


def test_response_for_unregistered_is_typed_error():
    class Lonely: ...

    ctx = SimContext(experiment=Lonely(), device=None, xpts=())
    with pytest.raises(ResponseError, match="Lonely"):
        response_for(ctx)


# -- mechanism: shot sampling --------------------------------------------------------------


def test_sample_shots_shape_and_statistics():
    rng = np.random.default_rng(7)
    pops = np.array([0.0, 0.25, 1.0])
    draws = sample_shots(pops, shots=4000, rng=rng)
    assert draws.shape == (3, 4000)
    assert set(np.unique(draws)) <= {0, 1}
    means = draws.mean(axis=-1)
    assert means[0] == 0.0 and means[2] == 1.0
    assert abs(means[1] - 0.25) < 5 * np.sqrt(0.25 * 0.75 / 4000)


def test_sample_shots_is_deterministic_in_the_injected_rng():
    a = sample_shots([0.4], shots=64, rng=np.random.default_rng(3))
    b = sample_shots([0.4], shots=64, rng=np.random.default_rng(3))
    assert np.array_equal(a, b)


def test_sample_shots_scalar_population():
    draws = sample_shots(0.5, shots=32, rng=np.random.default_rng(1))
    assert draws.shape == (32,)


def test_sample_shots_rejects_out_of_range_population():
    with pytest.raises(ValueError, match="population"):
        sample_shots([1.5], shots=8, rng=np.random.default_rng(0))


# -- mechanism: IQ blob placement ----------------------------------------------------------


def test_populations_to_iq_shape_and_blob_means():
    rng = np.random.default_rng(11)
    shots = 4000
    sigma = 0.5
    iq = populations_to_iq(
        [0.0, 1.0], blob0=(1.0, -1.0), blob1=(4.0, 2.0), sigma=sigma, shots=shots, rng=rng
    )
    assert iq.shape == (2, shots, 2)  # (sweep, shot, IQ)
    tol = 5 * sigma / np.sqrt(shots)
    assert np.allclose(iq[0].mean(axis=0), (1.0, -1.0), atol=tol)
    assert np.allclose(iq[1].mean(axis=0), (4.0, 2.0), atol=tol)


def test_populations_to_iq_mixture_lands_between_the_blobs():
    rng = np.random.default_rng(12)
    shots = 8000
    iq = populations_to_iq(
        [0.5], blob0=(0.0, 0.0), blob1=(10.0, 0.0), sigma=0.3, shots=shots, rng=rng
    )
    mean_i = iq[0, :, 0].mean()
    assert abs(mean_i - 5.0) < 5 * 5.0 / np.sqrt(shots)


def test_populations_to_iq_deterministic_given_rng():
    kw = dict(blob0=(0.0, 0.0), blob1=(1.0, 1.0), sigma=0.1, shots=16)
    a = populations_to_iq([0.3], rng=np.random.default_rng(5), **kw)
    b = populations_to_iq([0.3], rng=np.random.default_rng(5), **kw)
    assert np.array_equal(a, b)


def test_average_iq_reduces_the_shot_axis():
    iq = np.zeros((3, 10, 2))
    iq[..., 0] = 2.0
    iq[..., 1] = -3.0
    avgi, avgq = average_iq(iq)
    assert avgi.shape == (3,) and avgq.shape == (3,)
    assert np.allclose(avgi, 2.0) and np.allclose(avgq, -3.0)


# -- mechanism: the per-shot conditional state machine -------------------------------------


def test_shot_machine_measures_the_prepared_state():
    m = ShotMachine(thermal_population=0.0)
    rec = m.run(Seq().measure(), np.random.default_rng(0))
    assert rec.measurements == (0,)
    assert rec.state == 0


def test_thermal_population_seeds_excited_shots():
    m = ShotMachine(thermal_population=1.0)
    rec = m.run(Seq().measure(), np.random.default_rng(0))
    assert rec.measurements == (1,) and rec.state == 1


def test_thermal_population_statistics():
    m = ShotMachine(thermal_population=0.2)
    rng = np.random.default_rng(4)
    recs = m.run_shots(Seq().measure(), shots=4000, rng=rng)
    frac = np.mean([r.state for r in recs])
    assert abs(frac - 0.2) < 5 * np.sqrt(0.2 * 0.8 / 4000)


def test_branch_runs_the_conditional_arm_and_flips_the_state():
    # measure → if the shot read |e> (value >= threshold) play a corrective pulse that flips it:
    # exactly the active-reset shape, with the flip supplied by the (pack-owned) physics hook.
    def flip(state, op, rng):
        return 1 - state

    seq = Seq().measure().branch(
        adc=0, threshold=0.5, then=_labelled_pulse_seq("pi"), test=">="
    )
    m = ShotMachine(thermal_population=1.0, apply_pulse=flip)
    rec = m.run(seq, np.random.default_rng(0))
    assert rec.measurements[0] == 1  # the herald read excited
    assert rec.state == 0            # the conditional arm reset it


def test_branch_else_arm_taken_when_the_test_fails():
    marks: list[str] = []

    def note(state, op, rng):
        marks.append(op.label)
        return state

    seq = Seq().measure().branch(
        adc=0,
        threshold=0.5,
        then=_labelled_pulse_seq("then"),
        else_=_labelled_pulse_seq("else"),
        test=">=",
    )
    m = ShotMachine(thermal_population=0.0, apply_pulse=note)
    m.run(seq, np.random.default_rng(0))
    assert marks == ["else"]


def test_repeat_runs_the_body_count_times():
    calls = []

    def count(state, op, rng):
        calls.append(op)
        return state

    seq = Seq().repeat(_labelled_pulse_seq("body"), 3)
    m = ShotMachine(apply_pulse=count)
    m.run(seq, np.random.default_rng(0))
    assert len(calls) == 3


def test_repeated_reset_rounds_accumulate_measurements():
    round_ = Seq().measure().branch(
        adc=0, threshold=0.5, then=_labelled_pulse_seq("pi"), test=">="
    )
    m = ShotMachine(thermal_population=1.0, apply_pulse=lambda s, op, rng: 1 - s)
    rec = m.run(Seq().repeat(round_, 2), np.random.default_rng(0))
    assert rec.measurements == (1, 0)  # round 1 caught |e>, round 2 confirms |g>


def test_branch_without_a_prior_measurement_is_typed_error():
    seq = Seq().branch(adc=0, threshold=0.5, then=_labelled_pulse_seq("then"))
    with pytest.raises(ResponseError, match="Branch"):
        ShotMachine().run(seq, np.random.default_rng(0))


def test_readout_error_hook_is_injectable():
    # a misreading readout: state 0 always reads 1
    m = ShotMachine(measure=lambda state, rng: 1)
    rec = m.run(Seq().measure(), np.random.default_rng(0))
    assert rec.state == 0 and rec.measurements == (1,)


# -- helpers --------------------------------------------------------------------------------


def _labelled_pulse_seq(label: str) -> Seq:
    from strumento.core.pulses import Const, Pulse
    from strumento.core.wiring import LineRef

    s = Seq()
    s.ops.append(
        Pulse(
            line=LineRef("qubit", "drive"),
            freq_mhz=1.0,
            gain=0.5,
            envelope=Const(length_us=0.01),
            label=label,
        )
    )
    return s


# -- purity ---------------------------------------------------------------------------------


def test_module_imports_without_qick():
    import subprocess
    import sys

    code = "import sys; sys.modules['qick']=None; import strumento.core.response; print('ok')"
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0 and "ok" in r.stdout, r.stderr
