"""Safety guard tests for the cqed pack.

Verifies that the gain validators catch out-of-range pulses at the factory boundary,
and that calibrated values (which are within range) pass through cleanly.

Pure: no qick needed.
"""

from __future__ import annotations

import pytest

from strumento.core.device import Device
from strumento.core.sweeps import Sweep
from strumento.packs.cqed.experiments.t1 import T1
from strumento.packs.cqed.experiments.t2_echo import T2Echo
from strumento.packs.cqed.safety import (
    GainOutOfRange,
    SweepOutOfRange,
    validate_gain,
    validate_gain_sweep,
    validate_sweep_range,
)
from tests._paths import MULTIMODE_YAML


class TestValidateGain:
    """Unit tests for validate_gain."""

    def test_within_range_passes(self):
        assert validate_gain(0.25, limit=0.5, role="qubit_drive") == 0.25

    def test_at_limit_passes(self):
        assert validate_gain(0.5, limit=0.5, role="qubit_drive") == 0.5

    def test_above_limit_raises(self):
        with pytest.raises(GainOutOfRange, match="exceeds safe limit 0.5"):
            validate_gain(0.6, limit=0.5, role="qubit_drive", label="pi_ge")

    def test_negative_gain_checked_by_absolute_value(self):
        with pytest.raises(GainOutOfRange):
            validate_gain(-0.6, limit=0.5, role="test")

    def test_zero_gain_passes(self):
        assert validate_gain(0.0, limit=0.5, role="test") == 0.0


class TestValidateGainSweep:
    """Unit tests for validate_gain_sweep."""

    def test_within_range_passes(self):
        sweep = Sweep(start=0, stop=10000, on="amp")
        validate_gain_sweep(sweep, limit=0.5, maxv=32766, role="qubit_drive")  # no raise

    def test_exceeding_range_raises(self):
        sweep = Sweep(start=0, stop=20000, on="amp")
        with pytest.raises(GainOutOfRange, match="exceeds safe limit 0.4"):
            validate_gain_sweep(sweep, limit=0.4, maxv=32766, role="displacement")

    def test_zero_endpoint_is_exempt_from_the_domain_guard(self):
        """#78: a zero bound is a legitimate int-code "off", never a fraction-domain mistake
        — Sweep(start=0, stop=..., on=...) must not trip the domain guard on its zero side."""
        sweep = Sweep(start=0.0, stop=10000, on="amp")
        validate_gain_sweep(sweep, limit=0.5, maxv=32766, role="qubit_drive")  # no raise

    def test_fraction_domain_value_raises_even_when_naive_fraction_looks_safe(self):
        """#78's sharper half — the safety blind spot. BEFORE the domain guard existed, this
        function computed ``frac = abs(val) / maxv`` treating a fraction-domain-confused
        endpoint as if it were already an int code — for ``30000/32767`` (~0.9156, which
        *does* exceed the 0.5 qubit_drive ceiling) that computed a trivially-"safe" naive
        fraction of ~2.79e-5, letting the caller's actual-unsafe intended fraction slip past
        the ceiling check entirely. The two assertions below pin that pre-fix arithmetic down
        precisely; the guard must catch the domain confusion itself, before the misleading
        division ever runs, regardless of what it would have computed."""
        confused_fraction = 30000.0 / 32767.0
        limit = 0.5
        naive_frac_pre_fix = abs(confused_fraction) / 32766  # what the un-guarded code computed
        assert naive_frac_pre_fix <= limit, "the naive (mis-divided) fraction must look deceptively safe"
        assert confused_fraction > limit, "the caller's real intended fraction is actually unsafe"

        sweep = Sweep(start=0.0, stop=confused_fraction, on="amp")
        with pytest.raises(GainOutOfRange, match="qubit_drive"):
            validate_gain_sweep(sweep, limit=limit, maxv=32766, role="qubit_drive")


class TestValidateSweepRange:
    """Unit tests for validate_sweep_range."""

    def test_within_range_passes(self):
        sweep = Sweep(start=0.0, stop=100.0, on="t")
        validate_sweep_range(sweep, min_val=0.0, max_val=50000.0, label="delay")  # no raise

    def test_below_minimum_raises(self):
        sweep = Sweep(start=-1.0, stop=100.0, on="t")
        with pytest.raises(SweepOutOfRange, match="below minimum 0.0"):
            validate_sweep_range(sweep, min_val=0.0, label="delay")

    def test_above_maximum_raises(self):
        sweep = Sweep(start=0.0, stop=100000.0, on="t")
        with pytest.raises(SweepOutOfRange, match="exceeds maximum 50000.0"):
            validate_sweep_range(sweep, max_val=50000.0, label="delay")


class TestTimeSweepIntegration:
    """Regression: MAX_TIME_SWEEP/MIN_TIME_SWEEP existed but nothing called
    validate_sweep_range with them -- experiments with a swept delay (T1, T2Ramsey, T2Echo,
    CavityT1, CavityT2Ramsey, CrossKerr, RevivalTimeCalibration, StorageT1, ValidationCheck)
    had no guard against an unphysical time range at all."""

    def test_default_delays_pass(self):
        from strumento.core.device import Device
        from strumento.packs.cqed.experiments.t1 import T1
        dev = Device.load(MULTIMODE_YAML)
        T1(dev)  # should not raise — the default (0-500 us) is well within range

    def test_excessive_delay_span_raises(self):
        from strumento.core.device import Device
        from strumento.packs.cqed.experiments.t1 import T1
        dev = Device.load(MULTIMODE_YAML)
        with pytest.raises(SweepOutOfRange, match="exceeds maximum"):
            T1(dev, delays=Sweep(start=0.0, stop=60_000.0, on="t"))  # 60 ms, absurd for a T1

    @pytest.mark.parametrize("experiment_cls", [T1, T2Echo])
    def test_negative_delay_start_raises(self, experiment_cls):
        dev = Device.load(MULTIMODE_YAML)
        with pytest.raises(SweepOutOfRange, match="below minimum"):
            experiment_cls(
                dev, delays=Sweep(start=-10.0, stop=100.0, on="t")
            )  # negative delay, unphysical
    """Integration: the factories reject dangerous gains but pass calibrated ones."""

    def test_calibrated_pi_passes(self, dev):
        """The calibrated pi gain (0.25 frac) is well within the 0.5 limit."""
        pulse = dev.qubit.ge_pi()
        assert pulse.gain == pytest.approx(8192 / 32766, rel=1e-3)

    def test_explicit_high_gain_raises(self, dev):
        """An explicit gain above the limit is rejected."""
        with pytest.raises(GainOutOfRange, match="qubit_drive"):
            dev.qubit.ge_pi(gain=0.7)

    def test_calibrated_displacement_passes(self, dev):
        """The calibrated displacement gain (0.18 frac) passes the 0.4 limit."""
        pulse = dev.cavity.displace()
        assert pulse.gain == pytest.approx(6000 / 32766, rel=1e-3)

    def test_sweep_gain_within_limit_passes_factory(self, dev):
        """A gain sweep safely within the qubit_drive limit passes. Also #78: the zero-start
        endpoint is a completely normal int-code "off" and must not false-positive as a
        fraction-domain value under the new domain guard — it must convert unchanged."""
        pulse = dev.qubit.ge_pi(gain=Sweep(start=0, stop=16000, on="amp"))
        # Should not raise — 16000/32766 ~= 0.488, under the 0.5 limit
        assert hasattr(pulse.gain, "start")
        assert pulse.gain.start == 0.0
        assert pulse.gain.stop == pytest.approx(16000 / 32766)

    def test_sweep_gain_exceeding_limit_raises_at_factory(self, dev):
        """Regression: a gain sweep is validated at its raw endpoints against the
        same limit a scalar gain faces — it is not exempt from the safety ceiling just because
        it's a sweep. Previously this silently passed (the exact hole the audit found): a
        default AmplitudeRabi/EFRabi/SidebandRabi/PowerSweep sweep (0->30000, fraction ~0.916)
        blew past every declared role limit with no guard."""
        with pytest.raises(GainOutOfRange, match="qubit_drive"):
            dev.qubit.ge_pi(gain=Sweep(start=0, stop=30000, on="amp"))

    def test_fraction_domain_gain_sweep_raises_instead_of_collapsing(self, dev):
        """Bug #78: a gain Sweep endpoint that is already a v2 fraction (e.g. the caller did
        their own ``30000/32767`` division) used to sail through unflagged, then get divided
        by ``maxv`` a SECOND time in ``_gain.gain_out``/``Sweep.mapped`` — the DAC gain
        silently collapsed toward zero (``2.79e-5``) while the reported/plotted axis still
        showed the caller's original, plausible-looking value (``0.9156``, via
        ``_lab_inverse``). Sweep gain is always lab-native int-code domain (D23); a nonzero
        endpoint under 1.0 cannot plausibly be a real int code (every calibrated gain in this
        pack sits in the thousands — pi_ge=8192, pi_ef=7600, ...), so it must raise loudly
        instead of silently mis-converting."""
        with pytest.raises(GainOutOfRange, match="qubit_drive"):
            dev.qubit.ge_pi(gain=Sweep(start=0.0, stop=30000.0 / 32767.0, on="amp"))

    def test_fraction_domain_gain_sweep_raises_via_shared_choke_point(self, dev):
        """#78's fix lives in validate_gain_sweep — the single choke point TransmonOps,
        SidebandOps, ModeOps, CouplerOps and ReadoutOps all call before .mapped()/gain_out().
        Prove a non-Transmon caller (ModeOps' displacement drive) is protected too, not just
        the ge_pi path the issue was reproduced against directly."""
        with pytest.raises(GainOutOfRange, match="displacement"):
            dev.cavity.displace(gain=Sweep(start=0.0, stop=0.35, on="amp"))
