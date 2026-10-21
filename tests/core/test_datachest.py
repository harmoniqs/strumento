"""DataChest tests — write-once storage, round-trip, manifest, seq provenance."""

from __future__ import annotations

import json

import numpy as np
import pytest

from strumento.core.datachest import FileDataChest, seq_to_wire
from strumento.core.result import MeasurementResult


@pytest.fixture
def chest(tmp_path):
    return FileDataChest(path=tmp_path / "data")


def _make_result(n=21) -> MeasurementResult:
    return MeasurementResult(
        xpts=np.linspace(0, 10, n),
        avgi=np.random.default_rng(42).normal(0, 1, (1, n)),
        avgq=np.zeros((1, n)),
        meta={"environment": {"source": "test"}},
    )


class TestFileDataChest:
    """FileDataChest: save, load, list, content-addressing."""

    def test_save_returns_id(self, chest):
        result = _make_result()
        entry_id = chest.save(result, experiment="spin_t1")
        assert isinstance(entry_id, str) and len(entry_id) == 12

    def test_round_trip(self, chest):
        result = _make_result()
        entry_id = chest.save(result, experiment="spin_rabi")
        loaded = chest.load(entry_id)

        assert np.allclose(loaded["result"].avgi, result.avgi)
        assert np.allclose(loaded["result"].avgq, result.avgq)
        assert np.allclose(loaded["result"].xpts, result.xpts)
        assert loaded["meta"]["environment"]["source"] == "test"

    def test_content_addressing_is_deterministic(self, chest):
        result = _make_result()
        id1 = chest.save(result, experiment="t1")
        id2 = chest.save(result, experiment="t1")
        assert id1 == id2  # same arrays → same hash

    def test_different_data_different_id(self, chest):
        r1 = _make_result(21)
        r2 = _make_result(31)
        id1 = chest.save(r1, experiment="t1")
        id2 = chest.save(r2, experiment="t1")
        assert id1 != id2

    def test_an_unsalted_id_is_the_hash_of_the_arrays_alone(self, chest):
        # Pins the content address so a salt can never silently change unsalted ids.
        import hashlib

        result = _make_result()
        hasher = hashlib.sha256()
        for key in ("avgi", "avgq", "xpts"):   # sorted array names, as save() hashes them
            hasher.update(np.asarray(getattr(result, key)).tobytes())
        assert chest.save(result, experiment="t1") == hasher.hexdigest()[:12]
        assert chest.save(result, experiment="t1", salt=None) == hasher.hexdigest()[:12]

    def test_a_salt_gives_identical_data_its_own_entry(self, chest):
        result = _make_result()
        plain = chest.save(result, experiment="t1")
        a = chest.save(result, experiment="t1", salt="run-a")
        b = chest.save(result, experiment="t1", salt="run-b")
        assert len({plain, a, b}) == 3, "each salt must address its own entry"
        assert chest.save(result, experiment="t1", salt="run-a") == a, "a salt is deterministic"
        for entry in (plain, a, b):
            assert np.allclose(chest.load(entry)["result"].avgi, result.avgi)

    def test_each_salted_entry_keeps_its_own_metadata(self, chest):
        a = chest.save(_make_result(), experiment="t1", salt="a",
                       fit={"params": {"tau": 1.0}, "stderr": {}, "derived": {}})
        b = chest.save(_make_result(), experiment="t1", salt="b",
                       fit={"params": {"tau": 2.0}, "stderr": {}, "derived": {}})
        assert chest.load(a)["fit"]["params"]["tau"] == 1.0
        assert chest.load(b)["fit"]["params"]["tau"] == 2.0

    def test_manifest_grows(self, chest):
        for i in range(3):
            chest.save(_make_result(10 + i), experiment=f"exp_{i}")
        entries = chest.list_entries()
        assert len(entries) == 3
        assert entries[0]["experiment"] == "exp_2"  # newest first

    def test_manifest_filters_by_experiment(self, chest):
        chest.save(_make_result(10), experiment="rabi")
        chest.save(_make_result(11), experiment="t1")
        chest.save(_make_result(12), experiment="rabi")
        entries = chest.list_entries(experiment="rabi")
        assert len(entries) == 2
        assert all(e["experiment"] == "rabi" for e in entries)

    def test_save_with_fit(self, chest):
        result = _make_result()
        fit = {"params": {"tau": 50.0}, "stderr": {"tau": 1.2}, "derived": {"T1": 50.0}}
        entry_id = chest.save(result, experiment="t1", fit=fit)
        loaded = chest.load(entry_id)
        assert loaded["fit"]["derived"]["T1"] == 50.0

    def test_save_with_shots(self, chest):
        rng = np.random.default_rng(7)
        result = MeasurementResult(
            xpts=np.linspace(0, 5, 11),
            avgi=rng.normal(0, 1, (1, 11)),
            avgq=np.zeros((1, 11)),
            shots=rng.normal(0, 1, (11, 100, 2)),
        )
        entry_id = chest.save(result, experiment="ssr")
        loaded = chest.load(entry_id)
        assert loaded["result"].shots.shape == (11, 100, 2)

    def test_save_with_heralds(self, chest):
        result = MeasurementResult(
            xpts=np.array([0, 1, 2]),
            avgi=np.array([[0.1, 0.5, 0.9]]),
            avgq=np.zeros((1, 3)),
            heralds={"reset": np.array([True, False, True])},
            herald_lanes=("reset",),
        )
        entry_id = chest.save(result, experiment="active_reset")
        loaded = chest.load(entry_id)
        assert "reset" in loaded["result"].heralds
        assert loaded["result"].heralds["reset"].dtype == bool

    def test_load_nonexistent_raises(self, chest):
        with pytest.raises(KeyError, match="not found"):
            chest.load("nonexistent1")

    def test_task_id_in_manifest(self, chest):
        chest.save(_make_result(), experiment="t1", task_id="bringup-001/steps/t1_003")
        entries = chest.list_entries()
        assert entries[0]["task_id"] == "bringup-001/steps/t1_003"


class TestSeqToWire:
    """seq_to_wire serializes Seq with all op types."""

    def test_basic_sequence(self):
        from strumento.core.pulses import Gaussian, Pulse, Seq
        from strumento.core.wiring import LineRef

        seq = Seq()
        seq.play(Pulse(
            line=LineRef("qubit", "drive"),
            freq_mhz=4000.0,
            gain=0.25,
            phase_deg=0.0,
            envelope=Gaussian(sigma_us=0.03),
            label="pi_ge",
        ))
        seq.wait(0.5)
        seq.measure()

        wire = seq_to_wire(seq)
        assert len(wire["ops"]) == 3
        assert wire["ops"][0]["type"] == "Pulse"
        assert wire["ops"][0]["envelope"]["type"] == "Gaussian"
        assert wire["ops"][0]["label"] == "pi_ge"
        assert wire["ops"][1]["type"] == "Delay"
        assert wire["ops"][2]["type"] == "Measure"
        # JSON-safe
        json.dumps(wire)

    def test_arb_envelope(self):
        from strumento.core.pulses import Arb, Pulse, Seq
        from strumento.core.wiring import LineRef

        oc_pulse = Pulse(
            line=LineRef("qubit", "drive"),
            freq_mhz=4000.0,
            gain=0.25,
            envelope=Arb(idata=(0.0, 0.5, 1.0, 0.5, 0.0), qdata=(0.1, 0.2, 0.3, 0.2, 0.1)),
            label="oc_pi",
        )
        seq = Seq().play(oc_pulse)
        wire = seq_to_wire(seq)
        env = wire["ops"][0]["envelope"]
        assert env["type"] == "Arb"
        assert len(env["idata"]) == 5
        assert len(env["qdata"]) == 5
        json.dumps(wire)

    def test_readout_binding_is_preserved(self):
        from strumento.core.pulses import Const, Pulse, ReadoutBinding, Seq
        from strumento.core.wiring import LineRef

        seq = Seq().play(Pulse(
            line=LineRef("readout", "probe"),
            freq_mhz=750.0,
            gain=0.1,
            envelope=Const(length_us=1.5),
            readout=ReadoutBinding(adc_ch=0, phase_deg=0.0, window_us=1.5),
        ))

        wire = seq_to_wire(seq)
        assert wire["ops"][0]["readout"] == {"adc_ch": 0, "phase_deg": 0.0, "window_us": 1.5}

    def test_swept_phase_deg_is_serialized(self):
        """A swept phase_deg (e.g. a ParityFringe phase scan) round-trips as a __sweep__ dict.

        Regression: phase_deg bypassed the Sweep-aware _field(), so a phase-swept pulse raised
        'Object of type Sweep is not JSON serializable' when a runner persisted it — while a
        gain-swept pulse (which did go through _field) serialized fine.
        """
        from strumento.core.pulses import Gaussian, Pulse, Seq
        from strumento.core.sweeps import Sweep
        from strumento.core.wiring import LineRef

        seq = Seq().play(Pulse(
            line=LineRef("qubit", "drive"),
            freq_mhz=4000.0,
            gain=0.25,
            phase_deg=Sweep(start=0.0, stop=360.0, on="phase"),
            envelope=Gaussian(sigma_us=0.03),
            label="hpi_scan",
        ))
        wire = seq_to_wire(seq)
        assert wire["ops"][0]["phase_deg"] == {
            "__sweep__": True, "start": 0.0, "stop": 360.0, "on": "phase", "name": None,
        }
        json.dumps(wire)  # must be JSON-safe

    def test_swept_flat_top_length_is_serialized(self):
        """A swept FlatTop length_us (e.g. a SidebandRabi/SwapRabi length sweep) round-trips.

        Regression: FlatTop.length_us bypassed the Sweep-aware _field() (Const's already went
        through it), so a length-swept flat-top pulse raised 'Object of type Sweep is not JSON
        serializable' when a runner persisted the compiled sequence.
        """
        from strumento.core.pulses import FlatTop, Pulse, Seq
        from strumento.core.sweeps import Sweep
        from strumento.core.wiring import LineRef

        seq = Seq().play(Pulse(
            line=LineRef("qubit", "sideband"),
            freq_mhz=2006.5,
            gain=0.24,
            envelope=FlatTop(ramp_sigma_us=0.005, length_us=Sweep(start=0.0, stop=2.5, on="t")),
            label="f0-g1_len_scan",
        ))
        wire = seq_to_wire(seq)
        assert wire["ops"][0]["envelope"]["length_us"] == {
            "__sweep__": True, "start": 0.0, "stop": 2.5, "on": "t", "name": None,
        }
        json.dumps(wire)  # must be JSON-safe

    def test_sequence_with_seq_saved_to_datachest(self, chest):
        """Seq provenance survives a datachest round-trip."""
        from strumento.core.pulses import Gaussian, Pulse, Seq
        from strumento.core.wiring import LineRef

        seq = Seq().play(Pulse(
            line=LineRef("q0", "drive"), freq_mhz=5000.0, gain=0.4,
            envelope=Gaussian(sigma_us=0.02), label="spin_pi",
        )).measure()

        result = _make_result()
        entry_id = chest.save(result, seq=seq, experiment="spin_t1")
        loaded = chest.load(entry_id)

        assert loaded["sequence"] is not None
        assert loaded["sequence"]["ops"][0]["label"] == "spin_pi"
        assert loaded["sequence"]["ops"][0]["envelope"]["type"] == "Gaussian"
