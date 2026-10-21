"""Spin is simulatable from a fresh process (issue #132).

In-process tests cannot see this: by the time they run, some other test (or the test module
itself) has already imported ``strumento.packs.spin.responses``. So every check here runs in a
fresh ``python -c`` subprocess, where only what the code under test imports is loaded.
"""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap

import pytest

from tests._paths import ROOT, SPIN_YAML


def _run(code: str, *args: str, timeout: float = 120) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code), *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def _ok(proc: subprocess.CompletedProcess) -> str:
    assert proc.returncode == 0, f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    return proc.stdout


def test_importing_the_spin_pack_registers_its_response_models():
    out = _ok(_run("""
        import json
        import strumento.packs.spin as spin
        from strumento.core.response import registered_responses
        regs = set(registered_responses())
        spin_regs = [t for t in regs if t.__module__.startswith("strumento.packs.spin")]
        print(json.dumps({
            "spin_registered": sorted(t.__name__ for t in spin_regs),
            "spin_rabi": spin.EXPERIMENTS["spin_rabi"] in regs,
            "spin_t1": spin.EXPERIMENTS["spin_t1"] in regs,
        }))
    """))
    got = json.loads(out)
    assert got["spin_rabi"] and got["spin_t1"], got
    assert got["spin_registered"], "`import strumento.packs.spin` registered no response models"


def test_importing_the_spin_pack_does_not_import_qick():
    # qick poisoned: any `import qick` (direct or transitive) raises ImportError.
    out = _ok(_run("""
        import sys
        sys.modules["qick"] = None
        import strumento.packs.spin  # noqa: F401
        from strumento.core.response import registered_responses
        assert registered_responses(), "spin import registered nothing"
        print(sorted(m for m in sys.modules if m == "qick" or m.startswith("qick.")))
    """))
    assert out.strip() == "['qick']"  # only our poison entry, nothing qick-derived loaded


def test_spin_registration_does_not_change_cqed_response_resolution():
    out = _ok(_run("""
        import json
        import strumento.packs.cqed as cqed
        from strumento.core.response import resolve_response

        def resolved():
            out = {}
            for name, cls in cqed.EXPERIMENTS.items():
                try:
                    m = resolve_response(cls)
                except Exception:
                    continue
                out[name] = f"{m.__module__}:{m.__qualname__}"
            return out

        before = resolved()
        import strumento.packs.spin  # noqa: F401
        after = resolved()
        print(json.dumps({"before": before, "after": after}))
    """))
    got = json.loads(out)
    assert got["before"], "no cqed experiment resolved a response"
    assert got["after"] == got["before"]
    # amplitude_rabi in particular still resolves to the cqed pack's own model
    assert got["after"]["amplitude_rabi"].startswith("strumento.packs.cqed.")


@pytest.mark.qick
def test_fresh_process_runs_a_spin_experiment_on_spin_demo():
    pytest.importorskip("qick")
    out = _ok(_run("""
        import json, sys
        from strumento import Device
        from strumento.core.simulator import SimulatorSoc
        from strumento.packs.spin import EXPERIMENTS

        dev = Device.load(sys.argv[1])
        soc = SimulatorSoc.from_device(dev, truth={"T1": 68.0}, seed=42)
        outcome = EXPERIMENTS["spin_t1"](dev, dot_name="q0").go(soc, reps=200, progress=False)
        print(json.dumps({"success": bool(outcome.fit is not None and outcome.fit.success),
                          "tau": float(outcome.fit.derived["tau"])}))
    """, str(SPIN_YAML)))
    got = json.loads(out.strip().splitlines()[-1])
    assert got["success"], got
    assert got["tau"] == pytest.approx(68.0, rel=0.15)


@pytest.mark.qick
def test_spin_smoke_script_reports_per_experiment_results():
    pytest.importorskip("qick")
    from strumento.packs.spin import EXPERIMENTS

    proc = subprocess.run(
        [sys.executable, str(ROOT / "examples" / "sim_smoke" / "run_spin.py"), "--json"],
        cwd=ROOT, capture_output=True, text=True, timeout=600,
    )
    text = proc.stdout
    results = json.loads(text[text.index("["):])
    assert all(r["status"] != "SKIP" for r in results), results
    assert {r["experiment"] for r in results} == set(EXPERIMENTS)
    # every experiment gets through the pipeline (construct → compile → acquire → fit);
    # fit quality at the script's 10 reps is reported, not asserted
    assert [r for r in results if r["status"] == "ERROR"] == []
    assert any(r["status"] == "PASS" for r in results)
