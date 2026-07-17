"""Modality-agnostic core: wiring, pulse IR, sweeps, device, compiler, program,
mock, result, readout/feedback/environment protocols, and generic fitters.

Only ``compiler``, ``program``, and ``mock`` import ``qick``; everything else is
importable and unit-testable with no qick and no board.
"""
