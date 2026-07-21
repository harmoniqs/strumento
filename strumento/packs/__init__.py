"""Modality packs — reusable physics kits.

A pack supplies element *types*, pulse libraries, a readout kind, experiment
types, and a calibration *schema* for one physics class. Packs are discovered
via the ``strumento.packs`` entry-point group, so third-party packs install
rather than fork.

Status:
- ``cqed`` (superconducting transmon + bosonic) — **built** (slices 1–2).
- ``tweezer`` (neutral-atom AOD) and ``spin`` (quantum-dot) — **skeletons**: element types,
  calibration schema, and a readout kind, wired through the pack contract so ``core.device.load``
  composes them with no core change; pulse libraries / experiments are later work.

See ``docs/packs-and-devices.md`` for the element-composition model and how to add a pack.
"""
