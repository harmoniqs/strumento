"""cqed pulse libraries — element-type-keyed ops classes (D21).

Each module is keyed to an element *type*, not a device instance name: ``transmon.py`` ->
``TransmonOps``, ``mode.py`` -> ``ModeOps``, ``coupler.py`` -> ``CouplerOps``. A device binds
one ops instance per *named* element at load time, so ``dev.qubit`` / ``dev.cavity`` exist
because the instance YAML names them that — never because the pack hardcodes those names.

Gain units
----------
Every ops factory that takes ``gain=`` (``TransmonOps.ge_pi`` / ``ge_hpi`` / ``ef_pi`` /
``ef_hpi``, ``ReadoutOps.probe`` / ``dispersive_measure``, ``ModeOps.displace``,
``CouplerOps.swap``, ``SidebandOps.f0g1`` / ``multiphoton``) reads it in one of three ways,
depending on its type (D23). The pulse it returns always carries a **fraction of full scale**;
``maxv`` is the full-scale code of the generator that plays the pulse, from the device's soccfg
snapshot (``2**15 - 1`` when no snapshot is attached; 32766 on the demo devices).

``gain=None`` (the default)
    The calibrated gain. The calibration stores it as a **DAC integer code** (e.g.
    ``calib.qubit.pulses["pi_ge"].gain``); the factory converts it with
    :func:`~strumento.packs.cqed.calibration.gain_int_to_frac`, ``code / maxv``. Code 8192 at
    ``maxv = 32766`` is a fraction of about 0.25.
a number, e.g. ``gain=0.2``
    A **fraction of full scale**, used as is: no conversion. An integer code passed here is
    read as a fraction, so ``gain=8192`` is refused by the limit check below rather than
    played.
a :class:`~strumento.core.sweeps.Sweep`, e.g. ``gain=Sweep(start=0, stop=16000, on="amp")``
    **Integer DAC codes**, converted at each endpoint (``start / maxv``, ``stop / maxv``). The
    pulse's gain is a new ``Sweep`` over fractions on the same axis; the program's axis
    (``xpts``) comes back in codes. A nonzero endpoint with ``|v| < 1`` looks like a fraction
    and is refused (``GainOutOfRange``) instead of being divided by ``maxv`` again; ``0`` is a
    normal code.

Every resulting fraction, calibrated or explicit, and every sweep endpoint after conversion,
faces its role's limit (:mod:`strumento.packs.cqed.safety`, or the per-element limit when the
calibration sets one: ``max_drive_gain``, ``max_readout_gain``, ``max_displace_gain``); a
value over it raises ``GainOutOfRange`` before a pulse exists.

A ``TransmonOps`` built with ``overrides=`` plays the override pulse's own gain for
``gain=None``. That gain is already a fraction, so it is not converted, and it faces the drive
limit like every other pulse (a ``Sweep`` gain on an override is a fraction sweep: each endpoint
is checked as a fraction). A number or a ``Sweep`` passed as ``gain=`` is read as above.

One variation: ``ModeOps.displace_alpha`` takes no ``gain=``. It converts
``displacement_cal × |α|``, a code, the same way as ``gain=None``.
"""
