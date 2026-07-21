"""cqed pulse libraries — element-type-keyed ops classes (D21).

Each module is keyed to an element *type*, not a device instance name: ``transmon.py`` ->
``TransmonOps``, ``mode.py`` -> ``ModeOps``, ``coupler.py`` -> ``CouplerOps``. A device binds
one ops instance per *named* element at load time, so ``dev.qubit`` / ``dev.manipulate`` exist
because the instance YAML names them that — never because the pack hardcodes those names.
"""
