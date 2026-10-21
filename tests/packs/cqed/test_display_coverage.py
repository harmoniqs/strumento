"""Every registered cqed experiment resolves to a display function.

``go(..., display=True)`` returns ``outcome.figure is None`` for an experiment with no registered
display, silently. This guard makes a missing display a test failure instead: it walks each
class in the pack's ``EXPERIMENTS`` registry through the same MRO lookup ``display_for`` uses.

Pure: no qick and no matplotlib — the lookup only reads the registry (``plot.py`` imports
matplotlib lazily, inside the plot functions).
"""

from __future__ import annotations

from strumento.core.display import _DISPLAYS
from strumento.packs.cqed import EXPERIMENTS


def _resolved_display(cls: type):
    """The plot function ``display_for`` would call for an instance of ``cls`` (or None)."""
    return next((_DISPLAYS[base] for base in cls.__mro__ if base in _DISPLAYS), None)


def test_every_registered_cqed_experiment_resolves_to_a_display():
    missing = sorted(cls.__name__ for cls in EXPERIMENTS.values() if _resolved_display(cls) is None)
    assert missing == [], f"cqed experiments with no registered display: {missing}"


def test_display_table_lists_each_experiment_once_and_is_what_is_registered():
    """A class in two rows would register twice, last-wins, hiding the first row's display."""
    from strumento.packs.cqed.plot import DISPLAY_TABLE

    listed = [cls for _group, _fn, classes in DISPLAY_TABLE for cls in classes]
    assert len(listed) == len(set(listed))
    for _group, fn, classes in DISPLAY_TABLE:
        for cls in classes:
            assert _DISPLAYS[cls] is fn, f"{cls.__name__} is not registered to its table row"
