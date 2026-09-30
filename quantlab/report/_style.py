"""One consistent chart style for the whole tear sheet (dataviz skill: form -> color -> marks).

Palette is the validated reference instance (dataviz skill, ``references/palette.md``): fixed
categorical order, one sequential hue (blue) for magnitude, blue<->red for polarity (PnL sign,
drawdown), a neutral mid-grey text/grid so figures read on both light and dark notebook/Obsidian
themes (transparent figure background, no surface color to clash with either theme).
"""

from __future__ import annotations

from contextlib import contextmanager

import matplotlib as mpl

# No backend is forced here: a notebook kernel needs its own interactive backend for inline
# figures (DESIGN §7).  A headless caller (tests, a batch report run) selects one itself, e.g.
# ``matplotlib.use("Agg")`` *before* importing this module — matplotlib's own convention.

# Categorical order (fixed; dataviz skill) -- used only where more than one series must be told
# apart (e.g. CPCV path fan vs its median).  Everything else uses the sequential/diverging roles.
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
YELLOW = "#eda100"
MAGENTA = "#e87ba4"
GREEN = "#008300"
VIOLET = "#4a3aa7"
RED = "#e34948"
CATEGORICAL = (BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED)

# Sequential (magnitude): one hue, light -> dark (equity line, CPCV fan).
SEQ_BLUE = ("#cde2fb", "#9ec5f4", "#5598e7", "#2a78d6", "#184f95")

# Diverging (polarity): blue <-> red, neutral grey midpoint (monthly-return heatmap, drawdown).
DIVERGING = (BLUE, "#f0efec", RED)

# Status (fixed, never re-themed): good / warning / serious / critical.
GOOD = "#0ca30c"
WARNING = "#fab219"
SERIOUS = "#ec835a"
CRITICAL = "#d03b3b"

INK = "#6e6d68"           # mid-grey primary ink: legible on both a light and a dark surface
MUTED = "#898781"
GRID = "#9a988f"


@contextmanager
def style():
    """Context manager: apply the tear sheet's rcParams (transparent surfaces, one palette,
    thin recessive gridlines) for the duration of one figure, then restore matplotlib's state."""
    with mpl.rc_context({
        "figure.facecolor": "none",
        "axes.facecolor": "none",
        "savefig.facecolor": "none",
        "savefig.transparent": True,
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK,
        "axes.titlecolor": INK,
        "text.color": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "grid.color": GRID,
        "grid.alpha": 0.35,
        "grid.linewidth": 0.6,
        "axes.grid": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 10,
        "axes.titlesize": 11,
        "legend.frameon": False,
        "lines.linewidth": 1.6,
    }):
        yield
