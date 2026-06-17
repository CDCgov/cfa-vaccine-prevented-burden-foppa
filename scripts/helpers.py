import math
import string
from collections.abc import Iterable

from matplotlib.axes import Axes
from matplotlib.ticker import FuncFormatter

# See the "Wong" palette at <https://davidmathlogic.com/colorblind/>
colors = {
    "original": "#009E73",
    "improved": "#F78F47",
    "vax": "#0072B2",
    "oc": "black",
    "oc_cf": "gray",
}


def color_scale(domain: list[str]) -> list[str]:
    return [colors[x] for x in domain]


def _format_with_commas(value: float, _: float) -> str:
    if not math.isfinite(value):
        return ""
    if float(value).is_integer():
        return f"{int(value):,}"
    text = f"{value:,.6f}"
    return text.rstrip("0").rstrip(".")


def apply_comma_ticks(ax: Axes, x: bool = True, y: bool = True) -> None:
    formatter = FuncFormatter(_format_with_commas)
    if x:
        ax.xaxis.set_major_formatter(formatter)
    if y:
        ax.yaxis.set_major_formatter(formatter)


def add_panel_labels(
    axes: Iterable[Axes],
    start: int = 0,
    x: float = 0.02,
    y: float = 0.98,
) -> None:
    labels = [f"{letter})" for letter in string.ascii_lowercase]
    for i, ax in enumerate(axes):
        label = labels[(start + i) % len(labels)]
        ax.text(
            x,
            y,
            label,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=11,
            fontweight="bold",
        )
