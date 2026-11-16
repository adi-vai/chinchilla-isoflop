"""Plot the band-1 IsoFLOP fit: the 7 runs, the fitted parabola, its vertex ± SE.

Every number comes from src.fit.fit_band(); nothing is refitted here. Writes
plots/band1_isoflop.png (200 dpi) and plots/band1_isoflop.svg.

Run from the repo root: `uv run python -m scripts.plot_band1`.
"""

import matplotlib

matplotlib.use("Agg")  # files only, no window

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

from src.fit import fit_band
from src.train import REPO_ROOT

PLOTS_DIR = REPO_ROOT / "plots"

# Reference palette (dataviz skill): one series, recessive ink and grid
SERIES = "#2a78d6"
TEXT = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"

N_TICKS_M = (2, 3, 4, 5, 6, 7, 10, 15, 20)  # x ticks, in millions of params
# Rung label offsets in points, clear of the curve and the vertex marker; default
# is centred above the point
LABEL_OFFSETS = {192: (0, -10), 256: (0, -10), 288: (0, -10), 384: (-12, 6)}


def sci(value: float, digits: int) -> str:
    # 6.41e15 rather than 6.41e+15
    mantissa, exponent = f"{value:.{digits}e}".split("e")
    return f"{mantissa}e{int(exponent)}"


def schedule_label(rows: list[dict[str, str]]) -> str:
    # The LR schedule the fit rows actually ran with; every row must agree
    schedules = {(float(r["peak_lr"]), float(r["min_lr"])) for r in rows}
    if len(schedules) != 1:
        raise ValueError(f"fit rows ran with different LR schedules: {schedules}")
    ((peak_lr, min_lr),) = schedules
    return f"{sci(peak_lr, 1)} peak LR, cosine to {min_lr / peak_lr:.0%}"


def main() -> None:
    band = fit_band()
    fit, jk = band.fit, band.jk
    n = 10**band.x
    n_star = 10**fit.vertex
    se_factor = 10**jk.se

    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.edgecolor": TEXT_SECONDARY,
            "axes.labelcolor": TEXT,
            "xtick.color": TEXT_SECONDARY,
            "ytick.color": TEXT_SECONDARY,
            "svg.fonttype": "none",  # keep text as text in the SVG
        }
    )
    fig, ax = plt.subplots(figsize=(8, 5), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)

    # Vertex ± jackknife SE, as a band in N
    ax.axvspan(
        10 ** (fit.vertex - jk.se),
        10 ** (fit.vertex + jk.se),
        color=SERIES,
        alpha=0.12,
        linewidth=0,
    )

    # Fitted parabola over the sampled range only
    x_curve = np.linspace(band.x.min(), band.x.max(), 200)
    u = x_curve - fit.x_bar
    ax.plot(
        10**x_curve,
        fit.a * u**2 + fit.b * u + fit.c,
        color=SERIES,
        linewidth=2,
        zorder=2,
    )

    # The runs, each labelled with its rung
    ax.plot(
        n,
        band.y,
        "o",
        markersize=8,
        color=SERIES,
        markeredgecolor=SURFACE,
        markeredgewidth=2,
        zorder=3,
    )
    for row, ni, yi in zip(band.rows, n, band.y, strict=True):
        dx, dy = LABEL_OFFSETS.get(int(row["d"]), (0, 9))
        ax.annotate(
            f"d{row['d']}",
            (ni, yi),
            xytext=(dx, dy),
            textcoords="offset points",
            ha="center" if dx == 0 else ("right" if dx < 0 else "left"),
            va="top" if dy < 0 else "bottom",
            fontsize=10,
            color=TEXT_SECONDARY,
        )

    # Vertex
    ax.plot(
        n_star,
        fit.y_min,
        "D",
        markersize=8,
        color=TEXT,
        markeredgecolor=SURFACE,
        markeredgewidth=1.5,
        zorder=4,
    )

    # Leave-one-out vertices: faint ticks along the bottom of the axes
    tick_axes = ax.get_xaxis_transform()  # x in data, y in axes fraction
    ax.vlines(
        10**jk.vertices,
        0.03,
        0.09,
        transform=tick_axes,
        color=TEXT_SECONDARY,
        alpha=0.5,
        linewidth=1,
    )
    ax.text(
        10 ** (jk.vertices.max() + 0.03),
        0.06,
        "leave-one-out vertices",
        transform=tick_axes,
        va="center",
        fontsize=9,
        color=TEXT_SECONDARY,
    )

    ax.text(
        0.03,
        0.05,
        f"N* ≈ {n_star / 1e6:.1f}M (×/÷ {se_factor:.2f}, jackknife SE)\n"
        + f"y* = {fit.y_min:.3f}\n"
        + "1 seed per rung\n"
        + schedule_label(band.rows),
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=10,
        color=TEXT,
        linespacing=1.5,
        bbox={
            "boxstyle": "round,pad=0.5",
            "facecolor": SURFACE,
            "edgecolor": GRID,
        },
    )

    ax.set_xscale("log")
    ax.xaxis.set_major_locator(FixedLocator([m * 1e6 for m in N_TICKS_M]))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v / 1e6:g}M"))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xlim(n.min() / 1.12, n.max() * 1.12)
    y_pad = 0.25 * (band.y.max() - band.y.min())
    ax.set_ylim(band.y.min() - 1.5 * y_pad, band.y.max() + 0.6 * y_pad)
    ax.set_xlabel("N, total parameters (log scale)")
    ax.set_ylabel("final validation loss")
    realized = [float(row["realized_compute"]) for row in band.rows]
    ax.set_title(
        f"IsoFLOP band 1: TinyStories, C ≈ {sci(float(np.mean(realized)), 2)} FLOPs",
        loc="left",
        color=TEXT,
    )

    fig.tight_layout()
    PLOTS_DIR.mkdir(exist_ok=True)
    for suffix, kwargs in ((".png", {"dpi": 200}), (".svg", {})):
        path = PLOTS_DIR / f"band1_isoflop{suffix}"
        fig.savefig(path, facecolor=SURFACE, **kwargs)
        print(f"wrote {path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
