"""Band-1 IsoFLOP fit: parabola in log10 N, vertex, leave-one-out jackknife, checks.

Specification: NOTES.md, "Fit specification (decided 16 Nov)".

Run from the repo root: `uv run python -m src.fit`.
"""

import csv
import logging
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.run_plan import EXPECTED_RUNGS

logger = logging.getLogger(__name__)

RESULTS_CSV = Path(__file__).resolve().parent.parent / "results" / "band1.csv"

BAND_PEAK_LR = 1.8e-3
UNCERTAINTY_BAR = float(np.log10(1.5))  # ≈ 0.176 in log10 N
SHAPE_BAR = 0.05  # edges ≥ 5% above the minimum


@dataclass(frozen=True)
class ParabolaFit:
    a: float  # coefficients in u = x − x_bar, highest power first
    b: float
    c: float
    x_bar: float  # the centring used for this fit

    @property
    def vertex(self) -> float:
        """x* in log10 N (x_bar added back)."""
        return -self.b / (2 * self.a) + self.x_bar

    @property
    def y_min(self) -> float:
        """Fitted minimum loss y*."""
        return self.c - self.b**2 / (4 * self.a)


@dataclass(frozen=True)
class Jackknife:
    vertices: np.ndarray  # x*_(i), one per left-out point
    a_values: np.ndarray  # a for each leave-one-out fit (convexity guard)

    @property
    def se(self) -> float:
        """Jackknife SE: sqrt((n − 1)/n · Σ (x*_(i) − mean x*)²)."""
        n = len(self.vertices)
        deviations = self.vertices - self.vertices.mean()
        return float(np.sqrt((n - 1) / n * np.sum(deviations**2)))

    @property
    def spread(self) -> float:
        """max − min of the leave-one-out vertices."""
        return float(self.vertices.max() - self.vertices.min())


def load_rows(path: Path = RESULTS_CSV) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def select_fit_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """Rows entering the fit, one per rung in EXPECTED_RUNGS order.

    NOTES.md, "Fit row selection": peak LR 1.8e-3, seed 0, clip 1.0, real runs.
    Any missing, duplicate or unexpected rung is a ValueError, never a silent pick.
    """
    by_rung: dict[tuple[int, int, int], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if (
            math.isclose(float(row["peak_lr"]), BAND_PEAK_LR, rel_tol=1e-12)
            and int(row["seed"]) == 0
            and float(row["grad_clip"]) == 1.0
            and row["rehearsal"] == "False"
        ):
            by_rung[(int(row["d"]), int(row["L"]), int(row["H"]))].append(row)

    problems = []
    missing = [rung for rung in EXPECTED_RUNGS if rung not in by_rung]
    if missing:
        problems.append(f"missing rungs {missing}")
    duplicates = {
        rung: [row["run_id"] for row in kept]
        for rung, kept in by_rung.items()
        if len(kept) > 1
    }
    if duplicates:
        problems.append(f"duplicate rungs {duplicates}")
    unexpected = sorted(set(by_rung) - set(EXPECTED_RUNGS))
    if unexpected:
        problems.append(f"unexpected rungs {unexpected}")
    if problems:
        raise ValueError("fit row selection: " + "; ".join(problems))
    return [by_rung[rung][0] for rung in EXPECTED_RUNGS]


def fit_parabola(x: np.ndarray, y: np.ndarray, x_bar: float) -> ParabolaFit:
    """Fit y = a·u² + b·u + c with u = x − x_bar. a ≤ 0 is returned, not raised:
    check_convexity judges it."""
    a, b, c = np.polyfit(x - x_bar, y, 2)
    return ParabolaFit(float(a), float(b), float(c), x_bar)


def jackknife(x: np.ndarray, y: np.ndarray, x_bar: float) -> Jackknife:
    """Leave-one-out refits, all centred on the same full-sample x_bar."""
    fits = [
        fit_parabola(np.delete(x, i), np.delete(y, i), x_bar) for i in range(len(x))
    ]
    return Jackknife(
        vertices=np.array([fit.vertex for fit in fits]),
        a_values=np.array([fit.a for fit in fits]),
    )


def check_convexity(fit: ParabolaFit, jk: Jackknife) -> bool:
    """a > 0 in the full fit and in every leave-one-out fit."""
    return fit.a > 0 and bool(np.all(jk.a_values > 0))


def check_location(fit: ParabolaFit, x: np.ndarray) -> bool:
    """Vertex strictly inside the sampled range of log10 N."""
    return bool(x.min() < fit.vertex < x.max())


def check_uncertainty(jk: Jackknife) -> bool:
    """Jackknife SE and leave-one-out spread both below UNCERTAINTY_BAR."""
    return jk.se < UNCERTAINTY_BAR and jk.spread < UNCERTAINTY_BAR


def shape_margins(fit: ParabolaFit, y: np.ndarray) -> dict[str, float]:
    """Relative excess of each edge loss over the lowest observed loss and over y*.

    y is in EXPECTED_RUNGS order, so y[0] is the smallest rung and y[-1] the largest.
    """
    floors = {"observed min": float(y.min()), "y*": fit.y_min}
    return {
        f"{edge} vs {name}": float(loss / floor - 1)
        for edge, loss in (("smallest", y[0]), ("largest", y[-1]))
        for name, floor in floors.items()
    }


def check_shape(fit: ParabolaFit, y: np.ndarray) -> bool:
    """Both edge losses ≥ SHAPE_BAR above the lowest observed loss and above y*."""
    return all(m >= SHAPE_BAR for m in shape_margins(fit, y).values())


@dataclass(frozen=True)
class BandFit:
    rows: list[dict[str, str]]  # the fit rows, in EXPECTED_RUNGS order
    x: np.ndarray  # log10 N_total
    y: np.ndarray  # final val loss
    fit: ParabolaFit
    jk: Jackknife


def fit_band(path: Path = RESULTS_CSV) -> BandFit:
    """Select the fit rows and run the full fit and the jackknife: one source for
    main() and the plot."""
    rows = select_fit_rows(load_rows(path))
    x = np.log10([float(row["total_params"]) for row in rows])
    y = np.array([float(row["final_val_loss"]) for row in rows])
    x_bar = float(x.mean())
    return BandFit(rows, x, y, fit_parabola(x, y, x_bar), jackknife(x, y, x_bar))


def non_embedding_params(row: dict[str, str]) -> int:
    """Kaplan-convention N for the robustness fit. Decide the definition (see src/util.py)."""
    raise NotImplementedError


def main() -> None:
    band = fit_band()
    rows, x, y, fit, jk = band.rows, band.x, band.y, band.fit, band.jk
    x_bar = fit.x_bar

    logger.info(f"{'rung':<14}{'N':>12}{'log10 N':>9}{'val loss':>10}")
    for row, xi, yi in zip(rows, x, y, strict=True):
        rung = "d{} L{} H{}".format(row["d"], row["L"], row["H"])
        logger.info(f"{rung:<14}{int(row['total_params']):>12,}{xi:>9.4f}{yi:>10.4f}")
    logger.info(
        f"\nfit in u = x − {x_bar:.4f}: a = {fit.a:.6f}, b = {fit.b:.6f}, "
        + f"c = {fit.c:.6f}"
    )
    logger.info(
        f"vertex: log10 N* = {fit.vertex:.4f}, N* = {10**fit.vertex:,.0f}; "
        + f"y* = {fit.y_min:.4f}"
    )

    logger.info(f"\n{'left out':<14}{'vertex':>9}{'a':>11}")
    for row, vertex, a in zip(rows, jk.vertices, jk.a_values, strict=True):
        rung = "d{} L{} H{}".format(row["d"], row["L"], row["H"])
        logger.info(f"{rung:<14}{vertex:>9.4f}{a:>11.6f}")
    logger.info(f"jackknife SE = {jk.se:.4f}, spread = {jk.spread:.4f}")

    logger.info("\nchecks (value vs bar):")
    logger.info(
        f"convexity: full a = {fit.a:.6f}, min leave-one-out a = "
        + f"{jk.a_values.min():.6f}  (bar: > 0) -> {check_convexity(fit, jk)}"
    )
    logger.info(
        f"location: vertex {fit.vertex:.4f} in ({x.min():.4f}, {x.max():.4f}) "
        + f"-> {check_location(fit, x)}"
    )
    logger.info(
        f"uncertainty: SE {jk.se:.4f}, spread {jk.spread:.4f}  "
        + f"(bar: each < {UNCERTAINTY_BAR:.4f}) -> {check_uncertainty(jk)}"
    )
    for name, margin in shape_margins(fit, y).items():
        logger.info(f"shape: {name}: {margin:+.2%}  (bar: >= {SHAPE_BAR:.0%})")
    logger.info(f"shape -> {check_shape(fit, y)}")


if __name__ == "__main__":
    # force: importing src.model has already configured logging
    logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
    main()
