"""
Tests for the band-1 fit: row selection, vertex recovery on an exact parabola (full and
leave-one-out fits), convexity guard, and the three checks.
"""

import numpy as np
import pytest

from src.fit import (
    SHAPE_BAR,
    UNCERTAINTY_BAR,
    Jackknife,
    ParabolaFit,
    check_convexity,
    check_location,
    check_shape,
    check_uncertainty,
    fit_parabola,
    jackknife,
    load_rows,
    select_fit_rows,
)
from src.run_plan import EXPECTED_RUNGS

# An exact parabola sampled like band 1: 7 points, mean far from zero, vertex off-centre
A, X0, K = 2.0, 6.8, 1.58
X = np.linspace(6.35, 7.35, 7)
Y = A * (X - X0) ** 2 + K


def parabola_with(
    vertex: float, y_min: float, x_bar: float, a: float = 1.0
) -> ParabolaFit:
    # Coefficients in u = x − x_bar that put the vertex and minimum where asked
    u0 = vertex - x_bar
    return ParabolaFit(a=a, b=-2 * a * u0, c=y_min + a * u0**2, x_bar=x_bar)


def test_selects_exactly_one_row_per_rung_from_band1_csv() -> None:
    fit_rows = select_fit_rows(load_rows())
    assert len(fit_rows) == len(EXPECTED_RUNGS)
    for i, rung in enumerate(EXPECTED_RUNGS):
        assert float(fit_rows[i]["peak_lr"]) == pytest.approx(1.8e-3, rel=1e-12)
        assert int(fit_rows[i]["d"]) == rung[0]
        assert int(fit_rows[i]["L"]) == rung[1]
        assert int(fit_rows[i]["H"]) == rung[2]


def test_selection_fails_on_a_duplicate_rung() -> None:
    rows = load_rows()
    fit_rows = select_fit_rows(rows)
    rerun = {**fit_rows[3], "run_id": "rerun"}
    with pytest.raises(ValueError, match="duplicate"):
        _ = select_fit_rows([*rows, rerun])


def test_selection_fails_on_a_missing_rung() -> None:
    rows = load_rows()
    dropped = select_fit_rows(rows)[3]
    with pytest.raises(ValueError, match="missing"):
        _ = select_fit_rows([row for row in rows if row is not dropped])


def test_recovers_known_vertex_of_exact_parabola_with_mean_far_from_zero() -> None:
    a, x0, k = 2.0, 6.8, 1.58
    x = np.linspace(6.35, 7.35, 7)
    y = a * (x - x0) ** 2 + k
    fit = fit_parabola(x, y, float(np.mean(x)))
    assert fit.vertex == pytest.approx(x0, rel=1e-12)
    assert fit.y_min == pytest.approx(k, rel=1e-12)
    assert fit.a == pytest.approx(a, rel=1e-12)


def test_leave_one_out_fit_recovers_the_same_vertex_on_exact_data() -> None:
    jk = jackknife(X, Y, float(np.mean(X)))
    assert len(jk.vertices) == len(X)
    assert jk.vertices == pytest.approx(np.full(len(X), X0), rel=1e-12)
    assert jk.a_values == pytest.approx(np.full(len(X), A), rel=1e-12)


def test_recovers_fitted_minimum_of_exact_parabola() -> None:
    # Vertex near one end of the samples, so b in u is far from zero
    x0, k = 7.2, 1.6
    y = A * (X - x0) ** 2 + k
    fit = fit_parabola(X, y, float(np.mean(X)))
    assert fit.y_min == pytest.approx(k, rel=1e-12)


def test_jackknife_se_is_zero_on_exact_data() -> None:
    jk = jackknife(X, Y, float(np.mean(X)))
    assert jk.se == pytest.approx(0.0, abs=1e-9)
    assert jk.spread == pytest.approx(0.0, abs=1e-9)


def test_concave_data_gives_non_positive_a() -> None:
    x_bar = float(np.mean(X))
    fit = fit_parabola(X, -Y, x_bar)  # returned, not raised
    assert fit.a <= 0
    assert not check_convexity(fit, jackknife(X, -Y, x_bar))


def test_location_check_rejects_vertex_outside_the_range() -> None:
    x_bar = float(np.mean(X))
    assert check_location(parabola_with(X0, K, x_bar), X)
    assert not check_location(parabola_with(X.max() + 0.1, K, x_bar), X)
    assert not check_location(parabola_with(X.min() - 0.1, K, x_bar), X)
    assert not check_location(parabola_with(float(X.max()), K, x_bar), X)  # strict


def test_uncertainty_check_fails_when_spread_exceeds_bar_even_if_se_passes() -> None:
    # One wild leave-one-out vertex: the SE smooths it to just under the bar
    vertices = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2])
    jk = Jackknife(vertices=vertices, a_values=np.ones(7))
    assert jk.se < UNCERTAINTY_BAR
    assert jk.spread > UNCERTAINTY_BAR
    assert not check_uncertainty(jk)
    assert check_uncertainty(Jackknife(vertices=vertices / 2, a_values=np.ones(7)))


def test_shape_check_uses_both_observed_and_fitted_minimum() -> None:
    x_bar = float(np.mean(X))
    y = np.array([1.06, 1.03, 1.01, 1.00, 1.01, 1.03, 1.06])  # edges 6% above the min
    assert check_shape(parabola_with(X0, 1.00, x_bar), y)
    # Fitted minimum above the observed one: edges only ~3.9% above y*
    assert not check_shape(parabola_with(X0, 1.02, x_bar), y)
    # Observed minimum above the fitted one: edges only ~3.9% above the lowest loss
    raised = np.array([1.06, 1.04, 1.03, 1.02, 1.03, 1.04, 1.06])
    assert not check_shape(parabola_with(X0, 1.00, x_bar), raised)
    assert SHAPE_BAR == 0.05


def test_convexity_check_fails_when_one_leave_one_out_fit_is_concave() -> None:
    fit = parabola_with(X0, K, float(np.mean(X)))
    assert check_convexity(fit, Jackknife(vertices=np.zeros(7), a_values=np.ones(7)))
    a_values = np.array([1.0, 1.0, 1.0, -0.1, 1.0, 1.0, 1.0])
    assert not check_convexity(fit, Jackknife(vertices=np.zeros(7), a_values=a_values))
