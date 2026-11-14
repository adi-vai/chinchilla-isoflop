"""
Tests for the per-run plan: token coverage, realized compute, data margin, warmup.
"""

import math

import pytest

from src.ladder import TOKEN_MARGIN, TRAIN_TOKENS, build_ladder, make_config
from src.run_plan import EXPECTED_RUNGS, TOKENS_PER_STEP, band_compute_budget, plan_run
from src.util import flops_per_token

C = band_compute_budget(make_config(*EXPECTED_RUNGS[0]))


@pytest.mark.parametrize(("d", "L", "H"), EXPECTED_RUNGS)
def test_steps_cover_target_tokens_with_one_step_of_overshoot_at_most(
    d: int, L: int, H: int
) -> None:
    plan = plan_run(make_config(d, L, H), C, TOKENS_PER_STEP)
    assert plan.target_tokens <= plan.realized_tokens
    assert plan.realized_tokens < plan.target_tokens + TOKENS_PER_STEP


@pytest.mark.parametrize(("d", "L", "H"), EXPECTED_RUNGS)
def test_realized_compute_exceeds_budget_by_less_than_one_step(
    d: int, L: int, H: int
) -> None:
    config = make_config(d, L, H)
    plan = plan_run(config, C, TOKENS_PER_STEP)
    assert C <= plan.realized_compute
    assert plan.realized_compute < C + TOKENS_PER_STEP * flops_per_token(config)


@pytest.mark.parametrize(("d", "L", "H"), EXPECTED_RUNGS)
def test_tokens_never_exceed_margin_plus_one_step(d: int, L: int, H: int) -> None:
    plan = plan_run(make_config(d, L, H), C, TOKENS_PER_STEP)
    assert plan.realized_tokens < TOKEN_MARGIN * TRAIN_TOKENS + TOKENS_PER_STEP


@pytest.mark.parametrize(("d", "L", "H"), EXPECTED_RUNGS)
def test_warmup_is_five_percent_of_steps_rounded_up(d: int, L: int, H: int) -> None:
    plan = plan_run(make_config(d, L, H), C, TOKENS_PER_STEP)
    assert 100 < plan.warmup_steps  # never hit the floor
    assert plan.warmup_steps == math.ceil(0.05 * plan.steps)


def test_warmup_floor_applies_to_short_runs() -> None:
    config = make_config(*EXPECTED_RUNGS[0])
    fpt = flops_per_token(config)
    # A fake 1,000-step run, so warmup = 100
    plan = plan_run(config, 1_000 * fpt, 1)
    assert plan.steps == 1_000
    assert plan.warmup_steps == 100


def test_ladder_matches_chosen_rungs() -> None:
    ladder = build_ladder(head_size=32, aspect_ratio=32)
    assert ladder is not None, "Q=32, d/L=32 ladder is infeasible"
    rungs = [(r.model_width, r.model_depth, r.num_heads) for r in ladder]
    assert rungs == EXPECTED_RUNGS, f"ladder changed: {rungs}"


def test_band_compute_budget_matches_hand_calculated_value() -> None:
    assert C == pytest.approx(0.9 * 447_014_068 * 15_928_320, rel=1e-12)
