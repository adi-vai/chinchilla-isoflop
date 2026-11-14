"""
Tests for the per-run LR schedule: warmup start and peak, cosine midpoint, monotonic decay, min LR at and after the final step.
Steps are 0-indexed; warmup is nanoGPT-style (peak at step == warmup); decay ends at steps − 1.
"""

import pytest

from src.ladder import make_config
from src.run_plan import (
    EXPECTED_RUNGS,
    TOKENS_PER_STEP,
    RunPlan,
    band_compute_budget,
    plan_run,
)
from src.schedule import get_lr
from src.util import flops_per_token

C = band_compute_budget(make_config(*EXPECTED_RUNGS[0]))
_fake = make_config(*EXPECTED_RUNGS[0])
PLANS = [plan_run(make_config(*r), C, TOKENS_PER_STEP) for r in EXPECTED_RUNGS] + [
    plan_run(_fake, 1_000 * flops_per_token(_fake), 1)  # floor case: warmup = 100
]
IDS = [f"d{r[0]}" for r in EXPECTED_RUNGS] + ["floor"]

PEAK_LR = 6e-4
MIN_LR = PEAK_LR / 10


@pytest.mark.parametrize("plan", PLANS, ids=IDS)
def test_lr_at_step_zero_is_peak_over_warmup_plus_one(plan: RunPlan) -> None:
    lr = get_lr(0, plan.steps, plan.warmup_steps, PEAK_LR, MIN_LR)
    assert lr == pytest.approx(PEAK_LR / (plan.warmup_steps + 1), rel=1e-12)


@pytest.mark.parametrize("plan", PLANS, ids=IDS)
def test_lr_reaches_peak_at_step_warmup(plan: RunPlan) -> None:
    # Steps are 0-indexed
    lr = get_lr(plan.warmup_steps - 1, plan.steps, plan.warmup_steps, PEAK_LR, MIN_LR)
    assert lr < PEAK_LR
    lr = get_lr(plan.warmup_steps, plan.steps, plan.warmup_steps, PEAK_LR, MIN_LR)
    assert lr == pytest.approx(PEAK_LR, rel=1e-12)


@pytest.mark.parametrize("plan", PLANS, ids=IDS)
def test_lr_at_decay_midpoint_is_mean_of_peak_and_min(plan: RunPlan) -> None:
    if (plan.steps - 1 - plan.warmup_steps) % 2 == 0:
        mid_point = plan.warmup_steps + (plan.steps - 1 - plan.warmup_steps) / 2
        lr = get_lr(int(mid_point), plan.steps, plan.warmup_steps, PEAK_LR, MIN_LR)
        assert lr == pytest.approx((PEAK_LR + MIN_LR) / 2, rel=1e-12)
    else:  # bracketing
        left = int(plan.warmup_steps + (plan.steps - 1 - plan.warmup_steps - 1) / 2)
        right = int(plan.warmup_steps + (plan.steps - 1 - plan.warmup_steps + 1) / 2)
        left_lr = get_lr(left, plan.steps, plan.warmup_steps, PEAK_LR, MIN_LR)
        right_lr = get_lr(right, plan.steps, plan.warmup_steps, PEAK_LR, MIN_LR)
        assert left_lr > (PEAK_LR + MIN_LR) / 2
        assert right_lr < (PEAK_LR + MIN_LR) / 2


@pytest.mark.parametrize("plan", PLANS, ids=IDS)
def test_lr_never_increases_after_peak(plan: RunPlan) -> None:
    prev_lr = get_lr(plan.warmup_steps, plan.steps, plan.warmup_steps, PEAK_LR, MIN_LR)
    for current_step in range(plan.warmup_steps + 1, plan.steps):
        current_lr = get_lr(
            current_step, plan.steps, plan.warmup_steps, PEAK_LR, MIN_LR
        )
        assert current_lr < prev_lr
        prev_lr = current_lr


@pytest.mark.parametrize("plan", PLANS, ids=IDS)
def test_lr_reaches_min_at_final_step(plan: RunPlan) -> None:
    lr = get_lr(plan.steps - 1, plan.steps, plan.warmup_steps, PEAK_LR, MIN_LR)
    assert lr == pytest.approx(MIN_LR, rel=1e-12)


@pytest.mark.parametrize("plan", PLANS, ids=IDS)
def test_lr_stays_at_min_after_final_step(plan: RunPlan) -> None:
    lr = get_lr(plan.steps, plan.steps, plan.warmup_steps, PEAK_LR, MIN_LR)
    assert lr == pytest.approx(MIN_LR, rel=1e-12)
    lr = get_lr(plan.steps + 10, plan.steps, plan.warmup_steps, PEAK_LR, MIN_LR)
    assert lr == pytest.approx(MIN_LR, rel=1e-12)
