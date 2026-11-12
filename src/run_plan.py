"""Per-run token and step plan for the band-1 ladder.

C is set by the smallest rung (it gets the most tokens); each run's target D is
C / flops_per_token, rounded up to whole steps, so realised C slightly exceeds C.

Run from the repo root: `uv run python -m src.run_plan`.
"""

import logging
import math
from dataclasses import dataclass

from src.ladder import (
    CONTEXT_LENGTH,
    TOKEN_MARGIN,
    TRAIN_TOKENS,
    build_ladder,
    make_config,
)
from src.model import ModelConfig
from src.util import flops_per_token

logger = logging.getLogger(__name__)

MICRO_BATCH_SIZE = 64
TOKENS_PER_STEP = MICRO_BATCH_SIZE * CONTEXT_LENGTH  # B x T

EXPECTED_RUNGS = [
    (160, 6, 5),
    (192, 6, 6),
    (224, 7, 7),
    (256, 8, 8),
    (288, 10, 9),
    (320, 12, 10),
    (384, 12, 12),
]


@dataclass(frozen=True)
class RunPlan:
    target_tokens: float
    steps: int
    realized_tokens: int
    realized_compute: int
    warmup_steps: int


def band_compute_budget(smallest_config: ModelConfig) -> float:
    # The smallest rung gets the most tokens, so it sets C
    return TOKEN_MARGIN * TRAIN_TOKENS * flops_per_token(smallest_config)


def plan_run(
    config: ModelConfig, compute_budget: float, tokens_per_step: int
) -> RunPlan:
    fpt = flops_per_token(config)
    target_tokens = compute_budget / fpt
    steps = math.ceil(target_tokens / tokens_per_step)
    realized_tokens = steps * tokens_per_step
    return RunPlan(
        target_tokens=target_tokens,
        steps=steps,
        realized_tokens=realized_tokens,
        realized_compute=realized_tokens * fpt,
        warmup_steps=max(math.ceil(0.05 * steps), 100),
    )


def main() -> None:
    ladder = build_ladder(head_size=32, aspect_ratio=32)
    assert ladder is not None, "Q=32, d/L=32 ladder is infeasible"
    rungs = [(r.model_width, r.model_depth, r.num_heads) for r in ladder]
    assert rungs == EXPECTED_RUNGS, f"ladder changed: {rungs}"

    configs = [make_config(*rung) for rung in rungs]
    C = band_compute_budget(configs[0])
    logger.info(
        f"C = {C:.4e} FLOPs, {TOKENS_PER_STEP:,} tokens per step "
        + f"({MICRO_BATCH_SIZE} x {CONTEXT_LENGTH})"
    )
    logger.info(
        f"{'d':>4} {'L':>3} {'H':>3} {'FLOPs/token':>12} {'target D':>13} "
        + f"{'steps':>7} {'warmup':>7} {'realised D':>13} {'realised C':>11} {'excess':>8}"
    )
    for (d, L, H), config in zip(rungs, configs):
        plan = plan_run(config, C, TOKENS_PER_STEP)
        excess = plan.realized_compute / C - 1  # over the target C
        logger.info(
            f"{d:>4} {L:>3} {H:>3} {flops_per_token(config):>12,} "
            + f"{plan.target_tokens:>13,.0f} {plan.steps:>7,} {plan.warmup_steps:>7,} "
            + f"{plan.realized_tokens:>13,} {plan.realized_compute:>11.4e} {excess:>8.4%}"
        )


if __name__ == "__main__":
    main()
