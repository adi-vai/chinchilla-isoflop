import math


def get_lr(
    step: int, total_steps: int, warmup_steps: int, peak_lr: float, min_lr: float
) -> float:
    assert total_steps > warmup_steps + 1
    if step < warmup_steps:
        return peak_lr * (step + 1) / (warmup_steps + 1)
    if step >= total_steps:
        return min_lr
    else:
        return (peak_lr - min_lr) * 0.5 * (
            1
            + math.cos(
                math.pi * (step - warmup_steps) / (total_steps - warmup_steps - 1)
            )
        ) + min_lr
