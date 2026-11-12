import math

from src.ladder import TOKEN_MARGIN, TRAIN_TOKENS, make_config
from src.util import flops_per_token


def main():
    rungs = [
        (160, 6, 5),
        (192, 6, 6),
        (224, 7, 7),
        (256, 8, 8),
        (288, 10, 9),
        (320, 12, 10),
        (384, 12, 12),
    ]
    configs = [make_config(r[0], r[1], r[2]) for r in rungs]
    flops_per_tokens = [flops_per_token(config) for config in configs]
    C = TOKEN_MARGIN * TRAIN_TOKENS * flops_per_tokens[0]
    target_tokens = [C / f for f in flops_per_tokens]
    tokens_per_step = 64 * 256  # B x T
    steps = [
        math.ceil(target_token / tokens_per_step) for target_token in target_tokens
    ]
    realized_tokens = [step * tokens_per_step for step in steps]
    warmup_steps = [math.ceil(max(0.05 * step, 100)) for step in steps]
    for i in range(len(rungs)):
        C_actual = realized_tokens[i] * flops_per_tokens[i]
        print(
            f"Rung = {rungs[i]},\nConfig = {configs[i]},\nFLOPs per token = {flops_per_tokens[i]},\nTarget tokens = {target_tokens[i]},\nSteps = {steps[i]},\nRealized tokens = {realized_tokens[i]},\nActual compute (excess over realized) = {C_actual} ({100 * (-1 + C_actual / C)}),\nWarmup floor = {warmup_steps[i]}\n"
        )


if __name__ == "__main__":
    main()
