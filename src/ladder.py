"""Parameter counts for candidate model shapes (the N ladder for pilot band 1).

Step 1, calibration: reproduce Bryant & Liu's published parameter counts from
their (d, L, H) configs, to confirm our counting before trusting it for our own
ladder. With weight tying and no lm_head bias (as in nanoGPT), our total N must
match their N to the precision they publish it at.

Step 2, ladder search: for each shape rule (head size, target d/L), pick the
config closest in log N to each of 7 geometrically spaced targets, and rank the
rules by their worst rung. Chosen: Q=32, d/L=32 (NOTES.md).

Step 3, coupling check: derive C from the chosen ladder's bottom rung and check
the heuristic optimum sqrt(C/120) still sits near the middle rung.

Run from the repo root: `uv run python -m src.ladder`.
"""

import logging
import math
from dataclasses import dataclass

import numpy as np

from src.model import ModelConfig, TransformerLanguageModel
from src.util import compute_params, flops_per_token, total_params

logger = logging.getLogger(__name__)

VOCAB_SIZE = 2_000  # fixed by data.tinystories.prepare
CONTEXT_LENGTH = 256  # fixed across all runs (NOTES.md, decided 23 Oct)

# Bryant & Liu (arXiv 2605.09189):
# (model_width, model_depth, num_heads, published N, significant figures published).
# All their configs use a 4x MLP, which matches our FeedForward.
BRYANT_LIU = [
    (64, 4, 4, 344e3, 3),  # "344K"
    (128, 4, 4, 1.1e6, 2),  # "1.1M"
    (192, 6, 6, 3.1e6, 2),  # "3.1M"
    (320, 8, 8, 10.6e6, 3),  # "10.6M"
    (480, 10, 8, 28.8e6, 3),  # "28.8M"
]


def make_config(model_width: int, model_depth: int, num_heads: int) -> ModelConfig:
    return ModelConfig(
        vocab_size=VOCAB_SIZE,
        context_length=CONTEXT_LENGTH,
        model_width=model_width,
        model_depth=model_depth,
        num_heads=num_heads,
        dropout_rate=0.0,  # the sweep runs without dropout (NOTES.md)
    )


def to_sig_figs(x: float, sig_figs: int) -> float:
    # Round to the precision Bryant & Liu publish at, e.g. 343,680 -> 344,000 at 3
    return float(f"{x:.{sig_figs}g}")


def calibrate_against_bryant_liu() -> None:
    logger.info(
        f"{'d':>4} {'L':>3} {'H':>3} {'total N':>12} {'compute N':>12} {'B&L N':>12}  match"
    )
    for model_width, model_depth, num_heads, published, sig_figs in BRYANT_LIU:
        model = TransformerLanguageModel(
            make_config(model_width, model_depth, num_heads)
        )
        total = model.get_total_params()
        compute = model.get_compute_params()
        match = to_sig_figs(total, sig_figs) == published
        logger.info(
            f"{model_width:>4} {model_depth:>3} {num_heads:>3} {total:>12,} {compute:>12,} {published:>12,.0f}  {match}"
        )
        assert match, (
            f"total N {total:,} doesn't round to Bryant & Liu's {published:,.0f}"
        )


def check_analytic_param_count() -> None:
    for model_width, model_depth, num_heads, _, _ in BRYANT_LIU:
        model_config = make_config(model_width, model_depth, num_heads)
        model = TransformerLanguageModel(model_config)
        compute, total = compute_params(model_config), total_params(model_config)
        assert compute == model.get_compute_params(), (
            f"analytic compute N {compute:,} != model's {model.get_compute_params():,}"
        )
        assert total == model.get_total_params(), (
            f"analytic total N {total:,} != model's {model.get_total_params():,}"
        )
        logger.info(
            f"d={model_width} L={model_depth} H={num_heads}: compute N {compute:,}, total N {total:,} match the model"
        )


# Ladder search: a shape rule (head size Q, target aspect ratio d/L) is fixed for the
# whole ladder so shape doesn't drift between rungs and confound N; within a rule
# each rung takes the config closest in log N to its target.
N_OPT = 7.1e6  # Chinchilla heuristic, total N (NOTES.md open item 1)
NUM_RUNGS = 7
# 10x spread centred on N_OPT in log N; targets are total N, as Chinchilla counts N
TARGETS = np.geomspace(N_OPT / 10**0.5, N_OPT * 10**0.5, NUM_RUNGS)
HEAD_SIZES = (32, 64)  # B&L never reach GPT-2's 64; 16 caps score-matrix rank too low
NUM_HEADS = range(4, 13)  # 4 = B&L's minimum; 12 is arbitrary and can be pushed
ASPECT_RATIOS = (20, 32, 64, 128, 200)  # edges and interior of Kaplan's flat band
ASPECT_RATIO_TOLERANCE = 1.2  # realised d/L may lie in [r / tol, r · tol]
MIN_DEPTH = 4  # B&L's smallest depth


@dataclass(frozen=True)
class Rung:
    model_width: int
    model_depth: int
    num_heads: int
    params: int  # total N
    target: float

    @property
    def log_miss(self) -> float:
        return float(np.log(self.params / self.target))


def candidates(
    head_size: int,
    aspect_ratio: int,
) -> list[tuple[int, int, int]]:
    # Every (d, L, H) with d = Q·H, L ≥ MIN_DEPTH and realised d/L within tolerance of r
    configs: list[tuple[int, int, int]] = []
    for num_heads in NUM_HEADS:
        model_width = head_size * num_heads
        for model_depth in range(
            max(
                MIN_DEPTH,
                math.ceil(model_width / (ASPECT_RATIO_TOLERANCE * aspect_ratio)),
            ),
            math.floor(ASPECT_RATIO_TOLERANCE * model_width / aspect_ratio) + 1,
        ):
            configs.append((model_width, model_depth, num_heads))
    return configs


def build_ladder(head_size: int, aspect_ratio: int) -> list[Rung] | None:
    configs = candidates(head_size, aspect_ratio)
    if not configs:
        return None
    sizes = [total_params(make_config(*config)) for config in configs]
    ladder: list[Rung] = []
    for target in TARGETS:
        # Closest in log N, i.e. smallest relative miss
        best = min(range(len(configs)), key=lambda i: abs(np.log(sizes[i] / target)))
        ladder.append(Rung(*configs[best], params=sizes[best], target=float(target)))
    # Two rungs on one config (or out of order) means the rule can't resolve the grid
    params = [rung.params for rung in ladder]
    if any(a >= b for a, b in zip(params, params[1:])):
        return None
    return ladder


def search_ladders() -> None:
    logger.info(f"Targets (total N): {', '.join(f'{t:,.0f}' for t in TARGETS)}")
    results: list[tuple[int, int, list[Rung]]] = []
    for head_size in HEAD_SIZES:
        for aspect_ratio in ASPECT_RATIOS:
            ladder = build_ladder(head_size, aspect_ratio)
            if ladder is None:
                logger.info(f"Q={head_size} d/L={aspect_ratio}: infeasible")
                continue
            results.append((head_size, aspect_ratio, ladder))

    # Rank rules by worst rung (L-inf); RMS (L2) logged alongside to compare
    results.sort(key=lambda r: max(abs(rung.log_miss) for rung in r[2]))
    for head_size, aspect_ratio, ladder in results:
        misses = np.array([rung.log_miss for rung in ladder])
        worst = np.expm1(np.abs(misses).max())
        rms = np.expm1(np.sqrt(np.mean(misses**2)))
        logger.info(
            f"Q={head_size} d/L={aspect_ratio}: worst miss {worst:.1%}, RMS miss {rms:.1%}"
        )
        logger.info(
            f"  {'d':>4} {'L':>3} {'H':>3} {'d/L':>6} {'total N':>12} {'target':>12} {'miss':>7}"
        )
        for rung in ladder:
            logger.info(
                f"  {rung.model_width:>4} {rung.model_depth:>3} {rung.num_heads:>3} "
                + f"{rung.model_width / rung.model_depth:>6.1f} {rung.params:>12,} "
                + f"{rung.target:>12,.0f} {np.expm1(rung.log_miss):>7.1%}"
            )


TRAIN_TOKENS = 447_014_068  # train.bin, from data.tinystories.prepare
TOKEN_MARGIN = 0.9  # slack for rounding iterations up to whole batches


def coupling_check() -> None:
    # C comes from the bottom rung (it gets the most tokens); the heuristic optimum
    # under C = 6·N·20N = 120·N² should then land within half a rung of the middle
    c = TOKEN_MARGIN * TRAIN_TOKENS * flops_per_token(make_config(160, 6, 5))
    est_n_opt = math.sqrt(c / 120)
    middle_run = total_params(make_config(256, 8, 8))
    logger.info(
        f"C = {c:.3g}: estimated n_opt {est_n_opt:,.0f} vs middle run's {middle_run:,}, "
        + f"rel error of {est_n_opt / middle_run - 1:.1%}"
    )


def main() -> None:
    logger.info("Calibrating against Bryant & Liu")
    calibrate_against_bryant_liu()
    logger.info("Checking analytic parameter counts against the model")
    check_analytic_param_count()
    logger.info("Searching ladders")
    search_ladders()
    logger.info("Coupling check")
    coupling_check()


if __name__ == "__main__":
    main()
