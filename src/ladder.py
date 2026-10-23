"""Parameter counts for candidate model shapes (the N ladder for pilot band 1).

Step 1, calibration: reproduce Bryant & Liu's published parameter counts from
their (d, L, H) configs, to confirm our counting before trusting it for our own
ladder. With weight tying and no lm_head bias (as in nanoGPT), our total N must
match their N to the precision they publish it at.

Run from the repo root: `uv run python -m src.ladder`.
"""

import logging

from src.model import ModelConfig, TransformerLanguageModel

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


def main() -> None:
    calibrate_against_bryant_liu()


if __name__ == "__main__":
    main()
