"""Token windows from the TinyStories memmaps (NOTES.md, "Training data order").

Window i is tokens[i·T : i·T + T + 1], stride T. Inputs are window[:-1] and targets
window[1:] (the next token at each position), so consecutive windows share one
boundary token and every token is a target exactly once.
"""

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import torch

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "tinystories"
TOKEN_DTYPE = np.uint16  # must match data.tinystories.prepare.TOKEN_DTYPE
DATA_SEED = 0  # one data order for every rung, separate from the init seed


def load_tokens(split: str) -> np.memmap:
    return np.memmap(DATA_DIR / f"{split}.bin", dtype=TOKEN_DTYPE, mode="r")


def num_windows(tokens: np.ndarray, context_length: int) -> int:
    return (len(tokens) - 1) // context_length


def get_windows(
    tokens: np.ndarray, starts: np.ndarray, context_length: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    # (B, T+1) gather from the memmap; only the touched pages are read
    index = starts[:, None] + np.arange(context_length + 1)
    batch = torch.from_numpy(tokens[index].astype(np.int64))
    if device.type == "cuda":
        batch = batch.pin_memory().to(device, non_blocking=True)
    else:
        batch = batch.to(device)
    # Two views of one tensor, offset by one: (inputs, targets)
    return batch[:, :-1], batch[:, 1:]


class TrainBatches:
    """Micro-batches in one seeded permutation of the windows, without replacement."""

    def __init__(
        self,
        context_length: int,
        micro_batch_size: int,
        data_seed: int,
        device: torch.device,
    ) -> None:
        self.tokens = load_tokens("train")
        self.context_length = context_length
        self.micro_batch_size = micro_batch_size
        self.device = device
        self.num_windows = num_windows(self.tokens, context_length)
        self.order = np.random.default_rng(data_seed).permutation(self.num_windows)
        self.position = 0  # windows consumed so far

    def next_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        end = self.position + self.micro_batch_size
        if end > self.num_windows:
            raise RuntimeError(
                f"out of training windows: need {end:,}, have {self.num_windows:,}"
            )
        starts = self.order[self.position : end] * self.context_length
        self.position = end
        return get_windows(self.tokens, starts, self.context_length, self.device)


def val_batches(
    context_length: int, batch_size: int, device: torch.device
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    """Every validation window once, in order; the last batch may be smaller."""
    tokens = load_tokens("val")
    total = num_windows(tokens, context_length)
    for first in range(0, total, batch_size):
        window_ids = np.arange(first, min(first + batch_size, total))
        yield get_windows(tokens, window_ids * context_length, context_length, device)
