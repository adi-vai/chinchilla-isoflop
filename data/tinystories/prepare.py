# saves tinystories to a binary for training

import logging
import os
from hashlib import sha256

import numpy as np
from datasets import Dataset, load_dataset
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

num_proc = 5  # 10 cores // 2

num_proc_load_dataset = num_proc

END_OF_TEXT = "<|endoftext|>"  # separates stories in the token stream


def text_digest(text: str) -> str:
    # Single definition, so the validation set and the train loop can't drift apart
    return sha256(text.encode("utf-8")).hexdigest()


def build_tokenizer(vocab_size: int) -> tuple[Tokenizer, BpeTrainer]:
    # Shared by the vocab sweep and the frozen tokenizer, so the sweep's evidence
    # applies to exactly the configuration that gets frozen
    tokenizer = Tokenizer(BPE())
    tokenizer.pre_tokenizer = ByteLevel()
    trainer = BpeTrainer(
        special_tokens=[END_OF_TEXT],
        vocab_size=vocab_size,
        initial_alphabet=ByteLevel.alphabet(),
    )
    return tokenizer, trainer


def load_clean_splits() -> tuple[Dataset, Dataset]:
    """Load TinyStories; return (cleaned train, validation).

    Train is cleaned by dropping stories that also appear in validation (all
    copies) and exact duplicates (first occurrence kept). Validation is the
    official split, unchanged.
    """
    dataset = load_dataset("roneneldan/TinyStories", num_proc=num_proc_load_dataset)
    logger.info(f"Loaded dataset: {dataset}")

    train = dataset["train"]
    validation = dataset["validation"]
    num_train = len(train)

    # Contamination: digests of every validation story
    val_digests = {text_digest(s) for s in validation["text"]}

    # Exact dedup + drop train stories that appear in validation (all copies)
    seen: set[str] = set()
    keep_idx: list[int] = []
    duplicate_count = 0
    contaminated_count = 0
    contaminated_digests: set[str] = set()  # unique validation stories found in train
    for i, text in enumerate(train["text"]):
        digest = text_digest(text)
        if digest in val_digests:
            contaminated_count += 1
            contaminated_digests.add(digest)
            continue
        if digest in seen:
            duplicate_count += 1
            continue
        seen.add(digest)
        keep_idx.append(i)

    # select() on the same object the indices came from
    cleaned = train.select(keep_idx)

    pct_contaminated_train = contaminated_count / num_train
    pct_contaminated_val = len(contaminated_digests) / len(val_digests)
    pct_duplicates = duplicate_count / num_train
    pct_kept = len(cleaned) / num_train
    logger.info(f"Train stories before cleaning: {num_train:,}")
    logger.info(
        f"Contaminated (also in validation) removed: {contaminated_count:,} ({pct_contaminated_train:.2%} of train; {pct_contaminated_val:.2%} of validation)"
    )
    logger.info(f"Exact duplicates removed: {duplicate_count:,} ({pct_duplicates:.2%})")
    logger.info(f"Train stories after cleaning: {len(cleaned):,} ({pct_kept:.2%} kept)")

    return cleaned, validation


if __name__ == "__main__":
    cleaned, validation = load_clean_splits()

    # Next: train the frozen tokenizer (vocab 2,000) on all cleaned train stories
    # (decided 22 Oct; sweep evidence in vocab_sweep.py), then encode both splits
    # into train.bin / val.bin
