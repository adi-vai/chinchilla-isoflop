# saves tinystories to a binary for training

import logging
import os
from hashlib import sha256

import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

num_proc = 5  # 10 cores // 2

num_proc_load_dataset = num_proc
enc = tiktoken.get_encoding("gpt2")  # tokenizer? gpt2 or character encoding?


def text_digest(text: str) -> str:
    # Single definition, so the validation set and the train loop can't drift apart
    return sha256(text.encode("utf-8")).hexdigest()


if __name__ == "__main__":
    dataset = load_dataset("roneneldan/TinyStories", num_proc=num_proc_load_dataset)
    logger.info(f"Loaded dataset: {dataset}")

    train = dataset["train"]
    num_train = len(train)

    # Contamination: digests of every validation story
    val_digests = {text_digest(s) for s in dataset["validation"]["text"]}

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

    # Next, train a 2k tokenizer on cleaned train set
