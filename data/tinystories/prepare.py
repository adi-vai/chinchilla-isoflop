"""Prepare TinyStories for training: clean, tokenize, write token memmaps.

Pipeline:
1. Clean the train split: drop stories that also appear in validation (all
   copies) and exact duplicates. Validation is the official split, unchanged.
2. Train the frozen byte-level BPE tokenizer on all cleaned train stories.
3. Encode both splits (end-of-text id after every story) and write them as
   flat uint16 memmaps.

Outputs, next to this file: tokenizer.json (committed), train.bin and val.bin
(gitignored). Run from the repo root: `uv run python -m data.tinystories.prepare`.
"""

import logging
from hashlib import sha256
from pathlib import Path

import numpy as np
from datasets import Dataset, load_dataset
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Data source and machine
DATASET_ID = "roneneldan/TinyStories"
NUM_PROC = 5  # 10 cores // 2

# Tokenizer
VOCAB_SIZE = 2_000  # see vocab_sweep.py for the evidence; happens to match Bryant & Liu
END_OF_TEXT = "<|endoftext|>"  # separates stories in the token stream

# Outputs
OUTPUT_DIR = Path(__file__).resolve().parent
TOKEN_DTYPE = np.uint16  # ids are in [0, VOCAB_SIZE); uint8 tops out at 255


# --- 1. Cleaning -----------------------------------------------------------


def text_digest(text: str) -> str:
    # Single definition, so the validation set and the train loop can't drift apart
    return sha256(text.encode("utf-8")).hexdigest()


def load_clean_splits() -> tuple[Dataset, Dataset]:
    """Load TinyStories; return (cleaned train, validation)."""
    dataset = load_dataset(DATASET_ID, num_proc=NUM_PROC)
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


# --- 2. Tokenizer ----------------------------------------------------------


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


# --- 3. Encoding and writing -------------------------------------------------


def tokenize_split(
    tokenizer: Tokenizer, dataset: Dataset, end_of_text_id: int, desc: str
) -> Dataset:
    # Batched: each call gets a batch of stories as {"text": [...]}, and encode_batch
    # tokenizes them in parallel (in Rust), so no num_proc is needed here
    def tokenize_batch(
        batch: dict[str, list[str]],
    ) -> dict[str, list[list[int]] | list[int]]:
        all_ids = [
            encoding.ids + [end_of_text_id]
            for encoding in tokenizer.encode_batch(batch["text"])
        ]
        return {"ids": all_ids, "len": [len(ids) for ids in all_ids]}

    return dataset.map(tokenize_batch, batched=True, remove_columns=["text"], desc=desc)


def write_bin(dataset: Dataset, filename: Path) -> int:
    # nanoGPT's pattern: one flat memmap of all token ids, written in contiguous shards
    arr_len = int(np.sum(dataset["len"], dtype=np.uint64))
    arr = np.memmap(filename, dtype=TOKEN_DTYPE, mode="w+", shape=(arr_len,))
    total_batches = min(1024, len(dataset))
    idx = 0
    for batch_idx in tqdm(range(total_batches), desc=f"writing {filename.name}"):
        batch = dataset.shard(
            num_shards=total_batches, index=batch_idx, contiguous=True
        ).with_format("numpy")
        arr_batch = np.concatenate(batch["ids"])
        arr[idx : idx + len(arr_batch)] = arr_batch
        idx += len(arr_batch)
    arr.flush()
    assert idx == arr_len, f"wrote {idx:,} tokens, expected {arr_len:,}"
    return arr_len


# --- Pipeline ---------------------------------------------------------------


def main() -> None:
    cleaned, validation = load_clean_splits()

    # Frozen tokenizer: trained on all cleaned train stories (decided 22 Oct)
    tokenizer, trainer = build_tokenizer(VOCAB_SIZE)
    tokenizer.train_from_iterator(cleaned["text"], trainer)
    tokenizer_path = OUTPUT_DIR / "tokenizer.json"
    tokenizer.save(str(tokenizer_path))
    logger.info(
        f"Saved tokenizer ({tokenizer.get_vocab_size():,} tokens) to {tokenizer_path}"
    )

    end_of_text_id = tokenizer.token_to_id(END_OF_TEXT)
    assert end_of_text_id is not None, f"{END_OF_TEXT} missing from the vocabulary"
    # Every id is in [0, vocab_size), so the dtype is safe iff the vocab fits in it
    assert tokenizer.get_vocab_size() <= np.iinfo(TOKEN_DTYPE).max + 1, (
        f"vocab too big for {TOKEN_DTYPE.__name__}"
    )

    splits = {"train": cleaned, "val": validation}
    for split, dataset in splits.items():
        tokenized = tokenize_split(
            tokenizer, dataset, end_of_text_id, f"tokenizing {split}"
        )
        num_tokens = write_bin(tokenized, OUTPUT_DIR / f"{split}.bin")
        logger.info(
            f"{split}: {len(dataset):,} stories -> {num_tokens:,} tokens ({num_tokens / len(dataset):.1f} per story)"
        )


if __name__ == "__main__":
    main()
