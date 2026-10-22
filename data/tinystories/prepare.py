# saves tinystories to a binary for training

import logging
import sys
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

NUM_PROC = 5  # 10 cores // 2

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
    dataset = load_dataset("roneneldan/TinyStories", num_proc=NUM_PROC)
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


VOCAB_SIZE = 2_000  # see vocab_sweep.py for the evidence; happens to match Bryant & Liu
OUTPUT_DIR = Path(__file__).resolve().parent


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


def write_bin(dataset: Dataset, filename: Path, dtype: type[np.unsignedinteger]) -> int:
    # nanoGPT's pattern: one flat memmap of all token ids, written in contiguous shards
    arr_len = int(np.sum(dataset["len"], dtype=np.uint64))
    arr = np.memmap(filename, dtype=dtype, mode="w+", shape=(arr_len,))
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


if __name__ == "__main__":
    # Optional smoke test: `... prepare <n>` uses the first n stories of each split
    # and writes to OUTPUT_DIR/smoke/, so it can never overwrite the real artifacts
    subset = int(sys.argv[1]) if len(sys.argv) > 1 else None
    output_dir = OUTPUT_DIR / "smoke" if subset is not None else OUTPUT_DIR
    output_dir.mkdir(exist_ok=True)

    cleaned, validation = load_clean_splits()
    if subset is not None:
        cleaned = cleaned.select(range(subset))
        validation = validation.select(range(min(subset, len(validation))))
        logger.info(f"SMOKE TEST: using {subset:,} stories per split")

    # Frozen tokenizer: trained on all cleaned train stories (decided 22 Oct)
    tokenizer, trainer = build_tokenizer(VOCAB_SIZE)
    tokenizer.train_from_iterator(cleaned["text"], trainer)
    tokenizer_path = output_dir / "tokenizer.json"
    tokenizer.save(str(tokenizer_path))
    logger.info(
        f"Saved tokenizer ({tokenizer.get_vocab_size():,} tokens) to {tokenizer_path}"
    )

    end_of_text_id = tokenizer.token_to_id(END_OF_TEXT)
    assert end_of_text_id is not None, f"{END_OF_TEXT} missing from the vocabulary"

    # Every id is in [0, vocab_size), so uint16 is safe iff the vocab fits in it
    dtype = np.uint16  # uint8 tops out at 255, too small for 2,000
    assert tokenizer.get_vocab_size() <= np.iinfo(dtype).max + 1, (
        "vocab too big for uint16"
    )

    splits = {"train": cleaned, "val": validation}
    for split, dataset in splits.items():
        tokenized = tokenize_split(
            tokenizer, dataset, end_of_text_id, f"tokenizing {split}"
        )
        num_tokens = write_bin(tokenized, output_dir / f"{split}.bin", dtype)
        logger.info(
            f"{split}: {len(dataset):,} stories -> {num_tokens:,} tokens ({num_tokens / len(dataset):.1f} per story)"
        )
