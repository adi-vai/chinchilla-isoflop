# One-off experiment behind the vocab-size choice: characters per token on the
# validation split for BPE tokenizers of several sizes, each trained on a sample
# of the cleaned train split. Kept as the evidence for the choice.

import logging

from data.tinystories.prepare import build_tokenizer, load_clean_splits

logger = logging.getLogger(__name__)

VOCAB_SIZES = [500, 1_000, 2_000, 4_000, 8_000, 10_000]
SAMPLE_SIZE = 100_000

if __name__ == "__main__":
    cleaned, validation = load_clean_splits()

    sample_stories = cleaned.shuffle(seed=1969).select(range(SAMPLE_SIZE))["text"]
    validation_text = validation["text"]
    validation_chars = sum(len(text) for text in validation_text)

    for vocab_size in VOCAB_SIZES:
        tokenizer, trainer = build_tokenizer(vocab_size)
        tokenizer.train_from_iterator(sample_stories, trainer)
        validation_tokens = sum(
            len(encoding.ids) for encoding in tokenizer.encode_batch(validation_text)
        )
        chars_per_token = validation_chars / validation_tokens
        logger.info(
            f"Vocab size: {vocab_size:,}, characters per token: {chars_per_token:.2f}"
        )
