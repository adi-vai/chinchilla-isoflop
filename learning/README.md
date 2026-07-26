# learning/

Practice code from working up to the experiment — kept deliberately, in rough
chronological order. These are **frozen snapshots**, not maintained code. The
experimental harness lives in `src/`.

| File | What it was for |
|---|---|
| `hello.py` | PyTorch install smoke test; first tensor. |
| `func_approx.py` | Phase 1: hand-written training loop. Two-layer MLP (tanh) fitting `sin(x)`, MSE loss, Adam. No `DataLoader`, no abstractions — just forward / loss / `backward()` / `step()`. |
| `bigram.py` | Char-level bigram LM on tinyshakespeare. Embedding table *is* the logit table; cross-entropy over `(B*T, C)`; autoregressive sampling. Also the data download / vocab / encode-decode plumbing. |
| `better_bigram.py` | Phase 2: grew the bigram into a full decoder-only transformer — causal self-attention, multi-head, residual connections, pre-layer-norm blocks, dropout, and a `ModelConfig` / `TrainConfig` dataclass split. |

## Known issues (left in place on purpose)

- **`better_bigram.py` scales attention logits by `embedding_dimension**-0.5`
  instead of `head_size**-0.5`.** The two coincide in the single-head case this
  code grew out of, and the line wasn't updated when it became multi-head. The
  effect is over-damped attention logits (1/19.6 rather than 1/8 at
  `embedding_dimension=384`, `number_of_heads=6`). Fixed in `src/model.py`; left
  here so the snapshot stays honest about what it was.
- **`DATA_PATH` is resolved relative to the current working directory**, not to
  the script. Run these from the repo root (`uv run python learning/bigram.py`);
  running from inside `learning/` will download a second copy of the corpus into
  `learning/data/`. `src/` derives paths from `__file__` instead.
