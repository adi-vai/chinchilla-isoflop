# Working notes

Durable project state: what's decided, what's open, what's next. Complements
`CLAUDE.md` (which holds the invariants and the phase plan) — this file tracks
the moving parts. Last updated 2026-07-26.

## Where things stand

Phase 2 (transformer from scratch) is done, in `learning/better_bigram.py`:
multi-head causal self-attention, residual connections, pre-layer-norm blocks,
dropout, `ModelConfig` / `TrainConfig` dataclasses.

Phase 3 (harness) is starting. Repo has been split into `learning/` (frozen
practice snapshots) and `src/` (the experiment). `src/` is currently empty apart
from `__init__.py`.

**Immediate next step:** `src/model.py` — port the transformer out of
`learning/better_bigram.py`, with batched QKV attention.

## Decisions made

**`learning/` is frozen; `src/` is a clean rewrite.** No imports from
`learning/` into `src/`, and no maintaining both. The practice code stays as a
snapshot of what was understood at the time. Considered and rejected: promoting
the model classes into a shared module both trees import. Rejected because it
couples a teaching artifact to the experiment and means every later change has
to keep the old scripts running.

**`src/` is a package, invoked as `uv run python -m src.<module>` from the repo
root.** This puts the repo root on `sys.path`, so `from src.model import GPT`
resolves identically from a script, a notebook, or `plots/`. Running
`python src/train.py` instead would put only `src/` on the path, forcing
`from model import GPT`, which breaks the first time anything outside `src/`
imports the same module.

**Paths resolve from `__file__`, not the current working directory.**
`learning/` inherits the old CWD-relative `DATA_PATH` and so must be run from
the repo root. `src/` won't have that constraint — a sweep harness can get
launched from arbitrary directories on a rented GPU box.

**Port the model before reading further; read before writing the harness.**
nanoGPT's `model.py` is the only reading item that informs model code.
`prepare.py` and `train.py` inform harness structure (data pipeline, per-run LR
schedule, loop shape), so writing the harness before reading them means
designing it twice.

**`num_params()` lives on the model.** N is an axis of the whole experiment;
nanoGPT has `get_num_params()` for the same reason. Wanted before the harness
exists.

## Known issues

**Attention logits are scaled by `embedding_dimension**-0.5` instead of
`head_size**-0.5`** (`learning/better_bigram.py`, in `SelfAttentionHead.forward`).
The two coincide in the single-head case the code grew from, and the line wasn't
updated when it became multi-head.

The error factor is exactly `sqrt(number_of_heads)` — so it isn't a uniform
mis-scaling, it varies with head count. That matters here specifically: IsoFLOP
compares loss across configs, so a bug that's a function of a swept
hyperparameter biases runs against each other rather than shifting them all
equally.

Left in place in `learning/` on purpose (documented in `learning/README.md`);
to be fixed in `src/model.py`.

## Queued, not yet done

- **`data/` gitignore rule.** `data/input.txt` (1.1MB tinyshakespeare) is
  currently committed, which is fine. When the corpus becomes TinyStories or an
  OpenWebText subset it must be ignored — reproducibility comes from `uv.lock`
  plus a documented download URL and checksum, not from committing corpus bytes.
  Do this at the data-prep step.
- **`results/`** doesn't exist yet; created when the harness first logs.

## Verification approach for the model port

Not "does it train" — prove behavioural equivalence:

1. Seed, build the old per-head module and the new batched module.
2. Copy the old per-head `q`/`k`/`v` weights into the corresponding row-blocks
   of the new fused `Linear(C, 3*C)`.
3. Same input through both, assert `allclose`.

Working out which block of the fused matrix maps to which head is the part
that's easy to get subtly wrong, which is why it's worth asserting rather than
eyeballing.

The scaling fix and the port must be **separate steps** — the equivalence test
only means something if the port preserves the (wrong) `embedding_dimension**-0.5`
factor. Fix the scaling as a deliberate follow-up commit.

## nanoGPT reading list

Source: `karpathy/nanoGPT`. Reading it for the pieces not yet built — data
pipeline, per-run cosine LR schedule, FLOP/param counting, checkpointing.

**Done — 1. `model.py`.** Confirmed the `3 *` in `CausalSelfAttention` is the
fused QKV projection. nanoGPT batches all heads through one `Linear` + reshape
where `learning/better_bigram.py` uses a `ModuleList` of independent heads.
Also surfaced the scaling bug above (nanoGPT reads the head dim off the tensor,
`1.0 / math.sqrt(k.size(-1))`, so it can't drift).

Open question from this item, still unanswered: **where does the `6` in
`C ≈ 6ND` come from in `estimate_mfu()`'s exact formula `6*N + 12*L*H*Q*T`, and
what does the approximation drop?**

**2. `data/shakespeare_char/prepare.py`** — tokenize once, write
`train.bin`/`val.bin` as `uint16` memmaps plus `meta.pkl` (vocab_size, stoi,
itos), rather than re-tokenizing in-process each run. Matters because an IsoFLOP
sweep pushes one corpus through many (N, D) pairs; memory-mapping off disk
instead of holding one big tensor is what makes that scale.
*Question: what do those files look like for the chosen corpus, and where do
they live relative to `results/`?*

**3. `train.py`** — dense, read in three passes:
- (a) config block + `get_batch()`, against the existing `ModelConfig`/`TrainConfig` split.
- (b) `get_lr(it)` — cosine with linear warmup. This is the mechanism behind
  CLAUDE.md's hard constraint that the schedule decay to its minimum at each
  run's *own* token count. nanoGPT's `lr_decay_iters` is one global for the whole
  script; here it has to be computed per-run from that run's D.
- (c) the loop: gradient accumulation, eval, checkpointing on best val loss.
  Skip DDP (multi-GPU is cut-material per CLAUDE.md).

Out of scope: `configurator.py`'s `exec()`-based `sys.argv` override. It mutates
`globals()` directly and needs pre-declared keys — a deliberate hack for
single-file simplicity. The dataclass setup already in place is cleaner; worth
recognising so `config/train_*.py` references elsewhere parse, not worth adopting.

**4. `scaling_laws.ipynb`** — highest-value file for this project. Implements
`gpt_params()` and `chinchilla_flops()` and validates them against Chinchilla's
published Table 4 (the `6ND` approximation agrees with an exact per-component
count to within roughly 3–10%). Usable as an answer key for whatever N/D/C code
the harness ends up with.
*Scope warning: it also implements the Approach 3 parametric fit
(`L = E + A/N^α + B/D^β`). CLAUDE.md gates that as a stretch goal behind a
post-IsoFLOP checkpoint — read the FLOP/param cells, skip the curve fitting.*

**5. Optional** — `transformer_sizing.ipynb` (sizing/memory tables, maybe useful
when picking the N grid), `sample.py` (already covered by the existing
`generate`), `bench.py` (skip unless a real perf question shows up).
