# Working notes

Durable project state: what's decided, what's open, what's next. Complements
`CLAUDE.md` (which holds the invariants and the phase plan) — this file tracks
the moving parts. Last updated 2026-10-06.

## Glossary

| Sym | Meaning | Code name | nanoGPT |
|---|---|---|---|
| `N` | parameter count | `num_params()` | `get_num_params()` |
| `D` | training tokens, whole run | — | — |
| `C` | training compute, FLOPs | — | — |
| `L` | layers (depth) | `model_depth` | `n_layer` |
| `H` | attention heads | `num_heads` | `n_head` |
| `Q` | head size, per-head width | `head_size` | `n_embd // n_head` |
| `d` | residual-stream width | `model_width` | `n_embd` |
| `T` | sequence length (≤ context length) | `sequence_length` / `context_length` | `block_size` |
| `B` | batch size | `batch_size` | `batch_size` |

Identities doing the work:
- `H · Q = d` — heads partition the width.
- `N ≈ 12·L·d²` for the transformer body — `4d²` attention (qkv + output
  projection) plus `8d²` MLP (4× expansion) per layer.

Easy confusions:
- `T` vs `D`: `T` is a model config (tokens per sequence); `D` is total tokens
  seen in a run, `≈ B · T · iterations`.
- `T` in the FLOP formulas is the full context length (sequences assumed
  full). In code, the runtime sequence length can be shorter (e.g. during
  `generate`), hence `sequence_length` vs `context_length`.
- `d` is called "width", not "embedding dimension": it's the width of the
  residual stream every block reads and writes, not just the embedding table's
  output. Depth (`L`) and width (`d`) are the two independent size knobs.
- `B` never appears in FLOPs-per-token; batching is throughput, not per-token cost.

## Where things stand

Phases 1–2 done (`learning/`). Phase 3 (harness) in progress.

`src/model.py` exists: a port of `learning/better_bigram.py` with full type
annotations and batched attention. History:
- `59be118` — annotated, shapes relabelled, behaviour unchanged. **Reference
  version for the equivalence test.**
- `0d5e5f7` — q/k/v fused into one projection within each head.
- `5886224` — q/k/v fused across heads: one `Linear(d, 3d)` in
  `MultiHeadedSelfAttention`, reshaped to `(B, H, T, Q)`; `SelfAttentionHead`
  removed. **Verified equivalent** to `59be118` (see Equivalence test below).
- `24afd11` — attention scaling fixed to `head_size**-0.5`.

The model and training loop are still in one file; the split into
`src/model.py` and `src/train.py` comes with the harness.

**Next steps, in order:**
1. Paths from `__file__` (with the `train.py` split).
2. ~~Parameter counts~~ — done: `get_total_params()`, `get_compute_params()`.

## Order of work for the rest of phase 3

model port → read nanoGPT items 2–4 → settle sweep design and corpus → build
harness (`src/train.py`) → local dress rehearsal → rent GPU for the real sweep.

**Dress rehearsal:** a complete miniature IsoFLOP sweep on the laptop — 3–4 tiny
models at one very small `C`, through to a CSV and a plotted U-curve. The point is
the pipeline, not the result: per-run schedule length, realised-C logging, CSV
columns and the plot script all get exercised before paying for GPU time.

## Decisions made

**`learning/` is frozen; `src/` is a clean rewrite.** No imports from
`learning/` into `src/`, and no maintaining both. Considered and rejected:
promoting the model classes into a shared module both trees import — couples a
teaching artifact to the experiment.

**`src/` is a package, invoked as `uv run python -m src.<module>` from the repo
root.** Puts the repo root on `sys.path`, so `from src.model import TransformerLanguageModel`
resolves the same from a script, a notebook, or `plots/`.

**Paths resolve from `__file__`, not the current working directory.** A sweep
harness may be launched from anywhere on a rented GPU box.

**Compute exact FLOPs, not `6ND`.** Per token:

```
flops_per_token = 6·N + 12·L·H·Q·T
```

- `6N`: every parameter costs 2 FLOPs forward (multiply + add) and 4 backward.
- `12·L·H·Q·T`: the attention matmuls `QKᵀ` and `attn @ V`, which have no
  parameters, so `6N` cannot see them. Per head per layer, each costs `2·T²·Q`
  for a whole sequence; × H × L, ÷ T to get per token, × 3 for forward+backward.
  Since `H·Q = d`, this is `12·L·T·d`.

The dropped term as a fraction of `6N ≈ 72·L·d²` is `T / (6d)` — an aspect
ratio. At `T=256`: ~11% at `d=384`, ~33% at `d=128`. It grows as models get
smaller, so `6ND` would systematically under-count compute for the small-N
runs, skewing the U-curve whose minimum is the result. Chinchilla could use
`6ND` because at `d≈4096, T=2048` the term is ~8% and roughly flat across
their range; these models are too small for that.

**Token budget and schedule per run:**

```
D          = C / flops_per_token
iterations = D / (B · T)
```

Cosine LR decays to its minimum at exactly that run's `iterations` (CLAUDE.md
constraint). `iterations` must be rounded, so **recompute and log the realised C**
from the rounded count — the IsoFLOP curve plots what was spent, not what was asked for.

**Single epoch.** Chinchilla's laws assume fresh tokens; repeating data makes
losses incomparable across runs. The largest `D` in the sweep (largest `C`,
smallest `N`) sets the minimum corpus size. Compute it before choosing a corpus —
tinyshakespeare's 1.1MB will be nowhere near enough.

**Naming: `TransformerLanguageModel`, `model_width`, `model_depth`, `num_heads`.** Not `GPT`: although
generative, pre-trained and transformer are all literally accurate (this *is*
pre-training — next-token prediction from random init, no fine-tuning after),
`GPT` reads as a reimplementation of OpenAI's model family. nanoGPT can use it
because it is a GPT-2 reproduction; this isn't. Width/depth names make the two
size knobs read as a pair.

**Port the model before reading further; read before writing the harness.**
Only nanoGPT's `model.py` informs the model code; `prepare.py` and `train.py`
inform the harness.

**Two parameter counts, both on the model and both logged per run.**
- `get_total_params()`: every parameter.
- `get_compute_params()`: total minus the token and position embedding tables.
  This is the `N` in `6N`. Embedding lookups do no matmul work; `lm_head` does and
  is counted. (nanoGPT counts its token embedding because it is weight-tied to
  `lm_head`; this model doesn't tie weights, so both tables are excluded.)

Which count goes on the IsoFLOP plot's N axis is decided at plotting time —
both are logged, so nothing needs rerunning. Relevant: Kaplan et al. counted
non-embedding parameters, Chinchilla all parameters; Pearce & Song (2024),
*Reconciling Kaplan and Chinchilla Scaling Laws*, argue this explains much of
the gap between their exponents, most strongly at small scale.

**Dropout is 0.0 for the sweep.** Single epoch means no repeated data, so
nothing to overfit; dropout would only add noise and slow learning per token,
depressing every point on the curve. The modules and `dropout_rate` stay (0.0
is a pass-through) so tinyshakespeare experiments can keep 0.2. Check on the
single-epoch guarantee: train and val loss should track closely; val clearly
above train means data is being repeated somewhere.

**Type checking: basedpyright, `recommended` mode, Any-reporting rules off**
(config in `pyproject.toml`). `nn.Module.__call__` returns `Any`, so
`forward` annotations are documentation, not guarantees — shape asserts and
shape comments carry the real information. `register_buffer` attributes need a
class-level declaration plus a targeted
`# pyright: ignore[reportUninitializedInstanceVariable]`.

## Known issues

**Attention logits are scaled by `embedding_dimension**-0.5` instead of
`head_size**-0.5`** — in `learning/better_bigram.py` (left there on purpose).
Fixed in `src/model.py` in `24afd11`, after the equivalence test passed.

Why it's wrong: the scaling fixes the variance of a dot product. A score is
`Σ qᵢkᵢ` over `head_size` terms; with unit-variance components its variance is
`head_size`, so dividing by `√head_size` restores variance 1. `embedding_dimension`
isn't in that dot product. The two coincide only with a single head.

Effect: scores have std ≈ 0.41 instead of 1 at `d=384, H=6`, so attention starts
too uniform. The error factor is exactly `√num_heads`, so it varies with a
swept hyperparameter and biases runs against each other. The model partly
compensates by learning larger `W_q`, `W_k`, which is why it still trains.

## Equivalence test (done)

**Result:** the batched attention in `5886224` matches the per-head version in
`59be118` — max |difference| 1.3e-07 at `T=8` and 1.8e-07 at `T=5`
(`T < context_length`, exercising the mask slice), well inside `atol=1e-6`.
Deliberately broken copies fail: q/v blocks swapped (diff 1.3), q/k swapped
(0.08), one head's rows misplaced (0.12). After the scaling fix (`24afd11`) the
outputs differ by ~1e-01, as intended.

Throwaway script in `scratch/` (gitignored), not committed (it needs the old code).
Steps, for reference:

1. `git show 59be118:src/model.py > scratch/model_per_head.py`.
2. Import both; seed *after* the imports (importing `src.model` reseeds the RNG).
3. Build old and new `MultiHeadedSelfAttention` with distinct small sizes
   (e.g. `d=32, H=4, Q=8, T=8`); `.eval()` both so dropout is off.
4. Copy weights inside `torch.no_grad()`. New `qkv.weight` is `(3d, d)`: rows
   `0:d` are all heads' queries, `d:2d` keys, `2d:3d` values; within each block,
   head `h` is rows `h·Q:(h+1)·Q`. Old head `h` has `query`/`key`/`value`, each
   with weight `(Q, d)`. `projection.weight` and `.bias` copy directly.
5. Same random `(B, T, d)` input through both; `torch.allclose(..., atol=1e-6)`
   (not `==` — fused matmuls sum in a different order).
6. Break it on purpose (swap two heads' rows) and confirm it fails.

Copy weights rather than relying on seeds: fused and separate `Linear`s draw
from the same init distribution but not the same values.

## Open decisions

- **What "sweeping model architecture" means** — how `model_depth` and
  `model_width` move together to hit a target `N`, and whether `T` is held
  fixed or scaled with `d` (which would hold the `T/(6d)` ratio constant).
- **Corpus** — TinyStories vs an OpenWebText subset, gated on the largest `D`.

## Queued, not yet done

- **`data/` gitignore rule** at the data-prep step. Reproducibility comes from
  `uv.lock` plus a documented download URL and checksum, not committed corpus bytes.
- **`results/`** is created when the harness first logs.

## nanoGPT reading list

Source: `karpathy/nanoGPT`. Read for the pieces not yet built — data pipeline,
per-run cosine LR schedule, FLOP/param counting, checkpointing.

**Done — 1. `model.py`.** The `3 *` in `CausalSelfAttention` is the fused QKV
projection; now implemented in `src/model.py`. Its question — where the `6` in
`6ND` comes from and what the approximation drops — is answered above under
"Compute exact FLOPs".

**2. `data/shakespeare_char/prepare.py`** — tokenize once, write
`train.bin`/`val.bin` as `uint16` memmaps plus `meta.pkl` (vocab_size, stoi,
itos), rather than re-tokenizing each run. Matters because a sweep pushes one
corpus through many (N, D) pairs.
*Question: what do those files look like for the chosen corpus, and where do
they live relative to `results/`?*

**3. `train.py`** — read in three passes:
- (a) config block + `get_batch()`, against the `ModelConfig`/`TrainConfig` split.
- (b) `get_lr(it)` — cosine with linear warmup. nanoGPT's `lr_decay_iters` is one
  global; here it is computed per run from that run's `iterations`.
- (c) the loop: gradient accumulation, eval, checkpointing on best val loss.
  Skip DDP (multi-GPU is cut-material per CLAUDE.md).

Out of scope: `configurator.py`'s `exec()`-based `sys.argv` override. The
dataclass setup is cleaner; worth recognising, not adopting.

**4. `scaling_laws.ipynb`** — highest-value file. `gpt_params()` and
`chinchilla_flops()`, validated against Chinchilla's Table 4. An answer key for
the harness's N/D/C code.
*Scope warning: it also implements the Approach 3 parametric fit. CLAUDE.md gates
that behind a post-IsoFLOP checkpoint — read the FLOP/param cells, skip the fitting.*

**5. Optional** — `transformer_sizing.ipynb` (maybe useful when picking the N
grid), `sample.py` (covered by `generate`), `bench.py` (skip unless a perf
question shows up).
