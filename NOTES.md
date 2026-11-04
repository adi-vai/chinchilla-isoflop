# Working notes

Durable project state: what's decided, what's open, what's next. Complements
`CLAUDE.md` (which holds the invariants and the phase plan) — this file tracks
the moving parts. Last updated 2026-10-23.

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
- **Two different 12s.** `N ≈ 12·L·d²` is a *parameter count* (per layer:
  W_q, W_k, W_v = 3d², output projection d², MLP up 4d², MLP down 4d²). Those are
  weight matmuls, already priced by `6N`. `12·L·H·Q·T` is *compute with no
  parameters*: the attention matmuls `QKᵀ` and `scores @ V`, where both operands
  are activations. Derivation under "Compute exact FLOPs".
- **Where the parameter-free matmuls sit.** Same forward pass, same layer, between
  the weight matmuls — not a separate step and not outside learning. One attention
  block: `x @ W_qkv` (weights, in 6N) → `Q @ Kᵀ` (activations only) → scale, mask,
  softmax (elementwise, negligible) → `scores @ V` (activations only) → `@ W_o`
  (weights, in 6N). Backward runs the same chain in reverse: gradients must pass
  *through* `QKᵀ` and `scores @ V` to reach W_q, W_k, W_v, so those matmuls get
  backward costs too (gradients w.r.t. both operands, ~2× forward), but produce no
  weight gradients. `6N` counts FLOPs per parameter, so it misses them. Why they
  grow with `T`: weight matmuls act on each token independently (cost per token
  fixed), attention mixes every token with every other (cost per token ∝ `T`).
- **Matmul cost rule.** `(m×k)(k×n)` costs `2·m·n·k` FLOPs: `m·n` outputs, each a
  length-`k` dot product, one multiply + one add per term. The contracted `k`
  counts once (so `QKᵀ` is `2·T²·Q`, not `T²Q²`).

## Where things stand

**Handoff, end of 25 Oct.** N ladder decided (Q=32, d/L=32, 7 rungs 2.2M–22.1M; see
Decided). Analytic counts and `flops_per_token` (matmul FLOPs only) in
`src/util.py`, tested in `tests/` (`uv run pytest`). Provisional band-1
C = 6.41e15. **Next:** open items 3–4 together — margin, batch size, per-run D and
iterations, LR check plan — then 5–6, then the harness.

**Handoff, end of 22 Oct.** Data is ready: `data/tinystories/` holds the cleaned,
tokenized TinyStories (`train.bin` 447,014,068 tokens, `val.bin` 5,355,327, frozen
2K byte-level BPE `tokenizer.json`), reproducible via
`uv run python -m data.tinystories.prepare` (verified: two independent runs give
identical output). **Next session (23 Oct): pilot-plan open items 1 and 3** — the
N ladder (6–7 `(L, d)` pairs, depth/width rule, vs Bryant & Liu's) and the band's
compute `C`, from `D_max = C / flops_per_token(N_min) ≤ 447M` train tokens (both
the user's to work through). Then items 2 and 4–6, then the harness. The
near-duplicate contamination estimate (item 8) is independent and doesn't block.
The `.bin` files are gitignored; a fresh checkout must rerun `prepare` (~4 min).

Phases 1–2 done (`learning/`). Phase 3 (harness) in progress.

`src/model.py` exists: a port of `learning/better_bigram.py` with full type
annotations and batched attention. History:
- `0d62de7` — annotated, shapes relabelled, behaviour unchanged. **Reference
  version for the equivalence test.**
- `c718983` — q/k/v fused into one projection within each head.
- `36ec20d` — q/k/v fused across heads: one `Linear(d, 3d)` in
  `MultiHeadedSelfAttention`, reshaped to `(B, H, T, Q)`; `SelfAttentionHead`
  removed. **Verified equivalent** to `0d62de7` (see Equivalence test below).
- `60bf318` — attention scaling fixed to `head_size**-0.5`.

The model and training loop are still in one file; the split into
`src/model.py` and `src/train.py` comes with the harness.

**Next steps, in order:**
1. Paths from `__file__` (with the `train.py` split).
2. ~~Parameter counts~~ — done: `get_total_params()`, `get_compute_params()`.

## Order of work for the rest of phase 3

~~model port~~ (done) → ~~read `prepare.py`~~ (done) → ~~read `train.py`~~ (done
16 Oct) → **pilot band 1 design** (21 Oct — replaces the full feasibility study; see
"Pilot band 1 plan" below) → ~~data prep (TinyStories: download, dedup,
contamination)~~ (done 21 Oct) → ~~2K BPE tokenizer, freeze `train.bin`/`val.bin`~~
(done 22 Oct) → **pilot open items: N ladder, band C, …** (next) → harness (`src/train.py`) → local
dress rehearsal → GPU smoke test → **LR check → pilot band 1** → decide corpus →
full sweep.

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
  *Decided 25 Oct:* `C` counts **matmul FLOPs only**, so the `N` here is the matmul
  weights, `12·L·d² + d·V` (`matmul_params` in `src/util.py`). The `(10L + 2)·d`
  biases and LayerNorm vectors are elementwise (~2 FLOPs/param/token, not 6), as are
  softmax, ReLU, residual adds and the normalisation itself; all are left out. This
  is the Kaplan / PaLM / nanoGPT `estimate_mfu` convention, so our `C` means what
  theirs does. Size of the omission: 0.8% of N at d=64, 0.2% at d=480.
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

**Schedule shape: cosine (decided); warmup-stable-decay (WSD) only as a budget
fallback.** Two separate questions — the schedule's *shape and length* (decided
here) and its *peak LR* per `N` (open, feasibility input 6).
- Cosine is what Chinchilla used, so the replication stays methodologically
  faithful, and it has no extra knobs. Its cost: the curve is stretched to the
  run's total length, so every `(N, C)` pair is a separate run from scratch.
- WSD: warmup → hold peak LR constant → short cooldown (~last 10–20%) to near
  zero. The constant phase doesn't depend on total length, so one run per `N`
  can fork cooldowns at several `D`s, i.e. several budgets per run — potentially
  much cheaper. Hägele et al. 2024 (*Scaling Laws and Compute-Optimal Training
  Beyond Fixed Training Durations*) report cooldowns matching cosine (verify).
  Costs: departs from Chinchilla's method, adds checkpoint forking and cooldown
  length/shape as new settings.
- Switch to WSD only if the feasibility study shows the full cosine grid
  doesn't fit the budget — and then change the rule in CLAUDE.md deliberately.

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
- `get_compute_params()`: total minus the **position** embedding table. This is
  the `N` in `6N`. *Revised 23 Oct with weight tying:* the token embedding and
  `lm_head` are one `(V, d)` matrix — a free lookup on input, a real matmul on
  output — so it is counted (once, as compute). `nn.Module.parameters()` yields a
  shared parameter once, so `get_total_params()` needs no special case.

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
Fixed in `src/model.py` in `60bf318`, after the equivalence test passed.

Why it's wrong: the scaling fixes the variance of a dot product. A score is
`Σ qᵢkᵢ` over `head_size` terms; with unit-variance components its variance is
`head_size`, so dividing by `√head_size` restores variance 1. `embedding_dimension`
isn't in that dot product. The two coincide only with a single head.

Effect: scores have std ≈ 0.41 instead of 1 at `d=384, H=6`, so attention starts
too uniform. The error factor is exactly `√num_heads`, so it varies with a
swept hyperparameter and biases runs against each other. The model partly
compensates by learning larger `W_q`, `W_k`, which is why it still trains.

## Equivalence test (done)

**Result:** the batched attention in `36ec20d` matches the per-head version in
`0d62de7` — max |difference| 1.3e-07 at `T=8` and 1.8e-07 at `T=5`
(`T < context_length`, exercising the mask slice), well inside `atol=1e-6`.
Deliberately broken copies fail: q/v blocks swapped (diff 1.3), q/k swapped
(0.08), one head's rows misplaced (0.12). After the scaling fix (`60bf318`) the
outputs differ by ~1e-01, as intended.

Throwaway script in `scratch/` (gitignored), not committed (it needs the old code).
Steps, for reference:

1. `git show 0d62de7:src/model.py > scratch/model_per_head.py`.
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

## Pilot band 1 plan (decided 21 Oct)

The full multi-band feasibility study is deferred: the corpus question can't be
settled from papers, and at TinyStories scale a pilot band costs pennies. Design
the pilot so it answers that question; design the full sweep after the result.
The "Feasibility study" section below remains the reference for inputs.

### Decided
- **Corpus: TinyStories, provisional.** Single-epoch. With a 2K BPE tokenizer
  Bryant & Liu use up to 5×10⁸ unique tokens; measured: **447M clean train tokens**
  (open item 7). That supports single-epoch bands with optima up to roughly 5–10M
  params (smallest model needs D = C/(6·N_min) ≤ corpus).
- **Tokenizer: byte-level BPE, vocab 2,000 (decided 22 Oct).** Configuration in
  `build_tokenizer()` (`data/tinystories/prepare.py`): byte-level pre-tokenizer, full
  256-byte initial alphabet (any text encodes, nothing silently dropped), one special
  token `<|endoftext|>` between stories. Frozen tokenizer trained on **all 1,792,648
  cleaned train stories**. Evidence (`data/tinystories/vocab_sweep.py`: trained on a
  100k cleaned-train sample, measured on the full validation split):

  | V | 500 | 1,000 | 2,000 | 4,000 | 8,000 | 10,000 |
  |---|---|---|---|---|---|---|
  | chars/token | 2.29 | 3.04 | 3.60 | 3.94 | 4.11 | 4.12 |

  Gain per doubling roughly halves (+33%, +18%, +9%, +4%); flat above 8K. Cost side:
  each entry costs `d` params in the embedding and `d` in `lm_head`; at `d = 64`,
  2K → 4K adds 128K `lm_head` params (~13% of a ~1M-param smallest model) for 9%
  better compression. 2K sits before that trade turns bad for the smallest models.
  Reminder: loss in nats/token is only comparable under this one tokenizer — convert
  to bits per character/byte for any cross-tokenizer comparison.
- **Data artifacts (22 Oct).** `uv run python -m data.tinystories.prepare` writes
  `train.bin` / `val.bin` (`uint16` memmaps, gitignored, 894 MB / 11 MB) and
  `tokenizer.json` (committed — the frozen tokenizer; `<|endoftext|>` is id 0, ids
  0–1999) into `data/tinystories/`.
  Checks passed: end-of-text count = story count in both splits; decoded validation
  text round-trips exactly. Byte-level pre-tokenizer adds a prefix space, so every
  story's first word is tokenized like a mid-text word (`Ġ`-prefixed) — consistent
  across all data.
- **Context length T = 256, fixed for every run (decided 23 Oct).** Reasons:
  - Loss at different `T` is a different task — a longer context makes the next
    token easier to predict — so scaling `T` with `d` would mix model size with task
    difficulty along exactly the axis being fitted. Fixed `T` keeps one task.
  - The mean cleaned story is ~249 tokens (incl. `<|endoftext|>`), so a 256-token
    window holds roughly one story.
  - Matches Bryant & Liu's default.

  Consequence: the attention share of compute, `T/(6d)`, varies across the ladder —
  67% of `6N` at `d = 64`, 33% at 128, 22% at 192, 13% at 320, 9% at 480. That's the
  bias `6ND` would introduce; the exact `flops_per_token` (6N + 12·L·T·d) accounts
  for it, which is why exact FLOPs is non-negotiable here. The position embedding is
  `256·d` params (in `N_total`, not in `N_compute`).
- **Weight tying, no `lm_head` bias, nanoGPT initialisation (decided 23 Oct).**
  - Token embedding and `lm_head` share one `(V, d)` matrix, as in GPT-2, nanoGPT and
    Bryant & Liu. Untied, the input table is `V·d` params that do no compute — at
    `d = 64` that's 128K on a ~330K model (~40% of total N), inflating total N most
    at the small end of the ladder.
  - All `Linear`/`Embedding` weights ~ N(0, 0.02), biases zero. Without this, the
    token embedding (via `Linear`'s default init) started 14–38× smaller than the
    position embedding (N(0, 1)), with the ratio growing with `d` — an N-correlated
    difference in starting conditions. Now both start at std 0.02 at every width.
  - Residual projections (`mh_sa.attention_projection`, `ffwd.ffwd_projection`) ~
    N(0, 0.02/√(2L)), per GPT-2, so the residual stream's variance doesn't grow with
    depth (relevant: `L` varies across the ladder). Matched by name, listing both
    explicitly, with an assert that exactly `2L` layers were scaled — name matching
    silently skipped the feedforward layer once (`FeedForward` was restructured from
    an `nn.Sequential` to named layers; verified identical function). **To revisit:**
    the derivation of the `1/√(2L)` factor — copied from nanoGPT, not yet worked
    through.
  - Checks: initial loss on random targets 7.61–7.69 vs `ln 2000 = 7.60`; both
    embeddings std 0.020; residual projections std 0.0071 at `L = 4`.
  - **Calibration:** with tying, total N matches Bryant & Liu exactly for all five
    configs (343,680 / 1,080,576 / 3,099,264 / 10,578,560 / 28,779,840 vs
    344K / 1.1M / 3.1M / 10.6M / 28.8M). Their N = total params of a tied model with
    no `lm_head` bias; ours is now the same model.
- **N ladder: Q=32, d/L=32, 7 rungs (decided 25 Oct; `src/ladder.py`).**
  Method: 7 targets geometric over 10× in *total* N (Chinchilla's convention),
  centred on N_opt = 7.1M. A shape rule (head size Q, target d/L, tolerance ×/÷1.2,
  L ≥ 4, H 4–12) is fixed for the whole ladder so shape doesn't drift between rungs
  and confound N; each rung takes the candidate closest in log N; rules ranked by
  worst rung (RMS logged too). Searched Q ∈ {32, 64} × d/L ∈ {20, 32, 64, 128, 200}.

  |   d |  L |  H |  d/L |    total N |  miss |
  |----:|---:|---:|-----:|-----------:|------:|
  | 160 |  6 |  5 | 26.7 |  2,214,080 | −1.4% |
  | 192 |  6 |  6 | 32.0 |  3,099,264 | −6.0% |
  | 224 |  7 |  7 | 32.0 |  4,736,256 | −2.1% |
  | 256 |  8 |  8 | 32.0 |  6,889,984 | −3.0% |
  | 288 | 10 |  9 | 28.8 | 10,632,384 | +2.0% |
  | 320 | 12 | 10 | 26.7 | 15,506,560 | +1.4% |
  | 384 | 12 | 12 | 32.0 | 22,146,816 | −1.4% |

  Why this rule: realised d/L 26.7–32, the most consistent shape and well inside
  Kaplan's flat band; worst miss 6.3%; spread 10.0×; rung 2 is exactly B&L's 3.1M
  config. The only other feasible rule, d/L=20, sits at or below the band edge on
  most rungs (17.8–22.9). **Finding for the write-up:** at 2–22M only Q=32 with
  d/L ≈ 20–32 is feasible — Q=64 needs d ≥ 256, so with L ≥ 4 the smallest model is
  already ~3.7M; high d/L needs d ≥ 4r (d/L=128 → N ≥ ~12.6M). Consistent with B&L's
  16–48. **Coupling check:** C from the bottom rung (item 3) gives heuristic
  √(C/120) = 7.31M, +6.1% vs the middle rung; criterion is within half a rung
  (±21%), and the 20 tok/param heuristic is itself uncertain by far more (≈25 in
  Besiroglu et al. 2024; untested on TinyStories), which the 10× spread covers.
- **Staged spend: £200 committed; up to ~£1k only after band 1 works.** Rule: stage 1
  is a *subset* of the full plan, never a separate mini-plan. Locked at band 1, so
  choose for the ceiling: corpus, tokenizer, depth/width rule, held-out eval set, LR
  rule. Freely extendable later: higher bands, seeds, extra sizes in a band.
  Rough H100 scale (Oct 2026: ~$2–3/h ≈ £2/h; small-model MFU 10–30%): £200 ≈ 100
  H100-h ≈ 7×10¹⁹ FLOPs at 20% MFU. TinyStories single-epoch bands cost far less —
  data, not money, binds.
- **Sequence:** CPU/MPS smoke → GPU smoke (launch, log, checkpoint; loss starts near
  ln 2000 ≈ 7.6 nats and falls) → **LR check** → **pilot band** → decide. Keep the
  pipeline test and the science test separate: a harness bug must not masquerade as
  "TinyStories flattens".
- **Pilot band = the hardest band TinyStories supports**: the highest single-epoch
  `C`, optimum ~5–10M, so flattening (most likely at large N) is tested where it
  bites. Clean there ⇒ lower bands are fine too.
- **LR check before the band**, on the band's sizes — otherwise an untuned fixed LR
  can flatten the U and get blamed on the corpus (Porian et al.).
- **Pass/fail criterion (pre-registered — write numbers before any run):**
  - Approach 2 per band: fit a **parabola in log N**; the vertex is the optimum.
  - *Shape:* edge points ≥ **5%** above the minimum loss (fixed 4 Nov, before any
    run). This project's threshold, **not** a Chinchilla criterion — Chinchilla sets no
    such number, and depth mostly reflects how wide the N span is (10× here).
  - *Location:* vertex **inside** the swept range, and its uncertainty small — e.g.
    a second seed at 2–3 sizes or a bootstrap over points moves the vertex by
    < ~1.5× in N.
  - ~6–7 well-spread sizes, with margin either side (optimum location unknown in
    advance). Vertex near an edge ⇒ shift the range and rerun, not a fail.
- **Switch-corpus rule:** pilot verdict is scientific only — no clean, well-located U
  at the highest feasible TinyStories band ⇒ switch to an OpenWebText-scale corpus
  (cost: data-prep hours, not money). Full-sweep economics (many points on a small
  corpus vs fewer on a large one) is a separate, later decision. Caveat for the
  write-up: a tight fit on TinyStories is a methods result; its exponents aren't
  comparable to Chinchilla's web-text numbers.
- **Which N goes where (two jobs, only one is a choice):**
  - *Compute / IsoFLOP constraint — not a choice:* FLOPs/token = 6·N_compute +
    12·L·T·d. N_compute = params that do matmuls per token: **exclude** the
    embedding lookup (indexing, ~0 FLOPs), **include** the output layer d×V. With
    tied weights (nanoGPT ties wte/lm_head) one matrix is one param block but two
    uses — free lookup + costly matmul. Using total N here over-counts FLOPs by the
    embedding share, largest at small N ⇒ smallest models get too few tokens (an
    N-dependent bias).
  - *Reporting / fit x-axis — a choice:* **primary = N_total incl. embeddings**
    (Chinchilla convention; matters most at small N — at 2K vocab, d=64 the
    embedding is ~128K params vs ~200K in a 4-layer stack). **Robustness = refit with
    non-embedding N** (Kaplan convention) — same runs, second fit, free. Pearce & Song
    2024 (*Reconciling Kaplan and Chinchilla*) reportedly attribute much of the
    Kaplan–Chinchilla gap to this choice at small scale — **verify before citing**.
  - Check Chinchilla's own FLOP conventions via nanoGPT `scaling_laws.ipynb` against
    `get_compute_params`; record which convention is followed and why.
- **Bryant & Liu (arXiv 2605.09189): defaults only, not a comparison target.** They
  study the data-constrained regime (fixed unique D, T ≫ D, 5–5000 epochs, WSD +
  cooldown forks, fixed LR 3e-4 and batch 16K tokens for every N). This project is
  compute-constrained, single-epoch, D = T. Using their fitted law at D = T would be
  extrapolating outside their regime — don't. Usable: 2K BPE, context 256, AdamW
  (0.9, 0.95), wd 0.1, warmup 500; their ladder (d_model, n_layer, n_head, d_ff) ∈
  {(64,4,4,256), (128,4,4,512), (192,6,6,768), (320,8,8,1280), (480,10,8,1920)} →
  N ∈ {344K, 1.1M, 3.1M, 10.6M, 28.8M}. Note their disclosed config drift (W=100 vs
  500 on the 30M runs) — the honesty standard for this write-up.

### Open — to settle before the pilot runs
1. ~~N ladder~~ — **decided 25 Oct: Q=32, d/L=32, 7 rungs 2.2M–22.1M** (see Decided).
2. ~~Context T~~ — **decided 23 Oct: fixed T = 256** (see Decided).
3. **Band compute C:** highest single-epoch C given the token count and N_min.
   *25 Oct, provisional:* from the ladder's bottom rung (160, 6, 5),
   `C = 0.9 · 447,014,068 · flops_per_token = 6.41e15` (`coupling_check` in
   `src/ladder.py`). Still to settle: the margin, then per-run D and iterations,
   which need the batch size (item 4).
4. **LR check plan:** which sizes, which LR values, how many steps; batch size.
5. **Seeds:** which 2–3 sizes get a second seed for the location test.
6. **Cost / wall-clock estimate** for LR check + pilot band.
7. ~~Exact token count~~ — **done 22 Oct: 447,014,068 train tokens** (1,792,648 clean
   stories, 249.4 tokens/story incl. `<|endoftext|>`); validation 5,355,327 tokens
   (21,990 stories). Two independent runs gave identical counts and tokenizer.
8. **Near-duplicate contamination estimate (logged 21 Oct).** Exact sha256 matching
   only catches *verbatim* val→train leakage. TinyStories is synthetic and formulaic,
   so near-copies (a changed name, a reworded sentence) are likely, and near-duplicate
   memorisation is N-correlated in the same direction as exact. Plan: on a random
   sample of ~1,000 val stories (`shuffle(seed)` — the dataset head isn't
   representative), measure n-gram overlap / MinHash similarity against clean train;
   report the fraction above a stated threshold. Outcome decides: small ⇒ stated
   limitation in the write-up; material ⇒ consider removal. ~1 hour; does **not**
   block the tokenizer.

### Harness additions from this plan
- **Log MFU per run**: tokens/s × FLOPs/token ÷ peak. Same 6N + 12·L·T·d formula
  (PaLM appendix reportedly defines MFU with this attention term — verify).
- Small models underuse an H100: consider **several small runs per GPU
  concurrently**. Cheap wins: bf16 autocast, `scaled_dot_product_attention`
  (FlashAttention), fused AdamW, no per-step `.item()` syncs; `torch.compile` last.

## Feasibility study (reference inputs; superseded in scope by the pilot plan above)

Corpus, tokenizer and sweep design are one decision, not three: pick the
corpus that supports the sweep the experiment needs. The study sets up the
whole experiment; the harness after it is mostly implementation.

**Binding constraint (single epoch).** At each `C` the smallest model gets the
most tokens, so the run that needs the most data is always

```
D_max = C_max / flops_per_token(N_min at C_max)  ≤  training-split tokens
```

**Inputs:**
1. Corpus size **in tokens** — depends on the tokenizer (char-level gives ~3–4×
   the tokens of GPT-2 BPE for the same text). OpenWebText: ~9B train tokens
   with GPT-2 BPE per nanoGPT's `openwebtext/prepare.py`. TinyStories: count it.
2. Range of `C` — top set by GPU budget × realistic throughput and the
   timeline (band 1 first; later bands can follow once it works); bottom by the
   smallest model that still learns meaningfully. Budget GPU-hours at a realistic
   utilisation for small matmuls (more like 10–30% of peak than 50%+).
3. `N` spread at each `C` — wide enough for a clear U on both sides of the
   optimum (typically ~10× smallest to largest).
4. Architecture per `N` — how `model_depth` and `model_width` move together, and
   whether `T` is fixed or scaled with `d` — *decided 23 Oct: fixed at 256*.
5. `flops_per_token` per config — formula and parameter counts are in place.
   Validate against nanoGPT's `scaling_laws.ipynb` (reading item 4).
6. **Peak LR (and batch size) as a function of `N` — currently undecided.** Only
   the schedule *shape and length* are settled (cosine, decaying at each run's own
   `iterations`; see Decisions). Chinchilla gives a per-size LR table, not a rule, and its
   smallest model is larger than most of ours — following it means extrapolating
   below its range. Porian et al. 2024 (*Resolving Discrepancies in
   Compute-Optimal Scaling*) find untuned LR/batch size moves the optimum enough to
   explain much of the Kaplan–Chinchilla gap at small scale. To decide: what rule,
   where it comes from, and a cheap check that it holds at this scale (where in
   the `N` range would a mismatch distort the U-curve most?).
   *15 Oct (get_lr reading):* `min_lr` = peak/10 in nanoGPT, so this is one
   open value (peak LR vs `N`), not two.
7. **Measurement resolution.** Near the minimum, neighbouring `N` may differ by
   ~0.01 in loss. Is the val set large enough, and run-to-run noise small enough
   (e.g. one config × 2 seeds), to resolve that? Otherwise the minimum is noise.
   *Decided 16 Oct (eval):* the number fed into the fit is **one final eval at the
   last step** (LR fully decayed), on the **full fixed held-out set**, same for every
   run. Rejected: best-val checkpoint (min of noisy evals biases low; more evals on
   longer runs → bias varies across the sweep). Considered: averaging all evals within
   an LR band near `min_lr` — the band is a fixed fraction of every run (cosine
   depends only on t/T), and a bias constant across `N` at fixed `C` doesn't move the
   Approach-2 argmin. But (a) uniformity across `N` is an assumption at ~0.01
   resolution, (b) evals-per-band vary with run length unless `eval_interval` scales
   with the run, (c) on a fixed val set, re-scoring the same tokens doesn't reduce
   sampling noise — it only smooths weight jitter, small at min LR. Same compute in
   one bigger final eval gives the same noise reduction with zero bias.
   **Diagnostic:** also log the LR-band average (evals at fixed fractions of the
   run). If its argmin matches the final-eval argmin at every `C`, that's evidence
   the bias is uniform; if not, investigate. Intermediate evals otherwise =
   monitoring only. Eval is forward-only (~2N/token) and changes no weights, so it
   is outside `C` but inside the GPU budget.
   **Open: how many held-out tokens for the final eval?** Needs (a) how flat the
   IsoFLOP minimum is (rough guide: Bryant & Liu's curves) and (b) per-token loss
   variance (measure from any short run); standard error ∝ 1/√tokens.
   Also: eval overhead ratio ≈ (2 splits × eval_iters × 2N) / (eval_interval × 6N)
   — compute it for the chosen settings.
8. **Unique tokens, not raw tokens.** *Measured 21 Oct* (`data/tinystories/prepare.py`,
   exact sha256 match on story text):
   - Train 2,119,719 stories; **320,470 exact duplicates (15.1%)** — not the ~20%
     previously reported. Whitespace normalisation finds no extra duplicates, so
     exact matching is the rule. None in the first 10k rows: the head of the
     dataset isn't representative — sample with `shuffle(seed)` when testing.
   - Validation 21,990 stories, no internal duplicates; **6,601 (30.0% of
     validation) also appear verbatim in train**, each exactly once (0.31% of train).
   - **Decision:** remove contaminated stories from **train**, keep the official
     validation split intact (costs 0.3% of train; keeps the eval set others use
     and its precision). Unremoved, ~30% of val loss would measure memorisation,
     which grows with N — an N-correlated bias toward larger optima.
   - Clean train: **1,792,648 stories (84.6%)**. The single-epoch bound applies to
     these; token count pending the tokenizer (open item 7).
9. **Warmup per run (open).** nanoGPT: 2k of 600k iters (1/300), fixed count.
   Warmup exists so Adam's second-moment estimate `v` settles; its memory is
   ~1/(1−β2) steps (β2 = 0.95 → ~20), and it needs a few of those (~60–100
   steps). Candidates: pure fraction of `iterations`, or
   max(fraction × iterations, floor). Whether the floor binds depends on the
   shortest run's iteration count — which comes out of this study.

**Second criterion — meaningful at this scale, not just feasible.**
TinyStories: narrow, simple distribution; tiny models learn it, but loss may
flatten early and distort the fit. OpenWebText: closer to Chinchilla's data, so
exponents are more comparable, but tiny models may all sit on the steep early
part of the curve.

**Prior evidence on this corpus (confirmed on arXiv to exist and cover this,
from a summary only — read the relevant sections before citing).** Bryant & Liu 2026, *Practical Scaling Laws*
(arXiv 2605.09189), train transformers on TinyStories up to ~74M params (~206
A10G GPU-hours). Their grid shows which `N`/`C` ranges gave informative curves
here, and they use a warmup-stable-decay schedule with cooldown forks (see the
schedule decision: a budget fallback only). Their thesis —
Chinchilla's form mishandles data-constrained/repeated regimes — is exactly what
the single-epoch bound guards against.

**Vocabulary size feeds back into N.** `lm_head` is `d · V`; a 50k BPE vocabulary
with small `d` makes it the largest layer of the smallest models.

**Related harness requirements surfaced so far:**
- A data loader that walks the corpus **without repetition** (sequential or
  shuffled-without-replacement), stopping before it runs out. nanoGPT's
  `get_batch` samples random offsets with replacement, which repeats some tokens
  and skips others — that breaks single epoch.
- Store tokens as `uint16` memmaps (vocab < 65,536), as in nanoGPT's `prepare.py`.
- **Gradient accumulation (16 Oct, train.py pass c).** Effective batch is the
  scientific variable (tied to `N`, open in input 6); the micro-batch / accumulation
  split is engineering only — mathematically identical to one big batch given the
  loss scaling below (no dropout/batchnorm). Per run: `accum_steps = effective_batch /
  (micro_batch × T)`, must divide evenly; micro-batch is memory-bound and shrinks as
  `N` grows, so `accum_steps` grows with `N`. Iterations = D / (accum_steps ×
  micro_batch × T). The split doesn't change FLOPs.
- **Scale each micro-batch loss by 1/accum_steps before `backward()`.** `.grad`
  accumulates by summation, so unscaled it holds k·∇L. Adam's m/√v hides that, but
  gradient clipping doesn't — and k grows with `N`, so a missed scaling would clip
  larger models harder: an `N`-correlated confounder (same pattern as √n_heads).
- **Log the true loss:** multiply the scaled loss back by `accum_steps` before logging.
- **Log per run: pre-clip gradient norm and fraction of steps clipped.** Even with
  correct scaling, a fixed clip threshold (nanoGPT: 1.0) can bind differently across
  `N`. If the clipped fraction trends with `N`, it's a confounder to fix before fitting.

## Queued, not yet done

- ~~`data/` gitignore rule~~ — done 21 Oct: `data/**/*.bin` ignored. Reproducibility
  comes from `uv.lock` + `prepare.py` + the HF dataset ID, not committed corpus bytes.
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
  *Done 16 Oct* — decisions in input 7 (eval) and harness requirements (accumulation).
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
