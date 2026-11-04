# chinchilla-isoflop

A small-scale, from-scratch replication of Chinchilla's IsoFLOP curves
(Hoffmann et al., 2022, Approach 2) in PyTorch. The driving question: does
Chinchilla's compute-optimal allocation — parameters and training tokens scaling
equally with compute, at roughly 20 tokens per parameter — hold at the small end
of model size, and what effects start to bias it there? The sweep trains
2–22M-parameter models for a single epoch on TinyStories, deduplicated and
decontaminated (30% of validation stories also appeared in the training set; those
copies were removed from train), with a 2K byte-level BPE tokenizer trained on the
cleaned training split. At these sizes attention is a non-trivial share of compute:
it adds 11–23% on top of the 6N FLOPs per token that `C ≈ 6ND` counts. Because that
share changes with N, using 6ND would tilt the IsoFLOP curves rather than simply
rescale them, so this project counts FLOPs exactly.

## Status

**Built:**
- Decoder-only transformer (`src/model.py`), written from scratch following
  nanoGPT: batched multi-head attention (verified equivalent to a per-head
  reference implementation), tied embedding / output weights, nanoGPT
  initialisation.
- Data pipeline (`data/tinystories/prepare.py`): deduplication,
  decontamination, tokenizer training, token memmaps. Reproducible: two
  independent runs give identical output.
- Exact parameter and FLOPs counting (`src/util.py`), tested against the model
  and against Bryant & Liu's published parameter counts (`tests/`).
- The band-1 model ladder (`src/ladder.py`), below.

**Next:** fix the band's compute budget, batch size and learning rate (via a
short LR check), then the training harness, a local dress rehearsal of a
miniature sweep, a GPU smoke test, and pilot band 1.

## Method

**IsoFLOP (Approach 2).** Fix a compute budget C, train a ladder of model sizes
N with each run's token count set by the budget, `D = C / flops_per_token(N)`,
and plot final loss against N. The curve is U-shaped: too small a model is
capacity-limited, too large a model sees too few tokens. A parabola fit in log N
locates the compute-optimal size. Repeating across budgets gives the exponents in
`N_opt ∝ C^a` and `D_opt ∝ C^b`; Chinchilla finds a ≈ b ≈ 0.5.

**Decisions that matter at this scale.** Full reasoning, measurements and
alternatives considered are in [NOTES.md](NOTES.md).

- **Exact FLOPs.** `flops_per_token = 6·N + 12·L·T·d`: the 6N term for the weight
  matmuls, plus the attention-score matmuls (`QKᵀ` and `attn @ V`), which have no
  parameters and so are invisible to 6N. Matmul FLOPs only — biases, LayerNorm and
  softmax are elementwise and left out, the same convention as Kaplan et al.,
  PaLM and nanoGPT. Each run will log its realised C, after rounding to whole
  iterations.
- **One schedule per run.** Each run's cosine learning-rate schedule decays to its
  minimum at that run's own token count. A schedule shared across runs would
  under-train some and cut others off mid-decay, biasing the comparison.
- **Single epoch, clean data.** 447M training tokens after cleaning: 15% of
  training stories were exact duplicates, and 6,601 validation stories (30% of
  validation) also appeared in train. Both were removed from train; validation is
  the official split, unchanged. The corpus is fixed so that only the N/D tradeoff
  varies.
- **Tokenizer: 2K byte-level BPE.** Chosen from a sweep of compression against
  vocabulary size (`data/tinystories/vocab_sweep.py`): the gain per doubling
  roughly halves each time (+9% from 2K to 4K, flat above 8K), while each extra
  vocabulary entry costs `d` parameters — a large share of the smallest models.
- **Fixed context length T = 256.** Loss at a different context length is a
  different task, so T stays fixed rather than scaling with model width.
- **Parameter counting calibrated.** With tied weights, total N reproduces all five
  of Bryant & Liu's published TinyStories model sizes exactly.

**Band-1 ladder.** Seven sizes, geometrically spaced over 10× in total N and
centred on the Chinchilla-heuristic optimum for the largest single-epoch budget
TinyStories supports. One shape rule for every rung (head size 32, width/depth
ratio ≈ 32), so shape doesn't vary along with N:

| width d | depth L | heads | d/L | total N |
|---:|---:|---:|---:|---:|
| 160 |  6 |  5 | 26.7 |  2,214,080 |
| 192 |  6 |  6 | 32.0 |  3,099,264 |
| 224 |  7 |  7 | 32.0 |  4,736,256 |
| 256 |  8 |  8 | 32.0 |  6,889,984 |
| 288 | 10 |  9 | 28.8 | 10,632,384 |
| 320 | 12 | 10 | 26.7 | 15,506,560 |
| 384 | 12 | 12 | 32.0 | 22,146,816 |

At 2–22M parameters only narrow shapes are feasible: a 64-wide head or a
width/depth ratio much above 32 can't reach the bottom of the range with at least
four layers.

## Results

Pilot band 1 is not yet run. The pass/fail criterion is pre-registered in
[NOTES.md](NOTES.md) before any run: a parabola in log N whose vertex lies inside
the swept range, with edge points at least 5% above the minimum loss, and a vertex
location stable under a second seed at a few sizes.

## Next steps

- Pilot band 1, then further compute bands to fit the exponents a and b.
- **Planned extension: inference-aware model sizing.** Adding an inference-cost
  term to each run's total cost should move the cost-optimal model size towards
  smaller, over-trained models (Sardana et al., 2023). This is a reanalysis of the
  multi-band results rather than new training, but it needs several bands, and the
  single-epoch token budget caps how far into over-training this corpus can go.

## Approach

I wrote the code by hand on purpose: this project is how I'm learning deep
learning, by building a transformer and its training setup myself rather than
starting from a library. Karpathy's nanoGPT lecture and repository were the
starting point. `learning/` holds the frozen practice code from that phase, ending
at `better_bigram.py`, which despite its name is a full decoder-only transformer.
It keeps the mistakes I made at the time, including an attention-logit scaling bug
later fixed in `src/model.py` (see [learning/README.md](learning/README.md)).

What `src/` and `data/` add over the lecture code is mostly verification and
experimental design:

- batched multi-head attention, verified numerically equivalent to a per-head
  reference implementation;
- parameter and FLOPs counts tested against the model and calibrated against
  Bryant & Liu's published model sizes. The FLOPs formula is the standard one
  (nanoGPT's `estimate_mfu` uses it); applying it exactly across the sweep is the
  point here;
- a data pipeline that deduplicates and decontaminates TinyStories, and the
  tokenizer-size sweep behind the 2K vocabulary;
- a search over model shapes for the IsoFLOP ladder, and the sweep design itself:
  exact FLOPs, per-run schedules, a pre-registered pass/fail criterion.

I used an AI assistant as a tutor and code reviewer, and it wrote some of the
supporting code (the counting tests, the ladder-search loop) to my design.

## Reproducing

Managed with [uv](https://docs.astral.sh/uv/); run everything from the repo root.

```
uv sync
uv run python -m data.tinystories.prepare      # download, clean, tokenize (~4 min)
uv run python -m data.tinystories.vocab_sweep  # the tokenizer-size evidence
uv run python -m src.ladder                    # calibration and ladder search
uv run pytest                                  # counting tests
```

`prepare` writes `train.bin` and `val.bin` (gitignored) and the frozen
`tokenizer.json` (committed) into `data/tinystories/`. `uv.lock` is committed —
reproducibility is part of the deliverable.

## Layout

```
src/         model, counting utilities, ladder search; training harness to come
data/        corpus preparation (TinyStories) and the frozen tokenizer
tests/       parameter and FLOPs counting tests
learning/    practice code from building up to the harness; frozen snapshots
NOTES.md     decisions, measurements, open questions
```

## References

- Hoffmann et al. (2022). *Training Compute-Optimal Large Language Models.*
- Kaplan et al. (2020). *Scaling Laws for Neural Language Models.*
- Besiroglu et al. (2024). *Chinchilla Scaling: A Replication Attempt.*
- Sardana et al. (2023). *Beyond Chinchilla-Optimal: Accounting for Inference in
  Language Model Scaling Laws.*
- Bryant & Liu (2026). *Practical Scaling Laws: Converting Compute into Performance
  in a Data-Constrained World.* arXiv:2605.09189.
- Eldan & Li (2023). *TinyStories: How Small Can Language Models Be and Still
  Speak Coherent English?* Dataset:
  [roneneldan/TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories),
  CDLA-Sharing-1.0; the committed `tokenizer.json` is trained on it.
- Karpathy, [nanoGPT](https://github.com/karpathy/nanoGPT).
