# CLAUDE.md — chinchilla-isoflop

## What this project is
A from-scratch replication of the Chinchilla compute-optimal scaling result
(Hoffmann et al. 2022) using the IsoFLOP method (their Approach 2), plus an
extension probing inference-aware over-training (how the compute-optimal model
size shifts LEFT / smaller once an inference-cost term enters the objective).

Goal: train a family of small transformers across compute budgets, fit the
IsoFLOP frontier (N_opt ∝ C^a, D_opt ∝ C^b, targets a≈b≈0.5, ~20 tok/param),
then show the cost-optimal model size migrates into the over-trained region.

This is a portfolio + learning project. The author is deliberately learning
deep learning through this build — see "How to work with me" below.

## Core facts to respect
- Training compute: C ≈ 6ND (N params, D tokens). 6 = fwd(2) + bwd(4) per param per token.
- The harness counts compute EXACTLY: flops_per_token = 6N + 12·L·H·Q·T. The second term
  is attention (no params, invisible to 6N); it is ~T/(6d) of compute — large for the
  small models here, and varying with N, so 6ND would bias the sweep. See NOTES.md.
- IsoFLOP (Approach 2): fix C, sweep N, D = C / flops_per_token; loss-vs-N is U-shaped; min = optimal N.
- Cosine LR MUST decay to its minimum at each run's OWN token count. Every IsoFLOP
  run has a different D → each needs its own schedule length. Do NOT share one schedule.
- Approach 3 (parametric fit L = E + A/N^α + B/D^β) is FRAGILE and is a STRETCH GOAL
  only, gated on a post-IsoFLOP budget checkpoint. Do not build it early.
- Read losses as "distance to the entropy floor E", not "distance to 0".

## Stack & conventions
- Python + PyTorch, managed with uv (pyproject.toml + uv.lock; .venv/ created and
  managed by uv). Run scripts with `uv run python ...`; add deps with `uv add ...`.
  The uv.lock file is committed — reproducibility is part of the deliverable.
- Start on CPU with tiny models (correctness first); rent a GPU only at the IsoFLOP
  sweep phase.
- Dataset: fixed clean corpus (TinyStories or an OpenWebText subset). Data curation is
  deliberately NOT a variable — keep it fixed to isolate the N/D tradeoff.
- Log per run: N, D, realised C (exact FLOPs, after rounding iterations), final/val loss → CSV in results/. Keep raw logs committed.
- Repo layout (create as needed, not up front): src/ (model, train), configs/,
  results/, notebooks/ or plots/, README.md, NOTES.md (decisions + open questions).
  learning/ holds frozen practice snapshots — src/ never imports from it.
- src/ is a package: run as `uv run python -m src.<module>` from the repo root, and
  resolve paths from `__file__`, not the CWD.

## Build phases (current focus at top)
1. PyTorch fluency: hand-written training loop on a tiny MLP. (DONE)
2. Transformer from scratch (follow nanoGPT), understand attention/residual/layernorm. (DONE)
3. GPU harness + config-driven runs + CSV logging. (CURRENT)
4. IsoFLOP core sweep (Approach 2) — the must-land deliverable.
5. Over-training / inference-cost extension.
6. Writeup (protected — cut experiments before cutting writeup polish).

## How to work with me (IMPORTANT)
- I am learning DL through this build. Do NOT just write whole solutions.
- Explain approach first, let me attempt, then review. Prefer small diffs I can follow.
- When I ask "why", derive/explain from first principles — I have the maths background
  (derived backprop by hand; comfortable with probability, optimisation, linear algebra).
- Point out mistakes directly; I want correction, not agreement.
- If I'm scope-creeping (e.g. reaching for Approach 3 or a distributed run before the
  core exists), flag it.
- When you correct a recurring mistake, add a line to this file so it doesn't repeat.

## Timeline
Quality hours > total hours. Stretch goals (Approach 3, multi-GPU) are cut-material at the post-IsoFLOP checkpoint.

---
<!-- NOTE (personal scaffolding — review before making repo public):
The "How to work with me" and "Timeline" sections are for the private build phase.
Before flipping public, decide what stays: the project/facts/conventions sections are
genuinely useful repo docs; the personal coaching + timeline notes probably move out
(into a private note) or get trimmed. JIT this at the go-public step, not before. -->
