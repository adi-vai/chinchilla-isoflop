# chinchilla-isoflop

A from-scratch replication of the Chinchilla compute-optimal scaling result
(Hoffmann et al., 2022) using the IsoFLOP method, plus an extension probing
inference-aware over-training.

**Status:** early. Transformer built from scratch; experimental harness in progress.

## Layout

```
src/         experimental harness (model, training, sweep) — the deliverable
learning/    practice code from building up to it; frozen snapshots
data/        corpus
results/     per-run logs: N, D, C = 6ND, final/val loss
```

## Running

Managed with [uv](https://docs.astral.sh/uv/). Run from the repo root:

```
uv run python -m src.<module>
```

`uv.lock` is committed — reproducibility is part of the deliverable.

## Method

Training compute is approximated as `C ≈ 6ND` for `N` parameters and `D`
training tokens. The IsoFLOP approach fixes `C`, sweeps `N`, and sets
`D = C / 6N`; loss as a function of `N` is U-shaped, and its minimum gives the
compute-optimal model size at that budget. Repeating across budgets gives
`N_opt ∝ C^a` and `D_opt ∝ C^b`.

Writeup to follow.
