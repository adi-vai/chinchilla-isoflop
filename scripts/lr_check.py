"""LR-check summary for band 1: is the default peak LR (6e-4) good enough?

Only rungs run at more than one LR (seed 0) are part of the check; rungs with a
single LR (sweep runs) are dropped. The LR columns are every peak LR in the
checked rungs, so LRs added after the pre-registered grid show up as extra
columns. For each checked rung that has all three pre-registered LRs: the best
final val loss over every LR it was run at, the relative gap from 6e-4's loss to
the best, pass/fail against 0.5%, and an edge flag when the best LR is the
smallest or largest that rung was run at (the optimum may lie outside them).
Then seed noise, the relative gap between seeds 0 and 1 for the middle rung at
6e-4, and the clipped fraction of each checked run.

The script reports; the judgement is yours. Run from the repo root:
    uv run python -m scripts.lr_check
"""

import csv
from collections import defaultdict

from src.run_plan import EXPECTED_RUNGS
from src.train import BAND_CSV, REPO_ROOT

PREREGISTERED_LRS = (2e-4, 6e-4, 1.8e-3)  # 6e-4 ×/÷ 3; a rung needs all three
DEFAULT_LR = 6e-4
PASS_GAP = 0.005  # 6e-4 passes if its loss is within 0.5% of the best LR's
NOISE_RUNG = EXPECTED_RUNGS[len(EXPECTED_RUNGS) // 2]  # (256, 8, 8)

Rung = tuple[int, int, int]


def relative_gap(loss: float, reference: float) -> float:
    return loss / reference - 1


def rung_label(rung: Rung) -> str:
    return "d{} L{} H{}".format(*rung)


def row_rung(row: dict[str, str]) -> Rung:
    return (int(row["d"]), int(row["L"]), int(row["H"]))


def load_runs() -> list[dict[str, str]]:
    with (REPO_ROOT / "results" / BAND_CSV).open(newline="") as f:
        return [row for row in csv.DictReader(f) if row["rehearsal"] == "False"]


def main() -> None:
    runs = load_runs()
    # (rung, lr, seed) -> final val loss; a duplicate (a rerun) is an error, not
    # a silent choice between two numbers
    losses: dict[tuple[Rung, float, int], float] = {}
    for row in runs:
        rung = row_rung(row)
        # csv writes str(float), which float() reads back exactly, so equal LRs
        # compare equal
        key = (rung, float(row["peak_lr"]), int(row["seed"]))
        if key in losses:
            raise ValueError(f"two runs for rung {rung}, LR {key[1]}, seed {key[2]}")
        losses[key] = float(row["final_val_loss"])

    by_rung: dict[Rung, dict[float, float]] = defaultdict(dict)
    for (rung, lr, seed), loss in losses.items():
        if seed == 0:
            by_rung[rung][lr] = loss
    # A rung at one LR is a sweep run, not part of the LR check
    by_rung = {rung: lrs for rung, lrs in by_rung.items() if len(lrs) > 1}
    lr_columns = sorted({lr for lr_losses in by_rung.values() for lr in lr_losses})

    print(
        f"{'rung':<12}"
        + "".join(f"{lr:>9.1e}" for lr in lr_columns)
        + f"{'best LR':>9}{'gap':>8}  {'0.5%':<5} edge"
    )
    for rung in sorted(by_rung):
        lr_losses = by_rung[rung]
        label = rung_label(rung)
        cells = "".join(
            f"{lr_losses[lr]:>9.4f}" if lr in lr_losses else f"{'-':>9}"
            for lr in lr_columns
        )
        missing = [lr for lr in PREREGISTERED_LRS if lr not in lr_losses]
        if missing:
            print(
                f"{label:<12}{cells}  incomplete: missing "
                + ", ".join(f"{lr:.1e}" for lr in missing)
            )
            continue
        best_lr = min(lr_losses, key=lr_losses.__getitem__)
        gap = relative_gap(lr_losses[DEFAULT_LR], lr_losses[best_lr])
        verdict = "pass" if gap <= PASS_GAP else "FAIL"
        edge = "EDGE" if best_lr in (min(lr_losses), max(lr_losses)) else "-"
        print(f"{label:<12}{cells}{best_lr:>9.1e}{gap:>8.2%}  {verdict:<5} {edge}")

    seeds = {
        seed: loss
        for (rung, lr, seed), loss in losses.items()
        if rung == NOISE_RUNG and lr == DEFAULT_LR
    }
    label = rung_label(NOISE_RUNG)
    if {0, 1} <= seeds.keys():
        noise = abs(seeds[0] - seeds[1]) / min(seeds[0], seeds[1])
        print(
            f"\nseed noise, {label} at {DEFAULT_LR:.0e}: seed 0 {seeds[0]:.4f}, "
            + f"seed 1 {seeds[1]:.4f}, gap {noise:.2%} "
            + f"(pass threshold {PASS_GAP:.1%})"
        )
    else:
        print(
            f"\nseed noise, {label} at {DEFAULT_LR:.0e}: needs seeds 0 and 1, "
            + f"have {sorted(seeds)}"
        )

    print(f"\n{'run_id':<34}{'peak_lr':>9}{'seed':>6}{'clipped':>9}")
    checked = [row for row in runs if row_rung(row) in by_rung]
    for row in sorted(
        checked, key=lambda r: (int(r["d"]), float(r["peak_lr"]), int(r["seed"]))
    ):
        print(
            f"{row['run_id']:<34}{float(row['peak_lr']):>9.1e}{row['seed']:>6}"
            + f"{float(row['clipped_fraction']):>9.1%}"
        )


if __name__ == "__main__":
    main()
