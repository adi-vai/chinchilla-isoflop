"""Smoke test: a short CPU rehearsal of the smallest rung, end to end through the CLI."""

import csv
import math
import subprocess
import sys
from pathlib import Path

import pytest

from src.data import DATA_DIR
from src.run_plan import EXPECTED_RUNGS

REPO_ROOT = Path(__file__).resolve().parent.parent
STEPS = 20


@pytest.mark.skipif(
    not (DATA_DIR / "train.bin").exists() or not (DATA_DIR / "val.bin").exists(),
    reason="token memmaps missing: run `uv run python -m data.tinystories.prepare`",
)
def test_smallest_rung_rehearsal_trains_and_writes_its_row(tmp_path: Path) -> None:
    d, L, H = EXPECTED_RUNGS[0]
    # 2,048 tokens per step in micro-batches of 4 windows: accum_steps = 2, so the
    # accumulation path runs. 64 val windows keep the eval to a second.
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.train",
            "--rung",
            str(d),
            str(L),
            str(H),
            "--peak-lr",
            "6e-4",
            "--seed",
            "0",
            "--device",
            "cpu",
            "--max-steps",
            str(STEPS),
            "--rehearsal-tokens-per-step",
            "2048",
            "--micro-batch",
            "4",
            "--log-interval",
            "5",
            "--rehearsal-val-windows",
            "64",
            "--results-dir",
            str(tmp_path),
        ],
        cwd=REPO_ROOT,
        check=False,  # assert on returncode below, so a failure shows stderr
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr

    with (tmp_path / "rehearsals.csv").open(newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    row = rows[0]
    assert row["rehearsal"] == "True"
    assert int(row["steps"]) == STEPS
    assert int(row["accum_steps"]) == 2
    assert math.isfinite(float(row["final_val_loss"]))
    assert not (tmp_path / "band1.csv").exists()  # rehearsals never reach band1

    (step_csv,) = (tmp_path / "runs").glob("*.csv")
    with step_csv.open(newline="") as f:
        train_losses = [float(r["train_loss"]) for r in csv.DictReader(f)]
    assert len(train_losses) == STEPS // 5
    assert all(math.isfinite(loss) for loss in train_losses)
