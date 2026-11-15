"""One-off migration: add the gpu_name and grad_clip columns to existing results.

Run ONLY after the final band1.csv (all 10 rows) has been copied back from the
pod; a later copy would overwrite the migrated file. Dry run by default; --apply
writes. Refuses a file that already has the columns, or a row on an unexpected
device.

Run from the repo root: `uv run python -m scripts.backfill_gpu_name_grad_clip`.
"""

import argparse
import csv
from pathlib import Path

from src.train import REPO_ROOT, RUN_COLUMNS

RESULTS = REPO_ROOT / "results"
# file -> (the device every row must be on, gpu_name to fill)
FILL = {
    "band1.csv": ("cuda", "NVIDIA A100-SXM4-80GB"),  # torch.cuda.get_device_name()
    "rehearsals.csv": ("mps", "Apple M4"),  # sysctl machdep.cpu.brand_string
}
GRAD_CLIP = "1.0"  # every run so far used the old fixed clip


def migrate(path: Path, device: str, gpu_name: str, apply: bool) -> None:
    if not path.exists():
        print(f"{path.name}: missing, skipped (copy it back from the pod first)")
        return
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if rows and ("gpu_name" in rows[0] or "grad_clip" in rows[0]):
        raise SystemExit(f"{path.name}: already migrated")
    for row in rows:
        if row["device"] != device:
            raise SystemExit(
                f"{path.name}: {row['run_id']} ran on {row['device']}, expected {device}"
            )
        row["gpu_name"] = gpu_name
        row["grad_clip"] = GRAD_CLIP
        assert set(row) == set(RUN_COLUMNS), set(row) ^ set(RUN_COLUMNS)
    print(
        f"{path.name}: {len(rows)} rows -> gpu_name={gpu_name}, grad_clip={GRAD_CLIP}"
    )
    if apply:
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=RUN_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    _ = parser.add_argument(
        "--apply", action="store_true", help="write the files (default: dry run)"
    )
    args = parser.parse_args()
    for name, (device, gpu_name) in FILL.items():
        migrate(RESULTS / name, device, gpu_name, args.apply)
    if not args.apply:
        print("dry run: nothing written; rerun with --apply")


if __name__ == "__main__":
    main()
