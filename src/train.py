"""Train one IsoFLOP run (NOTES.md: optimizer, data order, eval, accumulation).

Run from the repo root:
    uv run python -m src.train --rung 160 6 5 --peak-lr 6e-4 --seed 0

--max-steps makes the run a rehearsal (pipeline check, not a sweep point): the
schedule is built for that length, and --rehearsal-tokens-per-step may shrink the
step and --rehearsal-val-windows cap the final eval. Rehearsal rows go to
results/rehearsals.csv, never results/band1.csv, so band1.csv holds sweep points only.
"""

import argparse
import contextlib
import csv
import dataclasses
import datetime
import logging
import math
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import torch

from src.data import DATA_SEED, TrainBatches, val_batches
from src.ladder import CONTEXT_LENGTH, make_config
from src.model import ModelConfig, TransformerLanguageModel
from src.run_plan import (
    EXPECTED_RUNGS,
    TOKENS_PER_STEP,
    RunPlan,
    band_compute_budget,
    plan_run,
)
from src.schedule import get_lr
from src.util import flops_per_token, matmul_params, total_params

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
BAND_CSV = "band1.csv"  # sweep points only
REHEARSAL_CSV = "rehearsals.csv"

# Optimizer (NOTES.md, decided 14 Nov)
BETAS = (0.9, 0.95)
EPS = 1e-8
WEIGHT_DECAY = 0.1
GRAD_CLIP = 1.0
MIN_LR_RATIO = 0.1  # min_lr = peak / 10


@dataclass(frozen=True)
class RunSetup:
    config: ModelConfig
    plan: RunPlan
    rehearsal: bool
    tokens_per_step: int
    micro_batch_size: int
    accum_steps: int
    device: torch.device
    autocast_dtype: torch.dtype | None  # None = plain fp32
    peak_lr: float
    min_lr: float
    val_windows: int | None  # None = the full val set


@dataclass(frozen=True)
class StepLog:
    step: int  # 0-indexed, the step just taken
    lr: float
    train_loss: float  # this step's mean over its micro-batches, un-scaled
    grad_norm: float  # pre-clip global norm
    tokens_per_second: float  # since the previous log


@dataclass(frozen=True)
class TrainResult:
    clipped_fraction: float
    train_seconds: float  # the step loop only; excludes setup and final eval


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    _ = parser.add_argument(
        "--rung", type=int, nargs=3, required=True, metavar=("D", "L", "H")
    )
    _ = parser.add_argument("--peak-lr", type=float, required=True)
    _ = parser.add_argument(
        "--seed", type=int, required=True, help="model initialisation"
    )
    _ = parser.add_argument(
        "--data-seed",
        type=int,
        default=DATA_SEED,
        help="data order; the same for every rung",
    )
    _ = parser.add_argument(
        "--micro-batch", type=int, default=64, help="sequences per forward pass"
    )
    _ = parser.add_argument(
        "--device",
        choices=("cuda", "mps", "cpu"),
        help="default: cuda, else mps, else cpu",
    )
    _ = parser.add_argument(
        "--max-steps", type=int, help="rehearsal: override the planned step count"
    )
    _ = parser.add_argument(
        "--rehearsal-tokens-per-step",
        type=int,
        help=f"rehearsal only: tokens per step instead of {TOKENS_PER_STEP:,}",
    )
    _ = parser.add_argument(
        "--log-interval", type=int, default=50, help="steps between host syncs/logs"
    )
    _ = parser.add_argument(
        "--rehearsal-val-windows",
        type=int,
        help="rehearsal only: evaluate the first N val windows, not all",
    )
    _ = parser.add_argument(
        "--peak-flops", type=float, help="device peak FLOP/s, for MFU (else blank)"
    )
    _ = parser.add_argument("--results-dir", type=Path, default=REPO_ROOT / "results")
    return parser.parse_args()


def pick_device(requested: str | None) -> torch.device:
    if requested is not None:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def accumulation_steps(tokens_per_step: int, micro_batch_size: int) -> int:
    tokens_per_micro_batch = micro_batch_size * CONTEXT_LENGTH
    if tokens_per_step % tokens_per_micro_batch != 0:
        raise ValueError(
            f"micro-batch {micro_batch_size} x T {CONTEXT_LENGTH} doesn't divide "
            + f"{tokens_per_step:,} tokens per step"
        )
    return tokens_per_step // tokens_per_micro_batch


def rehearsal_plan(config: ModelConfig, steps: int, tokens_per_step: int) -> RunPlan:
    # Rehearsals only: plan_run's arithmetic at a forced length, warmup 5% with no
    # floor (the floor would exceed a short run). Real runs never come here.
    realized_tokens = steps * tokens_per_step
    return RunPlan(
        target_tokens=realized_tokens,
        steps=steps,
        realized_tokens=realized_tokens,
        realized_compute=realized_tokens * flops_per_token(config),
        warmup_steps=math.ceil(0.05 * steps),
    )


def build_optimizer(
    model: torch.nn.Module, peak_lr: float, device: torch.device
) -> torch.optim.AdamW:
    # Decay matrices only (p.dim() >= 2): Linear weights, tied embedding, position
    # embedding. parameters() yields the tied matrix once.
    params = list(model.parameters())
    decay = [p for p in params if p.dim() >= 2]
    no_decay = [p for p in params if p.dim() < 2]
    logger.info(
        f"weight decay on {len(decay)} tensors ({sum(p.numel() for p in decay):,} "
        + f"params), none on {len(no_decay)} ({sum(p.numel() for p in no_decay):,})"
    )
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": WEIGHT_DECAY},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=peak_lr,  # overwritten every step from get_lr
        betas=BETAS,
        eps=EPS,
        fused=device.type == "cuda",
    )


def set_up_run(args: argparse.Namespace) -> RunSetup:
    rung = tuple(args.rung)
    if rung not in EXPECTED_RUNGS:
        raise ValueError(f"{rung} is not a band-1 rung: {EXPECTED_RUNGS}")
    config = make_config(*rung)
    rehearsal = args.max_steps is not None
    if args.rehearsal_tokens_per_step is not None and not rehearsal:
        raise ValueError("--rehearsal-tokens-per-step needs --max-steps")
    if args.rehearsal_val_windows is not None and not rehearsal:
        raise ValueError("--rehearsal-val-windows needs --max-steps")
    if rehearsal:
        tokens_per_step = args.rehearsal_tokens_per_step or TOKENS_PER_STEP
        plan = rehearsal_plan(config, args.max_steps, tokens_per_step)
    else:
        tokens_per_step = TOKENS_PER_STEP
        compute_budget = band_compute_budget(make_config(*EXPECTED_RUNGS[0]))
        plan = plan_run(config, compute_budget, TOKENS_PER_STEP)
    device = pick_device(args.device)
    return RunSetup(
        config=config,
        plan=plan,
        rehearsal=rehearsal,
        tokens_per_step=tokens_per_step,
        micro_batch_size=args.micro_batch,
        accum_steps=accumulation_steps(tokens_per_step, args.micro_batch),
        device=device,
        autocast_dtype=torch.bfloat16 if device.type == "cuda" else None,
        peak_lr=args.peak_lr,
        min_lr=args.peak_lr * MIN_LR_RATIO,
        val_windows=args.rehearsal_val_windows,
    )


def train(
    model: torch.nn.Module,
    optimizer: torch.optim.AdamW,
    batches: TrainBatches,
    setup: RunSetup,
    autocast: contextlib.AbstractContextManager[None],
    log_interval: int,
    on_log: Callable[[StepLog], None],
) -> TrainResult:
    plan = setup.plan
    device = setup.device
    # Running totals stay on the device: reading them (.item()) forces the host to
    # wait for the GPU, so that only happens at logging intervals
    num_clipped = torch.zeros((), dtype=torch.int64, device=device)
    model.train()
    start = last_log_time = time.perf_counter()
    last_log_step = -1
    for step in range(plan.steps):
        lr = get_lr(step, plan.steps, plan.warmup_steps, setup.peak_lr, setup.min_lr)
        for group in optimizer.param_groups:
            group["lr"] = lr

        step_loss = torch.zeros((), device=device)
        for _ in range(setup.accum_steps):
            inputs, targets = batches.next_batch()
            with autocast:
                _, loss = model(inputs, targets)
            step_loss += loss.detach()
            # .grad sums over micro-batches; scaling makes it the mean (NOTES.md)
            (loss / setup.accum_steps).backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        num_clipped += grad_norm > GRAD_CLIP
        _ = optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        if (step + 1) % log_interval == 0 or step == plan.steps - 1:
            train_loss = (step_loss / setup.accum_steps).item()  # host sync
            if not math.isfinite(train_loss):
                raise RuntimeError(f"non-finite train loss {train_loss} at step {step}")
            now = time.perf_counter()
            tokens = (step - last_log_step) * setup.tokens_per_step
            on_log(
                StepLog(
                    step=step,
                    lr=lr,
                    train_loss=train_loss,
                    grad_norm=grad_norm.item(),
                    tokens_per_second=tokens / (now - last_log_time),
                )
            )
            last_log_time, last_log_step = now, step

    return TrainResult(
        clipped_fraction=num_clipped.item() / plan.steps,
        train_seconds=time.perf_counter() - start,
    )


def evaluate(
    model: torch.nn.Module,
    setup: RunSetup,
    autocast: contextlib.AbstractContextManager[None],
) -> tuple[float, int]:
    # Forward only, after the last step: outside the compute budget C, so nothing
    # here is added to realised compute
    model.eval()
    loss_sum = torch.zeros((), device=setup.device)
    windows = 0
    with torch.no_grad():
        for inputs, targets in val_batches(
            CONTEXT_LENGTH, setup.micro_batch_size, setup.device, setup.val_windows
        ):
            with autocast:
                _, loss = model(inputs, targets)
            # loss is the mean over this batch's B·T tokens; weight by B so the
            # smaller last batch counts per token, like every other
            batch_windows = inputs.shape[0]
            loss_sum += loss.float() * batch_windows
            windows += batch_windows
    model.train()
    return (loss_sum / windows).item(), windows  # one host sync for the whole eval


def git_commit() -> str:
    # Short hash of HEAD, "-dirty" if tracked files have uncommitted changes
    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()

    try:
        sha = git("rev-parse", "--short", "HEAD")
        dirty = git("status", "--porcelain", "--untracked-files=no")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return f"{sha}-dirty" if dirty else sha


def append_row(path: Path, row: dict[str, object]) -> None:
    # Append one CSV row, writing the header if the file is new. An existing file
    # with different columns is an error, never a silently misaligned row.
    path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not path.exists()
    if not is_new:
        with path.open(newline="") as f:
            header = next(csv.reader(f))
        if header != list(row):
            raise ValueError(f"{path} has columns {header}, expected {list(row)}")
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        if is_new:
            writer.writeheader()
        writer.writerow(row)


def main() -> None:
    started = time.perf_counter()
    args = parse_args()
    setup = set_up_run(args)
    # Seed after the imports (importing src.model calls manual_seed), and build the
    # model on the CPU so a seed gives the same initial weights on every device
    _ = torch.manual_seed(args.seed)
    model = TransformerLanguageModel(setup.config).to(setup.device)
    optimizer = build_optimizer(model, setup.peak_lr, setup.device)
    batches = TrainBatches(
        CONTEXT_LENGTH, setup.micro_batch_size, args.data_seed, setup.device
    )

    windows_needed = setup.plan.steps * setup.accum_steps * setup.micro_batch_size
    if windows_needed > batches.num_windows:
        raise ValueError(
            f"run needs {windows_needed:,} windows, train has {batches.num_windows:,}"
        )

    autocast = (
        torch.autocast(setup.device.type, dtype=setup.autocast_dtype)
        if setup.autocast_dtype is not None
        else contextlib.nullcontext()
    )
    logger.info(f"{'REHEARSAL ' if setup.rehearsal else ''}run: {setup}")

    d, L, H = args.rung
    timestamp = datetime.datetime.now(datetime.UTC).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{timestamp}_d{d}_L{L}_H{H}_s{args.seed}"
    if setup.rehearsal:
        run_id += "_rehearsal"
    step_csv = args.results_dir / "runs" / f"{run_id}.csv"

    def log_step(entry: StepLog) -> None:
        logger.info(
            f"step {entry.step:>6,}/{setup.plan.steps:,}  lr {entry.lr:.2e}  "
            + f"loss {entry.train_loss:.4f}  grad norm {entry.grad_norm:.3f}  "
            + f"{entry.tokens_per_second:,.0f} tok/s"
        )
        append_row(step_csv, dataclasses.asdict(entry))

    result = train(
        model, optimizer, batches, setup, autocast, args.log_interval, log_step
    )
    val_loss, val_windows = evaluate(model, setup, autocast)

    fpt = flops_per_token(setup.config)
    tokens_per_second = setup.plan.realized_tokens / result.train_seconds
    run_csv = args.results_dir / (REHEARSAL_CSV if setup.rehearsal else BAND_CSV)
    append_row(
        run_csv,
        {
            "run_id": run_id,
            "d": d,
            "L": L,
            "H": H,
            "total_params": total_params(setup.config),
            "matmul_params": matmul_params(setup.config),
            "flops_per_token": fpt,
            "target_tokens": setup.plan.target_tokens,
            "steps": setup.plan.steps,
            "realized_tokens": setup.plan.realized_tokens,
            "realized_compute": setup.plan.realized_compute,  # training only
            "warmup_steps": setup.plan.warmup_steps,
            "peak_lr": setup.peak_lr,
            "min_lr": setup.min_lr,
            "seed": args.seed,
            "data_seed": args.data_seed,
            "micro_batch": setup.micro_batch_size,
            "accum_steps": setup.accum_steps,
            "tokens_per_step": setup.tokens_per_step,
            "device": setup.device.type,
            "dtype": "bf16-autocast" if setup.autocast_dtype is not None else "fp32",
            "final_val_loss": val_loss,
            "val_windows": val_windows,
            "wall_seconds": time.perf_counter() - started,
            "tokens_per_second": tokens_per_second,
            "mfu": tokens_per_second * fpt / args.peak_flops if args.peak_flops else "",
            "clipped_fraction": result.clipped_fraction,
            "git_commit": git_commit(),
            "rehearsal": setup.rehearsal,
        },
    )
    logger.info(
        f"{run_id}: final val loss {val_loss:.4f} on {val_windows:,} windows, "
        + f"{tokens_per_second:,.0f} tok/s; row in {run_csv}"
    )


if __name__ == "__main__":
    main()
