from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

MODELS = [
    "gpt-4o-2024-05-13",
    "gpt-4o-mini-2024-07-18",
    "gpt-4.1-2025-04-14",
    "gpt-4.1-mini-2025-04-14",
    "gemini-2.5-flash",
    "gemini-2.5-pro",
    "gemini-3-flash-preview",
    "gemini-3-pro-preview",
    "claude-haiku-4-5-20251001",
    "claude-sonnet-4-20250514",
    "claude-opus-4-20250514",
]

INPUT_SETS = ["fundamentals_only", "market_only"]
LOCK_PATH = ROOT / "result" / "ablations" / ".run_input_ablation_experiments.lock"


def is_ablation_process(pid: int) -> bool:
    cmdline_path = Path(f"/proc/{pid}/cmdline")
    try:
        cmdline = cmdline_path.read_text(encoding="utf-8").replace("\x00", " ")
    except FileNotFoundError:
        return False
    return "run_input_ablation_experiments.py" in cmdline or "run_llm_ranking_experiment.py" in cmdline


@contextmanager
def single_process_lock():
    LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(LOCK_PATH, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        try:
            pid = int(LOCK_PATH.read_text(encoding="utf-8").strip())
        except ValueError:
            pid = -1

        if pid > 0 and is_ablation_process(pid):
            raise

        LOCK_PATH.unlink(missing_ok=True)
        try:
            fd = os.open(LOCK_PATH, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            # Another driver acquired the lock between stale-lock cleanup and retry.
            raise
    try:
        os.write(fd, str(os.getpid()).encode("utf-8"))
        yield
    finally:
        os.close(fd)
        LOCK_PATH.unlink(missing_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Run LLM input ablation experiments with model-level retries.")
    parser.add_argument("--models", default=",".join(MODELS), help="Comma-separated model ids.")
    parser.add_argument("--input-sets", default=",".join(INPUT_SETS), help="Comma-separated input sets.")
    parser.add_argument("--modes", default="return,sharpe,sortino")
    parser.add_argument("--max-months", type=int, default=None, help="Limit months passed to each model run.")
    parser.add_argument("--dry-run", action="store_true", help="Prepare payloads without API calls.")
    parser.add_argument("--max-model-attempts", type=int, default=3)
    parser.add_argument("--retry-sleep", type=float, default=30.0)
    parser.add_argument(
        "--month-retries",
        type=int,
        default=1,
        help="Per-month retries inside each model run. Model-level retries handle full reruns.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    input_sets = [s.strip() for s in args.input_sets.split(",") if s.strip()]
    failed: list[tuple[str, str]] = []

    try:
        with single_process_lock():
            for input_set in input_sets:
                for model in models:
                    cmd = [
                        sys.executable,
                        str(ROOT / "scripts" / "run_llm_ranking_experiment.py"),
                        "--model",
                        model,
                        "--modes",
                        args.modes,
                        "--input-set",
                        input_set,
                        "--max-retries",
                        str(args.month_retries),
                    ]
                    if args.max_months is not None:
                        cmd.extend(["--max-months", str(args.max_months)])
                    if args.dry_run:
                        cmd.append("--dry-run")

                    for attempt in range(1, args.max_model_attempts + 1):
                        print("\n" + "=" * 100, flush=True)
                        print(
                            f"RUN input_set={input_set} model={model} attempt={attempt}/{args.max_model_attempts}",
                            flush=True,
                        )
                        print("=" * 100, flush=True)
                        result = subprocess.run(cmd, cwd=ROOT)
                        if result.returncode == 0:
                            break
                        print(
                            f"FAILED input_set={input_set} model={model} attempt={attempt}/{args.max_model_attempts} "
                            f"returncode={result.returncode}",
                            flush=True,
                        )
                        if attempt < args.max_model_attempts:
                            time.sleep(args.retry_sleep)
                    else:
                        failed.append((input_set, model))
                        print(f"SKIP unresolved input_set={input_set} model={model} after model-level retries", flush=True)
    except FileExistsError:
        raise SystemExit(f"Another input ablation driver is already running: {LOCK_PATH}") from None

    if failed:
        print("\nUnresolved failures after retries:", flush=True)
        for input_set, model in failed:
            print(f"- {input_set}: {model}", flush=True)

    print("\nAll requested input ablation experiments completed.", flush=True)


if __name__ == "__main__":
    main()
