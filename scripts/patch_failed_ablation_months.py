from __future__ import annotations

import argparse
import importlib.util
import json
import pickle
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

spec = importlib.util.spec_from_file_location("runner", ROOT / "scripts" / "run_llm_ranking_experiment.py")
runner = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(runner)

from conf.mode import mode2prompt
from conf.model_list import MODEL_KNOWLEDGE_CUTOFF
from utils import build_llm_payload, build_market_summary, llm_json_to_df, monthly_rank_panel


METADATA_DIR = ROOT / "result" / "ablations" / "metadata"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Patch failed ablation months by rerunning only failed months.")
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--retry-sleep", type=float, default=5.0)
    parser.add_argument("--max-output-tokens", type=int, default=8192)
    parser.add_argument("--google-thinking-budget", type=int, default=None)
    parser.add_argument("--google-response-mime-type", default=None)
    parser.add_argument("--models", default="", help="Optional comma-separated model filter.")
    parser.add_argument("--dry-run", action="store_true", help="Prepare failed-month payloads without API calls or writes.")
    return parser.parse_args()


def parse_metadata_name(path: Path) -> tuple[str, str, str]:
    stem = path.name.removesuffix("_metadata.json")
    for input_set in ("fundamentals_only", "market_only"):
        prefix = f"{input_set}_"
        if stem.startswith(prefix):
            rest = stem[len(prefix) :]
            for mode in mode2prompt:
                mode_prefix = f"{mode}_"
                if rest.startswith(mode_prefix):
                    return input_set, mode, rest[len(mode_prefix) :]
    raise ValueError(f"Cannot parse metadata name: {path.name}")


def normalize_rank(text: str) -> pd.DataFrame:
    return runner.normalize_rank_df(llm_json_to_df(text))


def main() -> None:
    args = parse_args()
    model_filter = {m.strip() for m in args.models.split(",") if m.strip()}
    patched = []
    unresolved = []

    for metadata_path in sorted(METADATA_DIR.glob("*_metadata.json")):
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        failed_months = [item["month"] for item in metadata.get("failed_months", [])]
        if not failed_months:
            continue

        input_set, mode, model = parse_metadata_name(metadata_path)
        if model_filter and model not in model_filter:
            continue

        provider = runner.infer_provider(model, "auto")
        knowledge_cutoff = MODEL_KNOWLEDGE_CUTOFF[model]
        run_args = argparse.Namespace(**metadata["args"])
        run_args.max_retries = args.max_retries
        run_args.retry_sleep = args.retry_sleep
        run_args.max_output_tokens = args.max_output_tokens
        run_args.google_thinking_budget = args.google_thinking_budget
        run_args.google_response_mime_type = args.google_response_mime_type

        fundamental_panel, market_panel = runner.prepare_panels(run_args, knowledge_cutoff)
        instruction = runner.make_instruction(mode, run_args.horizon_text)
        result_path = Path(metadata["result_path"])
        with result_path.open("rb") as f:
            panel = pickle.load(f)
        monthly_frames = {
            str(month): panel.loc[[month]].T.reset_index().rename(columns={"ticker": "ticker", month: "rank"})
            for month in panel.index.astype(str)
        }

        still_failed = []
        added_months = []
        for month in failed_months:
            print(f"PATCH {model} / {input_set} / {mode} / {month}", flush=True)
            names = runner.top_names_by_mcap(market_panel, month, run_args.top_n_by_mcap) if run_args.universe == "rolling" else None
            payload = runner.make_payload(
                run_args,
                month,
                build_llm_payload(fundamental_panel, month, top_k_items=run_args.top_k_items),
                build_market_summary(market_panel, month),
                names,
            )
            prompt = "<DATA_JSON>\n" + json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n</DATA_JSON>"
            if args.dry_run:
                approx_tokens = runner.count_tokens_safe(prompt)
                print(f"  DRY RUN payload_tokens={approx_tokens}", flush=True)
                still_failed.append({"month": month, "error": "dry_run_not_called"})
                continue
            last_error: Exception | None = None
            for attempt in range(1, args.max_retries + 1):
                try:
                    print(f"  attempt {attempt}/{args.max_retries}", flush=True)
                    text, prompt_tokens, completion_tokens = runner.call_model(provider, model, instruction, prompt, run_args)
                    monthly_frames[month] = normalize_rank(text)
                    metadata["prompt_tokens"] = int(metadata.get("prompt_tokens", 0)) + prompt_tokens
                    metadata["completion_tokens"] = int(metadata.get("completion_tokens", 0)) + completion_tokens
                    added_months.append(month)
                    print(f"  success prompt_tokens={prompt_tokens} completion_tokens={completion_tokens}", flush=True)
                    break
                except Exception as exc:  # noqa: BLE001
                    last_error = exc
                    print(f"  error: {exc}", flush=True)
                    time.sleep(args.retry_sleep)
            else:
                still_failed.append({"month": month, "error": str(last_error)})

        if added_months:
            updated_panel = monthly_rank_panel(monthly_frames)
            with result_path.open("wb") as f:
                pickle.dump(updated_panel, f)
            metadata["successful_months"] = sorted(set(metadata.get("successful_months", [])) | set(added_months))
            metadata["failed_months"] = still_failed
            metadata["patched_months"] = sorted(set(metadata.get("patched_months", [])) | set(added_months))
            metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
            patched.append((metadata_path.name, added_months, still_failed))
        elif still_failed:
            unresolved.append((metadata_path.name, still_failed))

    print("patched_jobs", len(patched))
    for name, months, still_failed in patched:
        print(name, "patched=", months, "remaining=", still_failed)
    print("unresolved_jobs", len(unresolved))
    for name, still_failed in unresolved:
        print(name, "remaining=", still_failed)


if __name__ == "__main__":
    main()
