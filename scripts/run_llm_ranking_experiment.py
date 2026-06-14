from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd
import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from conf.mode import mode2prompt
from conf.model_list import MODEL_KNOWLEDGE_CUTOFF, MODEL_PRICING
from utils import (
    build_llm_payload,
    build_market_summary,
    count_tokens,
    llm_json_to_df,
    monthly_last_quarter_with_last4q,
    monthly_rank_panel,
)


TOP30_UNIVERSE = [
    "NVDA", "AAPL", "MSFT", "GOOG", "AMZN", "META", "TSLA", "COST", "AMD", "PLTR",
    "CSCO", "LRCX", "MU", "ISRG", "LIN", "NFLX", "QCOM", "ASML", "PEP", "TMUS",
    "TXN", "ADBE", "HON", "CMCSA", "AVGO", "SBUX", "BKNG", "AMGN", "INTU", "ADP",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run SRLLM listwise stock-ranking experiments from the command line."
    )
    parser.add_argument("--model", required=True, help="Provider model id, e.g. gpt-4.1-2025-04-14")
    parser.add_argument(
        "--provider",
        choices=["auto", "openai", "google", "anthropic"],
        default="auto",
        help="Model provider. Defaults to inference from model id.",
    )
    parser.add_argument(
        "--modes",
        default="return,sharpe,sortino",
        help="Comma-separated prediction modes: return,sharpe,sortino",
    )
    parser.add_argument("--data-dir", default="data/ndx_rolling_20260107")
    parser.add_argument("--backtest-end", default="2025-12")
    parser.add_argument("--knowledge-cutoff", default=None, help="Override model knowledge cutoff date")
    parser.add_argument(
        "--universe",
        choices=["top30", "rolling"],
        default="top30",
        help="top30 reproduces the paper setup; rolling uses all post-cutoff NDX members.",
    )
    parser.add_argument(
        "--top-n-by-mcap",
        type=int,
        default=None,
        help="Optional per-month market-cap filter for rolling universe to control prompt size.",
    )
    parser.add_argument("--start-month", default=None)
    parser.add_argument("--end-month", default=None)
    parser.add_argument("--max-months", type=int, default=None, help="Limit months for smoke tests")
    parser.add_argument("--top-k-items", type=int, default=40)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-output-tokens", type=int, default=2048)
    parser.add_argument("--google-thinking-budget", type=int, default=None)
    parser.add_argument("--google-response-mime-type", default=None)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--retry-sleep", type=float, default=5.0)
    parser.add_argument(
        "--horizon-text",
        default="next quarter",
        help="Text inserted into the ranking instruction. Use 'next rebalancing period' for the edited paper wording.",
    )
    parser.add_argument("--result-dir", default="result")
    parser.add_argument("--metadata-dir", default="result/metadata")
    parser.add_argument(
        "--input-set",
        choices=["both", "fundamentals_only", "market_only"],
        default="both",
        help="Input ablation setting. 'both' reproduces the paper setup.",
    )
    parser.add_argument("--ablation-result-dir", default="result/ablations")
    parser.add_argument("--ablation-metadata-dir", default="result/ablations/metadata")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing result pickle")
    parser.add_argument("--dry-run", action="store_true", help="Prepare payloads and estimate tokens without API calls")
    return parser.parse_args()


def infer_provider(model: str, provider: str) -> str:
    if provider != "auto":
        return provider
    lower = model.lower()
    if lower.startswith("gpt") or lower.startswith("o"):
        return "openai"
    if lower.startswith("gemini"):
        return "google"
    if lower.startswith("claude"):
        return "anthropic"
    raise ValueError(f"Cannot infer provider from model id: {model}")


def pricing_key(model: str) -> str:
    if model in MODEL_PRICING:
        return model
    for key in sorted(MODEL_PRICING, key=len, reverse=True):
        if model == key or model.startswith(f"{key}-"):
            return key
    return model


def cutoff_for_model(model: str, override: str | None) -> str:
    if override:
        return override
    if model in MODEL_KNOWLEDGE_CUTOFF:
        return MODEL_KNOWLEDGE_CUTOFF[model]
    for key in sorted(MODEL_KNOWLEDGE_CUTOFF, key=len, reverse=True):
        if model == key or model.startswith(f"{key}-"):
            return MODEL_KNOWLEDGE_CUTOFF[key]
    raise ValueError(f"No knowledge cutoff found for {model}; pass --knowledge-cutoff")


def load_raw_data(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    member = pd.read_csv(data_dir / "ndx_data_member.csv", header=0)
    fundamental = pd.read_csv(data_dir / "ndx_fundamental_data.csv", header=0)
    market = pd.read_csv(data_dir / "ndx_market_data.csv", header=0)
    trading = pd.read_csv(data_dir / "ndx_tradingiteminfo.csv", header=0)
    item_list = pd.read_csv(ROOT / "data" / "data_item_list.csv", header=0)
    return member, fundamental, market, trading, item_list


def prepare_panels(args: argparse.Namespace, knowledge_cutoff: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    data_dir = (ROOT / args.data_dir).resolve()
    member, fundamental, market, trading, item_list = load_raw_data(data_dir)

    member.columns = [c.upper() for c in member.columns]
    member = pd.merge(member, trading, on="TRADINGITEMID", how="left")
    member = member.dropna(subset=["TICKERSYMBOL", "COMPANYID"])
    member = member.set_index(["DATE", "TRADINGITEMID"]).sort_index()

    item_list.columns = ["DATAITEMID", "DATAITEMNAME"]
    fundamental = pd.merge(fundamental, item_list, on="DATAITEMID", how="left")
    fundamental = fundamental.drop("DATAITEMID", axis=1)
    fundamental = fundamental.set_index(["QUARTER", "COMPANYID"]).sort_index()

    member_universe = member.loc[member.index.get_level_values(0) > knowledge_cutoff].copy()
    if args.universe == "top30":
        member_universe = member_universe.loc[member_universe["TICKERSYMBOL"].isin(TOP30_UNIVERSE)].copy()

    mapping_df = (
        member_universe.reset_index()[["COMPANYID", "TRADINGITEMID", "TICKERSYMBOL", "DATE"]]
        .dropna()
        .sort_values(["DATE", "TICKERSYMBOL", "TRADINGITEMID"])
    )
    companyid2ticker = (
        mapping_df.drop_duplicates("COMPANYID", keep="last")
        .set_index("COMPANYID")["TICKERSYMBOL"]
        .to_dict()
    )
    tradingitemid2ticker = (
        mapping_df.drop_duplicates("TRADINGITEMID", keep="last")
        .set_index("TRADINGITEMID")["TICKERSYMBOL"]
        .to_dict()
    )

    fundamental_universe = fundamental.loc[fundamental.index.get_level_values(1).isin(companyid2ticker.keys())]
    min_quarter = str(pd.to_datetime(knowledge_cutoff).to_period("Q") - 4)
    fundamental_universe = fundamental_universe.loc[fundamental_universe.index.get_level_values(0) > min_quarter]
    fundamental_universe = monthly_last_quarter_with_last4q(fundamental_universe, asof=args.backtest_end)
    fundamental_universe["TICKERSYMBOL"] = fundamental_universe["COMPANYID"].map(companyid2ticker)
    fundamental_universe = fundamental_universe.dropna(subset=["TICKERSYMBOL"])
    fundamental_universe = fundamental_universe.drop("COMPANYID", axis=1)
    fundamental_universe = fundamental_universe.set_index(["MONTH", "TICKERSYMBOL"]).sort_index()

    market_universe = market.loc[market["DATE"] > knowledge_cutoff].copy()
    market_universe = market_universe.loc[market_universe["TRADINGITEMID"].isin(tradingitemid2ticker.keys())]
    market_universe["TICKERSYMBOL"] = market_universe["TRADINGITEMID"].map(tradingitemid2ticker)
    market_universe = market_universe.dropna(subset=["TICKERSYMBOL"])
    market_universe = market_universe.drop("TRADINGITEMID", axis=1)
    market_universe = market_universe.set_index(["DATE", "TICKERSYMBOL"]).sort_index()

    return fundamental_universe, market_universe


def filter_months(months: list[str], args: argparse.Namespace) -> list[str]:
    out = []
    for month in months:
        if args.start_month and month < args.start_month:
            continue
        if args.end_month and month > args.end_month:
            continue
        out.append(month)
    if args.max_months is not None:
        out = out[: args.max_months]
    return out


def top_names_by_mcap(market_panel: pd.DataFrame, month: str, n: int | None) -> set[str] | None:
    if n is None:
        return None
    m = market_panel.reset_index()
    m["DATE"] = pd.to_datetime(m["DATE"])
    month_start = pd.Period(month, freq="M").start_time
    prev = m.loc[m["DATE"] < month_start]
    if prev.empty:
        return None
    last_date = prev["DATE"].max()
    snap = prev.loc[prev["DATE"] == last_date].copy()
    snap["MKTCAP"] = pd.to_numeric(snap["MKTCAP"], errors="coerce")
    return set(snap.sort_values("MKTCAP", ascending=False)["TICKERSYMBOL"].head(n))


def make_instruction(mode: str, horizon_text: str) -> str:
    return f"""
Rank the following stock tickers by their expected {mode2prompt[mode]} over the {horizon_text}.
Base your judgment on reasonable assumptions and available general market knowledge.
Use ascending order (1 = highest expected performance).
If the payload includes a universe field, rank every ticker in that universe even when some data blocks are sparse or empty.
Do not return an empty list; ties are not allowed.

Output exactly one ticker per line in the following format:
1: {{Ticker1}}
2: {{Ticker2}}
3: {{Ticker3}}
...

Return JSON only: [{{"ticker":"...","rank":1}}, ...]
"""


def count_tokens_safe(text: str) -> int:
    try:
        return count_tokens(text)
    except Exception:
        # Keep dry-run usable on older tiktoken versions that lack newer model encodings.
        return max(1, len(text) // 4)


def maybe_filter_payload(payload: dict[str, Any], names: set[str] | None) -> dict[str, Any]:
    if names is None:
        return payload
    out = dict(payload)
    out["fundamentals"] = [r for r in payload["fundamentals"] if r.get("TICKER") in names]
    out["market"] = [r for r in payload["market"] if r.get("TICKER") in names]
    return out


def output_paths(args: argparse.Namespace, mode: str) -> tuple[Path, Path]:
    if args.input_set == "both":
        result_dir = (ROOT / args.result_dir).resolve()
        metadata_dir = (ROOT / args.metadata_dir).resolve()
        stem = f"{mode}_{args.model}"
    else:
        result_dir = (ROOT / args.ablation_result_dir).resolve()
        metadata_dir = (ROOT / args.ablation_metadata_dir).resolve()
        stem = f"{args.input_set}_{mode}_{args.model}"
    return result_dir / f"{stem}_result.pkl", metadata_dir / f"{stem}_metadata.json"


def make_payload(
    args: argparse.Namespace,
    month: str,
    fundamental_out: list[dict[str, Any]],
    market_out: list[dict[str, Any]],
    names: set[str] | None,
) -> dict[str, Any]:
    universe = sorted(names) if names is not None else TOP30_UNIVERSE
    payload = {
        "fundamentals": fundamental_out if args.input_set in {"both", "fundamentals_only"} else [],
        "market": market_out if args.input_set in {"both", "market_only"} else [],
        "notes": {
            "q_order": "Q1 oldest, Q4 most recent",
            "asof_month": month,
            "input_set": args.input_set,
        },
    }
    if args.input_set != "both":
        payload["universe"] = universe
    return maybe_filter_payload(payload, names)


def call_openai(model: str, instruction: str, prompt: str, args: argparse.Namespace) -> tuple[str, int, int]:
    from conf.openai_conf import OPENAI_API_KEY
    from openai import OpenAI

    client = OpenAI(api_key=OPENAI_API_KEY)
    messages = [
        {"role": "system", "content": instruction},
        {"role": "user", "content": prompt},
    ]
    if model.startswith("o") or model.startswith("gpt-5"):
        # Reasoning models reject max_tokens and may spend part of the output
        # budget on hidden reasoning tokens.
        result = client.chat.completions.create(
            messages=messages,
            model=model,
            max_completion_tokens=args.max_output_tokens,
        )
    else:
        result = client.chat.completions.create(
            messages=messages,
            model=model,
            temperature=args.temperature,
            max_tokens=args.max_output_tokens,
        )
    return (
        result.choices[0].message.content or "",
        int(result.usage.prompt_tokens or 0),
        int(result.usage.completion_tokens or 0),
    )


def call_google(model: str, instruction: str, prompt: str, args: argparse.Namespace) -> tuple[str, int, int]:
    from conf.google_conf import GEMINI_API_KEY
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=GEMINI_API_KEY)
    config_kwargs = {
        "system_instruction": instruction,
        "temperature": args.temperature,
        "max_output_tokens": args.max_output_tokens,
    }
    thinking_budget = getattr(args, "google_thinking_budget", None)
    if thinking_budget is not None:
        config_kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=thinking_budget)
    response_mime_type = getattr(args, "google_response_mime_type", None)
    if response_mime_type:
        config_kwargs["response_mime_type"] = response_mime_type
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(**config_kwargs),
    )
    usage = response.usage_metadata
    return (
        response.text or "",
        int(getattr(usage, "prompt_token_count", 0) or 0),
        int(getattr(usage, "candidates_token_count", 0) or 0),
    )


def call_anthropic(model: str, instruction: str, prompt: str, args: argparse.Namespace) -> tuple[str, int, int]:
    from conf.anthropic_conf import ANTHROPIC_API_KEY
    from anthropic import Anthropic

    client = Anthropic(api_key=ANTHROPIC_API_KEY)
    response = client.messages.create(
        model=model,
        max_tokens=args.max_output_tokens,
        temperature=args.temperature,
        system=instruction,
        messages=[{"role": "user", "content": [{"type": "text", "text": prompt}]}],
    )
    text = "".join(getattr(block, "text", "") for block in response.content)
    return text, int(response.usage.input_tokens or 0), int(response.usage.output_tokens or 0)


def call_model(provider: str, model: str, instruction: str, prompt: str, args: argparse.Namespace) -> tuple[str, int, int]:
    if provider == "openai":
        return call_openai(model, instruction, prompt, args)
    if provider == "google":
        return call_google(model, instruction, prompt, args)
    if provider == "anthropic":
        return call_anthropic(model, instruction, prompt, args)
    raise ValueError(f"Unsupported provider: {provider}")


def normalize_rank_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).lower() for c in df.columns]
    if "ticker" not in df.columns or "rank" not in df.columns:
        raise ValueError(f"Expected ticker/rank columns, got {list(df.columns)}")
    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
    df["rank"] = pd.to_numeric(df["rank"], errors="coerce")
    return df.dropna(subset=["ticker", "rank"])


def estimate_cost(input_tokens: int, output_tokens: int, model: str) -> float | None:
    pricing = MODEL_PRICING.get(pricing_key(model))
    if not pricing:
        return None
    return (input_tokens * pricing["input"] + output_tokens * pricing["output"]) / 1_000_000


def run_mode(args: argparse.Namespace, provider: str, mode: str, fundamental_panel: pd.DataFrame, market_panel: pd.DataFrame) -> None:
    result_path, metadata_path = output_paths(args, mode)
    result_dir = result_path.parent
    metadata_dir = metadata_path.parent
    result_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    if result_path.exists() and not args.overwrite and not args.dry_run:
        print(f"SKIP existing result: {result_path}")
        return

    instruction = make_instruction(mode, args.horizon_text)
    months = sorted(fundamental_panel.index.get_level_values(0).unique().astype(str))
    months = filter_months(months, args)

    all_result: dict[str, pd.DataFrame] = {}
    failed_months: list[dict[str, str]] = []
    total_prompt_tokens = 0
    total_completion_tokens = 0
    dry_run_tokens = 0

    for month in tqdm.tqdm(months, desc=f"{args.model} / {mode} / {args.input_set}"):
        names = top_names_by_mcap(market_panel, month, args.top_n_by_mcap) if args.universe == "rolling" else None
        fundamental_out = build_llm_payload(fundamental_panel, month, top_k_items=args.top_k_items)
        market_out = build_market_summary(market_panel, month)
        payload = make_payload(args, month, fundamental_out, market_out, names)
        prompt = "<DATA_JSON>\n" + json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n</DATA_JSON>"

        if args.dry_run:
            approx_tokens = count_tokens_safe(instruction + prompt)
            dry_run_tokens += approx_tokens
            print(
                f"DRY {month}: fundamentals={len(payload['fundamentals'])}, "
                f"market={len(payload['market'])}, approx_tokens={approx_tokens}"
            )
            continue

        last_error: Exception | None = None
        for attempt in range(1, args.max_retries + 1):
            try:
                text, prompt_tokens, completion_tokens = call_model(provider, args.model, instruction, prompt, args)
                all_result[month] = normalize_rank_df(llm_json_to_df(text))
                total_prompt_tokens += prompt_tokens
                total_completion_tokens += completion_tokens
                break
            except Exception as exc:
                last_error = exc
                print(f"ERROR {month} attempt {attempt}/{args.max_retries}: {exc}")
                time.sleep(args.retry_sleep)
        else:
            if args.input_set == "both":
                raise RuntimeError(f"Failed month {month}") from last_error
            failed_months.append({"month": month, "error": str(last_error)})
            print(f"SKIP failed ablation month {month}: {last_error}")

    if args.dry_run:
        cost = estimate_cost(dry_run_tokens, 0, args.model)
        print(f"DRY RUN complete: months={len(months)}, approx_input_tokens={dry_run_tokens}, estimated_input_cost_usd={cost}")
        return

    if not all_result:
        raise RuntimeError(f"No successful months for {args.model} / {mode} / {args.input_set}")

    panel = monthly_rank_panel(all_result)
    with result_path.open("wb") as f:
        pickle.dump(panel, f)

    metadata = {
        "model": args.model,
        "provider": provider,
        "mode": mode,
        "input_set": args.input_set,
        "months": months,
        "successful_months": sorted(all_result),
        "failed_months": failed_months,
        "result_path": str(result_path),
        "prompt_tokens": total_prompt_tokens,
        "completion_tokens": total_completion_tokens,
        "estimated_cost_usd": estimate_cost(total_prompt_tokens, total_completion_tokens, args.model),
        "args": vars(args),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {result_path}")
    print(f"Wrote {metadata_path}")
    print(
        f"Tokens: input={total_prompt_tokens:,}, output={total_completion_tokens:,}, "
        f"estimated_cost_usd={metadata['estimated_cost_usd']}"
    )


def main() -> None:
    args = parse_args()
    provider = infer_provider(args.model, args.provider)
    knowledge_cutoff = cutoff_for_model(args.model, args.knowledge_cutoff)
    print(f"model={args.model} provider={provider} knowledge_cutoff={knowledge_cutoff}")

    fundamental_panel, market_panel = prepare_panels(args, knowledge_cutoff)
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    invalid = sorted(set(modes) - set(mode2prompt))
    if invalid:
        raise ValueError(f"Invalid modes: {invalid}")
    for mode in modes:
        run_mode(args, provider, mode, fundamental_panel, market_panel)


if __name__ == "__main__":
    main()
