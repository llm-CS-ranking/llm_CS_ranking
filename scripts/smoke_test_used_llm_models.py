from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from conf.google_conf import GEMINI_API_KEY
from conf.openai_conf import OPENAI_API_KEY


USED_MODELS = [
    {"vendor": "OpenAI", "model": "gpt-4o-2024-05-13", "display": "GPT-4o"},
    {"vendor": "OpenAI", "model": "gpt-4o-mini-2024-07-18", "display": "GPT-4o mini"},
    {"vendor": "OpenAI", "model": "gpt-4.1-2025-04-14", "display": "GPT-4.1"},
    {"vendor": "OpenAI", "model": "gpt-4.1-mini-2025-04-14", "display": "GPT-4.1 mini"},
    {"vendor": "Google", "model": "gemini-2.5-flash", "display": "Gemini 2.5 Flash"},
    {"vendor": "Google", "model": "gemini-2.5-pro", "display": "Gemini 2.5 Pro"},
    {"vendor": "Google", "model": "gemini-3-flash-preview", "display": "Gemini 3 Flash"},
    {"vendor": "Google", "model": "gemini-3-pro-preview", "display": "Gemini 3 Pro"},
]

# Models the user asked to exclude from live calls.
KNOWN_DEPRECATED = {
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Smoke-test configured LLM models.")
    parser.add_argument("--dry-run", action="store_true", help="List models without API calls.")
    return parser.parse_args()


def smoke_openai(model: str) -> str:
    from openai import OpenAI

    client = OpenAI(api_key=OPENAI_API_KEY)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "Return exactly: OK"}],
        max_tokens=8,
        temperature=0,
    )
    return (response.choices[0].message.content or "").strip()


def smoke_google(model: str) -> str:
    from google import genai

    client = genai.Client(api_key=GEMINI_API_KEY)
    response = client.models.generate_content(
        model=model,
        contents="Return exactly: OK",
        config={"max_output_tokens": 8},
    )
    return (response.text or "").strip()


def main() -> None:
    args = parse_args()
    rows = []
    for item in USED_MODELS:
        model = item["model"]
        row = {**item, "status": None, "response": None, "error": None}
        if args.dry_run:
            row["status"] = "dry_run_not_called"
            rows.append(row)
            print(f"DRY RUN {item['vendor']} {model}")
            continue
        if model in KNOWN_DEPRECATED:
            row["status"] = "excluded_deprecated"
            row["error"] = KNOWN_DEPRECATED[model]
            rows.append(row)
            print(f"SKIP {model}: {row['error']}")
            continue
        try:
            if item["vendor"] == "OpenAI":
                text = smoke_openai(model)
            else:
                text = smoke_google(model)
            row["status"] = "available" if "OK" in text else "unexpected_response"
            row["response"] = text
            print(f"OK {model}: {text!r}")
        except Exception as exc:
            row["status"] = "unavailable"
            row["error"] = str(exc)[:500]
            print(f"FAIL {model}: {row['error']}")
        rows.append(row)

    if args.dry_run:
        print("DRY RUN complete: no API calls and no result file written.")
        return

    out = ROOT / "result" / "tables" / "used_llm_model_smoke_test.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
