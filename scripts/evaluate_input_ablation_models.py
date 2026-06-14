from __future__ import annotations

import pickle
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate_retained_paper_models import evaluate_panel, load_market


RESULT_DIR = ROOT / "result" / "ablations"
TABLE_DIR = ROOT / "result" / "tables"
OUT_CSV = TABLE_DIR / "input_ablation_results.csv"
OUT_TEX = TABLE_DIR / "input_ablation_summary.tex"

INPUT_SETS = {
    "both": "Fund. + Market",
    "fundamentals_only": "Fund. only",
    "market_only": "Market only",
}
MODES = ["return", "sharpe", "sortino"]
MODE_LABELS = {"return": "Exp. Return", "sharpe": "Sharpe", "sortino": "Sortino"}

MODEL_CONFIG = {
    "gpt-4o-2024-05-13": {"display": "GPT-4o", "vendor": "OpenAI"},
    "gpt-4o-mini-2024-07-18": {"display": "GPT-4o mini", "vendor": "OpenAI"},
    "gpt-4.1-2025-04-14": {"display": "GPT-4.1", "vendor": "OpenAI"},
    "gpt-4.1-mini-2025-04-14": {"display": "GPT-4.1 mini", "vendor": "OpenAI"},
    "gpt-4.1-nano": {"display": "GPT-4.1 nano", "vendor": "OpenAI"},
    "o4-mini": {"display": "o4-mini", "vendor": "OpenAI"},
    "o3": {"display": "o3", "vendor": "OpenAI"},
    "gemini-2.5-flash": {"display": "Gemini 2.5 Flash", "vendor": "Google"},
    "gemini-2.5-pro": {"display": "Gemini 2.5 Pro", "vendor": "Google"},
    "claude-sonnet-4-20250514": {"display": "Claude Sonnet 4", "vendor": "Anthropic"},
    "claude-opus-4-20250514": {"display": "Claude Opus 4", "vendor": "Anthropic"},
}


def result_path(input_set: str, mode: str, model: str) -> Path:
    if input_set == "both":
        return ROOT / "result" / f"{mode}_{model}_result.pkl"
    return RESULT_DIR / f"{input_set}_{mode}_{model}_result.pkl"


def main() -> None:
    market, _cid2ticker, _member = load_market()
    existing_both = pd.read_csv(TABLE_DIR / "all_paper_models_results.csv")
    rows = []
    missing = []

    for input_set, input_label in INPUT_SETS.items():
        for model, cfg in MODEL_CONFIG.items():
            for mode in MODES:
                if input_set == "both":
                    prev = existing_both[
                        (existing_both["ModelID"] == model) & (existing_both["Mode"] == mode)
                    ]
                    if prev.empty:
                        missing.append(f"{input_set}:{mode}_{model}")
                        continue
                    row = prev.iloc[0].to_dict()
                else:
                    path = result_path(input_set, mode, model)
                    if not path.exists():
                        missing.append(f"{input_set}:{mode}_{model}")
                        continue
                    with path.open("rb") as f:
                        panel = pickle.load(f)
                    row = evaluate_panel(panel, market)
                    row.update(
                        {
                            "Model": cfg["display"],
                            "ModelID": model,
                            "Vendor": cfg["vendor"],
                            "Mode": mode,
                            "ModeLabel": MODE_LABELS[mode],
                        }
                    )
                row["InputSet"] = input_set
                row["InputLabel"] = input_label
                rows.append(row)

    out = pd.DataFrame(rows)
    TABLE_DIR.mkdir(exist_ok=True)
    out.to_csv(OUT_CSV, index=False)

    summary = (
        out.groupby(["InputSet", "InputLabel", "Mode", "ModeLabel"], sort=False)
        .agg(
            Mean_LS_CAGR=("LS_CAGR", "mean"),
            Mean_LS_Sharpe=("LS_Sharpe", "mean"),
            Mean_IC=("Mean_IC", "mean"),
            Best_LS_Sharpe=("LS_Sharpe", "max"),
            N=("LS_Sharpe", "count"),
        )
        .reset_index()
    )

    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Input ablation summary across LLM rankers. CAGR and IC are reported in percent; Sharpe is annualized.}",
        "\\label{tab:input-ablation}",
        "\\small",
        "\\begin{tabular}{llrrrrr}",
        "\\toprule",
        "\\textbf{Input} & \\textbf{Objective} & \\textbf{Mean CAGR} & \\textbf{Mean Shp} & \\textbf{Mean IC} & \\textbf{Best Shp} & \\textbf{N} \\\\",
        "\\midrule",
    ]
    for _, row in summary.iterrows():
        lines.append(
            f"{row['InputLabel']} & {row['ModeLabel']} & "
            f"{row['Mean_LS_CAGR'] * 100:.1f} & {row['Mean_LS_Sharpe']:.2f} & "
            f"{row['Mean_IC'] * 100:.1f} & {row['Best_LS_Sharpe']:.2f} & {int(row['N'])} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])
    OUT_TEX.write_text("\n".join(lines), encoding="utf-8")

    print(f"Wrote {OUT_CSV}")
    print(f"Wrote {OUT_TEX}")
    print(f"Missing panels: {missing}")


if __name__ == "__main__":
    main()
