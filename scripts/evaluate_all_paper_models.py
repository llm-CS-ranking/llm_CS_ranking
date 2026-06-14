from __future__ import annotations

import pickle
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate_retained_paper_models import evaluate_panel, load_market


RESULT_DIR = ROOT / "result"
TABLE_DIR = ROOT / "result" / "tables"
OUT_CSV = TABLE_DIR / "all_paper_models_results.csv"
OUT_TEX = TABLE_DIR / "all_paper_models_longshort.tex"

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


def main() -> None:
    market, _cid2ticker, _member = load_market()
    rows = []
    missing = []

    for model, cfg in MODEL_CONFIG.items():
        for mode in MODES:
            path = RESULT_DIR / f"{mode}_{model}_result.pkl"
            if not path.exists():
                missing.append(f"{mode}_{model}")
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
            rows.append(row)

    out = pd.DataFrame(rows)
    TABLE_DIR.mkdir(exist_ok=True)
    out.to_csv(OUT_CSV, index=False)

    pivot = out.pivot(index="Model", columns="Mode", values=["LS_CAGR", "LS_Sharpe", "LS_Sortino", "LS_MDD"])
    order = [cfg["display"] for cfg in MODEL_CONFIG.values()]
    pivot = pivot.reindex(order)

    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{Long--short (Q1$-$Q5) spread performance for all paper models.}",
        "\\label{tab:longshort-all-paper-models}",
        "\\small",
        "\\resizebox{\\textwidth}{!}{",
        "\\begin{tabular}{l rrrr rrrr rrrr}",
        "\\toprule",
        "& \\multicolumn{4}{c}{\\textbf{Exp. Return}} & \\multicolumn{4}{c}{\\textbf{Sharpe}} & \\multicolumn{4}{c}{\\textbf{Sortino}} \\\\",
        "\\cmidrule(lr){2-5} \\cmidrule(lr){6-9} \\cmidrule(lr){10-13}",
        "\\textbf{Model} & CAGR & Shp & Sort & MDD & CAGR & Shp & Sort & MDD & CAGR & Shp & Sort & MDD \\\\",
        "\\midrule",
    ]
    last_vendor = None
    for model_id, cfg in MODEL_CONFIG.items():
        model_name = cfg["display"]
        vendor = cfg["vendor"]
        if last_vendor is not None and vendor != last_vendor:
            lines.append("\\midrule")
        last_vendor = vendor
        cells = [model_name]
        for mode in MODES:
            cagr = pivot.loc[model_name, ("LS_CAGR", mode)] * 100
            shp = pivot.loc[model_name, ("LS_Sharpe", mode)]
            sortino = pivot.loc[model_name, ("LS_Sortino", mode)]
            mdd = pivot.loc[model_name, ("LS_MDD", mode)] * 100
            cells.extend([f"{cagr:.1f}", f"{shp:.2f}", f"{sortino:.2f}", f"{mdd:.1f}"])
        lines.append(" & ".join(cells) + " \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", "}", "\\end{table*}", ""])
    OUT_TEX.write_text("\n".join(lines), encoding="utf-8")

    print(f"Wrote {OUT_CSV}")
    print(f"Wrote {OUT_TEX}")
    print(f"Missing panels: {missing}")


if __name__ == "__main__":
    main()
