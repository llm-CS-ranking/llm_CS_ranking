from __future__ import annotations

from pathlib import Path

import pandas as pd

from llm_ranking.config import INPUT_SETS, MODES
from llm_ranking.io.panels import input_prefix
from llm_ranking.rendering.latex import best_value, fmt_num, fmt_pct, is_close, write_lines


INPUT_PREFIXES = [input_prefix(label) for label in INPUT_SETS.values()]
METRICS = ["LS_CAGR", "LS_Sharpe", "Mean_IC"]


def render_input_ablation_appendix_table(df: pd.DataFrame, path: Path) -> None:
    lines = [
        "\\begin{table}[H]",
        "\\centering",
        "\\caption{Input ablation results by model and objective. CAGR and IC are reported in percent; Sharpe is annualized; bold indicates the best input setting for each metric within each model--objective row.}",
        "\\label{tab:input-ablation-appendix}",
        "\\scriptsize",
        "\\setlength{\\tabcolsep}{2.5pt}",
        "\\resizebox{\\textwidth}{!}{",
        "\\begin{tabular}{ll rrr rrr rrr}",
        "\\toprule",
        "& & \\multicolumn{3}{c}{\\textbf{Fund. + Market}} & \\multicolumn{3}{c}{\\textbf{Fund. only}} & \\multicolumn{3}{c}{\\textbf{Market only}} \\\\",
        "\\cmidrule(lr){3-5} \\cmidrule(lr){6-8} \\cmidrule(lr){9-11}",
        "\\textbf{Objective} & \\textbf{Model} & CAGR & Shp & IC & CAGR & Shp & IC & CAGR & Shp & IC \\\\",
        "\\midrule",
    ]
    current_mode = None
    ordered = df.copy()
    if "Mode" in ordered:
        ordered["_RowOrder"] = range(len(ordered))
        ordered["ModeOrder"] = ordered["Mode"].map({mode: i for i, mode in enumerate(MODES)})
        ordered = ordered.sort_values(["ModeOrder", "_RowOrder"]).drop(columns=["ModeOrder", "_RowOrder"])

    for _, row in ordered.iterrows():
        if current_mode is not None and current_mode != row["Mode"]:
            lines.append("\\midrule")
        current_mode = row["Mode"]
        model = str(row["Model"]).replace("Gemini", "Gem.").replace("Claude", "Cl.")
        cells = [row["Objective"], model]
        best_in_row = {
            metric: best_value([row[f"{prefix}_{metric}"] for prefix in INPUT_PREFIXES])
            for metric in METRICS
        }
        for prefix in INPUT_PREFIXES:
            cells.extend(
                [
                    fmt_pct(
                        row[f"{prefix}_LS_CAGR"],
                        is_close(row[f"{prefix}_LS_CAGR"], best_in_row["LS_CAGR"]),
                    ),
                    fmt_num(
                        row[f"{prefix}_LS_Sharpe"],
                        is_close(row[f"{prefix}_LS_Sharpe"], best_in_row["LS_Sharpe"]),
                    ),
                    fmt_pct(
                        row[f"{prefix}_Mean_IC"],
                        is_close(row[f"{prefix}_Mean_IC"], best_in_row["Mean_IC"]),
                    ),
                ]
            )
        lines.append(" & ".join(cells) + " \\\\")

    lines.extend(["\\bottomrule", "\\end{tabular}", "}", "\\end{table}", ""])
    write_lines(path, lines)

