from __future__ import annotations

from pathlib import Path

import pandas as pd

from llm_ranking.config import MODES
from llm_ranking.rendering.latex import fmt_num, fmt_pct, short_model, write_lines
from llm_ranking.rendering.publish import write_text_to_many


LONGSHORT_METRICS = [
    ("LS_CAGR", "pct"),
    ("LS_Sharpe", "num"),
    ("LS_Sortino", "num"),
    ("LS_MDD", "pct"),
    ("Mean_IC", "ic"),
]


def _fmt_longshort(value: float, kind: str, bold: bool = False) -> str:
    if kind == "pct":
        return fmt_pct(value, bold, latex_sign=True)
    if kind == "ic":
        return fmt_num(value, bold, decimals=3, latex_sign=True)
    return fmt_num(value, bold, latex_sign=True)


def render_conservative_longshort(df: pd.DataFrame) -> str:
    work = df.copy()
    if "_row_id" not in work.columns:
        work["_row_id"] = work.index
    best = {}
    for mode in MODES:
        mode_rows = work.loc[work["Mode"].eq(mode)]
        for metric, _kind in LONGSHORT_METRICS:
            best[(mode, metric)] = int(mode_rows[metric].idxmax())

    rows_by_model = {
        model: group.set_index("Mode", drop=False)
        for model, group in work.groupby("Model", sort=False)
    }
    vendor_by_model = work.drop_duplicates("Model").set_index("Model")["Vendor"].to_dict()

    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{Conservative-window Q1$-$Q5 long--short performance and rank IC, net of 40\\,bps one-way costs on both legs. Bold indicates the best value in each column.}",
        "\\label{tab:longshort}",
        "\\scriptsize",
        "\\setlength{\\tabcolsep}{1.6pt}",
        "\\renewcommand{\\arraystretch}{0.95}",
        "\\resizebox{0.98\\textwidth}{!}{",
        "\\begin{tabular}{l rrrrr rrrrr rrrrr}",
        "\\toprule",
        "& \\multicolumn{5}{c}{\\textbf{Exp.\\ Return}} & \\multicolumn{5}{c}{\\textbf{Sharpe}} & \\multicolumn{5}{c}{\\textbf{Sortino}}  \\\\",
        "\\cmidrule(lr){2-6} \\cmidrule(lr){7-11} \\cmidrule(lr){12-16}",
        "\\textbf{Ranker} & CAGR & Shp & Sort & MDD & IC & CAGR & Shp & Sort & MDD & IC & CAGR & Shp & Sort & MDD & IC  \\\\",
        "\\midrule",
    ]

    last_vendor = None
    for model, group in rows_by_model.items():
        vendor = vendor_by_model[model]
        if last_vendor is not None and vendor != last_vendor:
            lines.append("\\midrule")
        last_vendor = vendor
        cells = [short_model(model)]
        for mode in MODES:
            row = group.loc[mode]
            row_id = int(row["_row_id"])
            for metric, kind in LONGSHORT_METRICS:
                cells.append(_fmt_longshort(float(row[metric]), kind, bold=row_id == best[(mode, metric)]))
        lines.append(" & ".join(cells) + " \\\\")

    lines.extend(["\\bottomrule", "\\end{tabular}", "}", "\\end{table*}", ""])
    return "\n".join(lines)


def write_conservative_longshort(df: pd.DataFrame, paths: list[Path]) -> None:
    write_text_to_many(render_conservative_longshort(df), paths)


def render_conservative_ensemble(df: pd.DataFrame) -> str:
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Conservative-window average-rank ensemble robustness. Consensus ranks are evaluated as the same net 40\\,bps Q1$-$Q5 portfolio.}",
        "\\label{tab:ensemble}",
        "\\scriptsize",
        "\\resizebox{\\columnwidth}{!}{%",
        "\\begin{tabular}{lrrrr}",
        "\\toprule",
        "\\textbf{Ensemble} & CAGR & Shp & MDD & IC \\\\",
        "\\midrule",
    ]
    for _, row in df.iterrows():
        lines.append(
            f"{row['Ensemble']} & {fmt_pct(row['LS_CAGR'])} & {fmt_num(row['LS_Sharpe'])} & "
            f"{fmt_pct(row['LS_MDD'])} & {fmt_num(row['Mean_IC'], decimals=3)} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", "%", "}", "\\end{table}", ""])
    return "\n".join(lines)


def write_conservative_ensemble(df: pd.DataFrame, paths: list[Path]) -> None:
    write_text_to_many(render_conservative_ensemble(df), paths)


def write_single_tex(tex: str, path: Path) -> None:
    write_lines(path, tex.splitlines())

