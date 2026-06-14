from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from llm_ranking.config import COST_BPS, COST_GRID
from llm_ranking.rendering.latex import fmt_num, fmt_pct, is_close, write_lines


def render_common_window_table(df: pd.DataFrame, path: Path) -> None:
    rows = df.loc[df["Panel"].isin(["LLM", "Baseline"])].copy()
    rows = rows.loc[(rows["Mode"] == "return") | (rows["Panel"] == "Baseline")]
    rows["SortKey"] = rows["Panel"].map({"Baseline": 0, "LLM": 1}).fillna(2)
    rows = rows.sort_values(["SortKey", "Vendor", "Order"])
    best_values = {
        "LS_CAGR": rows["LS_CAGR"].max(),
        "LS_Sharpe": rows["LS_Sharpe"].max(),
        "LS_Sortino": rows["LS_Sortino"].max(),
        "LS_MDD": rows["LS_MDD"].max(),
        "Mean_IC": rows["Mean_IC"].max(),
    }

    lines = [
        "\\begin{table}[H]",
        "\\centering",
        "\\caption{Common-calendar-window long--short performance. All rankers are restricted to the same common ranking window with a 2-day execution lag and corrected 40 bps one-way costs on both long and short legs. CAGR and IC are reported in percent; bold indicates the best performance value in each column.}",
        "\\label{tab:common-window}",
        "\\scriptsize",
        "\\setlength{\\tabcolsep}{2.5pt}",
        "\\resizebox{0.95\\textwidth}{!}{",
        "\\begin{tabular}{llrrrrrrrr}",
        "\\toprule",
        "\\textbf{Type} & \\textbf{Ranker} & \\textbf{LS CAGR} & \\textbf{LS Shp} & \\textbf{LS Sort} & \\textbf{MDD} & \\textbf{IC} & \\textbf{IC Mo.} & \\textbf{Avg N} & \\textbf{Avg LS Turn} \\\\",
        "\\midrule",
    ]
    last_type = None
    for _, row in rows.iterrows():
        if last_type is not None and row["Panel"] != last_type:
            lines.append("\\midrule")
        last_type = row["Panel"]
        lines.append(
            f"{row['Panel']} & {row['Ranker']} & "
            f"{fmt_pct(row['LS_CAGR'], is_close(row['LS_CAGR'], best_values['LS_CAGR']))} & "
            f"{fmt_num(row['LS_Sharpe'], is_close(row['LS_Sharpe'], best_values['LS_Sharpe']))} & "
            f"{fmt_num(row['LS_Sortino'], is_close(row['LS_Sortino'], best_values['LS_Sortino']))} & "
            f"{fmt_pct(row['LS_MDD'], is_close(row['LS_MDD'], best_values['LS_MDD']))} & "
            f"{fmt_pct(row['Mean_IC'], is_close(row['Mean_IC'], best_values['Mean_IC']))} & {int(row['Months'])} & "
            f"{row['Avg_Names']:.1f} & {row['Avg_LS_Turnover']:.2f} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", "}", "\\end{table}", ""])
    write_lines(path, lines)


def render_cost_table(df: pd.DataFrame, path: Path, scale_for_appendix: bool = False) -> None:
    ordered = df.loc[df["CostBps"] == COST_BPS].copy()
    ordered["SortKey"] = ordered["Panel"].map({"Baseline": 0, "LLM": 1}).fillna(2)
    ordered = ordered.sort_values(["SortKey", "Vendor", "Order"])
    pivot = df.pivot_table(index="Ranker", columns="CostBps", values="LS_Sharpe")
    best_by_cost = {cost_bps: pivot[cost_bps].max() for cost_bps in COST_GRID}

    lines = [
        "\\begin{table}[H]",
        "\\centering",
        "\\caption{Common-window transaction-cost sensitivity for all expected-return rankers. The long--short portfolio subtracts one-way costs on both long and short turnover; bold indicates the best Sharpe at each cost level.}",
        "\\label{tab:cost-sensitivity}",
    ]
    if scale_for_appendix:
        lines.extend(["\\scriptsize", "\\setlength{\\tabcolsep}{2.5pt}", "\\resizebox{0.67\\columnwidth}{!}{"])
    else:
        lines.append("\\scriptsize")
    lines.extend(
        [
            "\\begin{tabular}{lrrrrr}",
            "\\toprule",
            "\\textbf{Ranker} & \\textbf{Avg LS Turn} & \\textbf{0 bps} & \\textbf{20 bps} & \\textbf{40 bps} & \\textbf{100 bps} \\\\",
            "\\midrule",
        ]
    )
    for _, ordered_row in ordered.iterrows():
        ranker = ordered_row["Ranker"]
        row = pivot.loc[ranker]
        lines.append(
            f"{ranker} & {ordered_row['Avg_LS_Turnover']:.2f} & "
            f"{fmt_num(row.get(0.0, np.nan), is_close(row.get(0.0, np.nan), best_by_cost[0.0]))} & "
            f"{fmt_num(row.get(20.0, np.nan), is_close(row.get(20.0, np.nan), best_by_cost[20.0]))} & "
            f"{fmt_num(row.get(40.0, np.nan), is_close(row.get(40.0, np.nan), best_by_cost[40.0]))} & "
            f"{fmt_num(row.get(100.0, np.nan), is_close(row.get(100.0, np.nan), best_by_cost[100.0]))} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    if scale_for_appendix:
        lines.append("}")
    lines.extend(["\\end{table}", ""])
    write_lines(path, lines)

