from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from llm_ranking.config import INPUT_SET_ORDER, REPRESENTATIVE_ABLATION_MODELS
from llm_ranking.rendering.latex import fmt_signed, sig_stars, write_lines


def fmt_beta(value: float, p: float) -> str:
    stars = sig_stars(p)
    formatted = fmt_signed(value, decimals=2)
    if stars:
        return f"{formatted}$^{{{stars}}}$"
    return formatted


def fmt_char_value(value: float, p: float) -> str:
    formatted = fmt_beta(value, p)
    if sig_stars(p):
        return f"\\multicolumn{{1}}{{l}}{{{formatted}}}"
    return formatted


def representative_ff6_rows(ff6_df: pd.DataFrame) -> pd.DataFrame:
    rep = ff6_df[ff6_df["ModelID"].isin(REPRESENTATIVE_ABLATION_MODELS)].copy()
    rep["ModelShort"] = rep["ModelID"].map(REPRESENTATIVE_ABLATION_MODELS)
    rep["InputOrder"] = rep["InputSet"].map({k: i for i, k in enumerate(INPUT_SET_ORDER)})
    rep["ModelOrder"] = rep["ModelID"].map({k: i for i, k in enumerate(REPRESENTATIVE_ABLATION_MODELS)})
    return rep.sort_values(["ModelOrder", "InputOrder"])


def render_input_ablation_ff6_table(ff6_df: pd.DataFrame, path: Path) -> None:
    rep = representative_ff6_rows(ff6_df)
    lines = [
        "\\begin{table}[H]",
        "\\centering",
        "\\caption{FF6 factor loadings of representative Q1$-$Q5 portfolios under input ablations. $\\alpha$ is annualized percent; stars use Newey--West HAC SE.}",
        "\\label{tab:ablation-ff6}",
        "\\scriptsize",
        "\\setlength{\\tabcolsep}{2.5pt}",
        "\\resizebox{\\textwidth}{!}{",
        "\\begin{tabular}{llrllllllr}",
        "\\toprule",
        "\\textbf{Model} & \\textbf{Input} & $\\alpha$ & $\\beta_{\\text{MKT}}$ & $\\beta_{\\text{SMB}}$ & $\\beta_{\\text{HML}}$ & $\\beta_{\\text{RMW}}$ & $\\beta_{\\text{CMA}}$ & $\\beta_{\\text{MOM}}$ & $R^2_{\\text{adj}}$ \\\\",
        "\\midrule",
    ]
    current_model = None
    for _, row in rep.iterrows():
        if current_model is not None and current_model != row["ModelShort"]:
            lines.append("\\midrule")
        current_model = row["ModelShort"]
        lines.append(
            f"{row['ModelShort']} & {row['InputLabel']} & "
            f"{row['alpha_ann_pct']:.1f} & "
            f"{fmt_beta(row['beta_Mkt_RF'], row['p_Mkt_RF'])} & "
            f"{fmt_beta(row['beta_SMB'], row['p_SMB'])} & "
            f"{fmt_beta(row['beta_HML'], row['p_HML'])} & "
            f"{fmt_beta(row['beta_RMW'], row['p_RMW'])} & "
            f"{fmt_beta(row['beta_CMA'], row['p_CMA'])} & "
            f"{fmt_beta(row['beta_MOM'], row['p_MOM'])} & "
            f"{row['r2_adj']:.2f} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", "}", "\\end{table}", ""])
    write_lines(path, lines)


def render_input_ablation_char_table(char_df: pd.DataFrame, path: Path) -> None:
    char_rep = char_df[char_df["ModelID"].isin(REPRESENTATIVE_ABLATION_MODELS)].copy()
    char_rep["ModelShort"] = char_rep["ModelID"].map(REPRESENTATIVE_ABLATION_MODELS)
    char_rep["InputOrder"] = char_rep["InputSet"].map({k: i for i, k in enumerate(INPUT_SET_ORDER)})
    char_rep["ModelOrder"] = char_rep["ModelID"].map({k: i for i, k in enumerate(REPRESENTATIVE_ABLATION_MODELS)})

    pivot = char_rep.pivot_table(
        index=["ModelShort", "InputLabel", "InputOrder", "ModelOrder"],
        columns="characteristic",
        values="Q1_minus_Q5",
        aggfunc="first",
    ).reset_index()
    p_pivot = char_rep.pivot_table(
        index=["ModelShort", "InputLabel", "InputOrder", "ModelOrder"],
        columns="characteristic",
        values="Q1_minus_Q5_p",
        aggfunc="first",
    ).reset_index()
    pivot = pivot.merge(
        p_pivot,
        on=["ModelShort", "InputLabel", "InputOrder", "ModelOrder"],
        suffixes=("", "_p"),
    )
    pivot = pivot.sort_values(["ModelOrder", "InputOrder"])

    lines = [
        "\\begin{table}[H]",
        "\\centering",
        "\\caption{Within-universe style fingerprint under input ablations. Values are average Q1$-$Q5 normalized characteristic ranks; stars test the monthly mean spread with Newey--West HAC SE: $^{*}p<0.10$, $^{**}p<0.05$, $^{***}p<0.01$.}",
        "\\label{tab:ablation-char}",
        "\\scriptsize",
        "\\setlength{\\tabcolsep}{3pt}",
        "\\resizebox{\\textwidth}{!}{",
        "\\begin{tabular}{ll llllll}",
        "\\toprule",
        "\\textbf{Model} & \\textbf{Input} & Mom & Rev & LowVol & E/P & ROE & LowInv \\\\",
        "\\midrule",
    ]
    current_model = None
    for _, row in pivot.iterrows():
        if current_model is not None and current_model != row["ModelShort"]:
            lines.append("\\midrule")
        current_model = row["ModelShort"]
        lines.append(
            f"{row['ModelShort']} & {row['InputLabel']} & "
            f"{fmt_char_value(row.get('mom_12_1', np.nan), row.get('mom_12_1_p', np.nan))} & "
            f"{fmt_char_value(row.get('rev_1m', np.nan), row.get('rev_1m_p', np.nan))} & "
            f"{fmt_char_value(row.get('vol_60d', np.nan), row.get('vol_60d_p', np.nan))} & "
            f"{fmt_char_value(row.get('ep_ratio', np.nan), row.get('ep_ratio_p', np.nan))} & "
            f"{fmt_char_value(row.get('roe', np.nan), row.get('roe_p', np.nan))} & "
            f"{fmt_char_value(row.get('asset_gr', np.nan), row.get('asset_gr_p', np.nan))} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", "}", "\\end{table}", ""])
    write_lines(path, lines)

