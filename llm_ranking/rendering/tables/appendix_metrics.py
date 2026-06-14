from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from llm_ranking.config import CHAR_SPECS, MODE_LABELS, MODES
from llm_ranking.rendering.latex import fmt_num, fmt_pct, is_close, short_model, sig_stars, write_lines


def _append_vendor_rows(lines: list[str], rows: pd.DataFrame, row_writer, sep_range: str = "2-9") -> None:
    last_vendor = None
    for _, row in rows.iterrows():
        if last_vendor is not None and row["Vendor"] != last_vendor:
            lines.append(f"\\cmidrule(l){{{sep_range}}}")
        last_vendor = row["Vendor"]
        lines.append(row_writer(row))


def render_quintile_cagr(df: pd.DataFrame, path: Path) -> None:
    lines = [
        "\\section{Quintile Portfolio Performance}",
        "\\label{app:quintile-cagr}",
        "",
        "Table~\\ref{tab:quintile-cagr} reports the gross CAGR of each quintile portfolio before transaction costs.",
        "Monotonically decreasing values from Q1 to Q5 indicate effective ranking; anti-monotonic patterns (Q5 $>$ Q1) indicate counter-productive signals.",
        "",
        "\\begin{table}[H]",
        "\\centering",
        "\\caption{Gross full-window quintile CAGR (\\%) by model and prediction mode, before transaction costs. Bold marks the best value in each column within each panel.}",
        "\\label{tab:quintile-cagr}",
        "\\scriptsize",
        "\\setlength{\\tabcolsep}{3pt}",
        "\\renewcommand{\\arraystretch}{0.80}",
        "\\begin{adjustbox}{width=\\textwidth,totalheight=0.78\\textheight,keepaspectratio}",
        "\\begin{tabular}{ll rrrrrr}",
        "\\toprule",
        "\\textbf{Mode} & \\textbf{Model} & \\textbf{Q1} & \\textbf{Q2} & \\textbf{Q3} & \\textbf{Q4} & \\textbf{Q5} & \\textbf{Q1$-$Q5} \\\\",
    ]
    cagr_cols = ["Q1_CAGR", "Q2_CAGR", "Q3_CAGR", "Q4_CAGR", "Q5_CAGR", "Q1_minus_Q5_CAGR"]
    for mode in MODES:
        rows = df[df["Mode"] == mode].copy()
        best_values = {col: rows[col].max() for col in cagr_cols}
        lines.extend(
            [
                "\\midrule",
                f"\\multicolumn{{8}}{{l}}{{\\textit{{Panel {chr(65 + MODES.index(mode))}: {MODE_LABELS[mode]}}}}} \\\\",
                "\\midrule",
            ]
        )
        _append_vendor_rows(
            lines,
            rows,
            lambda row: (
                f"& {short_model(row['Model'])} & "
                f"{fmt_pct(row['Q1_CAGR'], is_close(row['Q1_CAGR'], best_values['Q1_CAGR']), latex_sign=True)} & "
                f"{fmt_pct(row['Q2_CAGR'], is_close(row['Q2_CAGR'], best_values['Q2_CAGR']), latex_sign=True)} & "
                f"{fmt_pct(row['Q3_CAGR'], is_close(row['Q3_CAGR'], best_values['Q3_CAGR']), latex_sign=True)} & "
                f"{fmt_pct(row['Q4_CAGR'], is_close(row['Q4_CAGR'], best_values['Q4_CAGR']), latex_sign=True)} & "
                f"{fmt_pct(row['Q5_CAGR'], is_close(row['Q5_CAGR'], best_values['Q5_CAGR']), latex_sign=True)} & "
                f"{fmt_pct(row['Q1_minus_Q5_CAGR'], is_close(row['Q1_minus_Q5_CAGR'], best_values['Q1_minus_Q5_CAGR']), latex_sign=True)} \\\\"
            ),
            sep_range="2-8",
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{adjustbox}", "\\end{table}", "", "% " + "-" * 61, ""])
    write_lines(path, lines)


def render_extended_table(
    df: pd.DataFrame,
    portfolio: str,
    path: Path,
    section: str,
    label: str,
    caption: str,
    intro: str,
) -> None:
    lines = [
        f"\\section{{{section}}}",
        f"\\label{{{label}}}",
        "",
        intro,
        "",
        "\\begin{table}[H]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{tab:{'q1-extended' if portfolio == 'Q1' else 'ls-extended'}}}",
        "\\scriptsize",
        "\\setlength{\\tabcolsep}{3pt}",
        "\\renewcommand{\\arraystretch}{0.80}",
        "\\begin{adjustbox}{width=\\textwidth,totalheight=0.78\\textheight,keepaspectratio}",
        "\\begin{tabular}{ll rrrrrrr}",
        "\\toprule",
        "\\textbf{Mode} & \\textbf{Model} & \\textbf{CAGR} & \\textbf{Vol} & \\textbf{Sharpe} & \\textbf{Sortino} & \\textbf{MDD} & \\textbf{Calmar} & \\textbf{Hit\\%} \\\\",
    ]
    metric_specs = [
        (f"{portfolio}_CAGR", lambda value, bold=False: fmt_pct(value, bold, latex_sign=True), "max"),
        (f"{portfolio}_Vol", lambda value, bold=False: fmt_pct(value, bold, latex_sign=True), "min"),
        (f"{portfolio}_Sharpe", lambda value, bold=False: fmt_num(value, bold, latex_sign=True), "max"),
        (f"{portfolio}_Sortino", lambda value, bold=False: fmt_num(value, bold, latex_sign=True), "max"),
        (f"{portfolio}_MDD", lambda value, bold=False: fmt_pct(value, bold, latex_sign=True), "max"),
        (f"{portfolio}_Calmar", lambda value, bold=False: fmt_num(value, bold, decimals=3, latex_sign=True), "max"),
        (f"{portfolio}_Hit", lambda value, bold=False: fmt_pct(value, bold, latex_sign=True), "max"),
    ]

    for mode in MODES:
        rows = df[df["Mode"] == mode].copy()
        best_values = {
            col: rows[col].min() if direction == "min" else rows[col].max()
            for col, _formatter, direction in metric_specs
        }
        lines.extend(
            [
                "\\midrule",
                f"\\multicolumn{{9}}{{l}}{{\\textit{{Panel {chr(65 + MODES.index(mode))}: {MODE_LABELS[mode]}}}}} \\\\",
                "\\midrule",
            ]
        )
        _append_vendor_rows(
            lines,
            rows,
            lambda row: (
                f"& {short_model(row['Model'])} & "
                + " & ".join(
                    formatter(row[col], is_close(row[col], best_values[col]))
                    for col, formatter, _direction in metric_specs
                )
                + " \\\\"
            ),
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{adjustbox}", "\\end{table}", "", "% " + "-" * 61, ""])
    write_lines(path, lines)


def _fmt_corr(value: float, p: float = np.nan) -> str:
    if pd.isna(value):
        return "N/A"
    sign = "$+$" if value >= 0 else "$-$"
    formatted = f"{sign}{abs(value):.2f}"
    stars = sig_stars(p)
    if stars:
        formatted = f"{formatted}$^{{{stars}}}$"
    return f"\\multicolumn{{1}}{{l}}{{{formatted}}}"


def render_within_universe_corr(corr: pd.DataFrame, path: Path) -> None:
    lines = [
        "\\section{Within-Universe Factor-Mimicking Portfolios}",
        "\\label{app:within-univ}",
        "",
        "Because the 30-stock universe is concentrated in mega-cap technology, global Fama--French factors may not capture style tilts that occur \\emph{within} the universe.",
        "For each characteristic $c$, we construct a monthly within-universe long--short portfolio by sorting the 30 stocks on $c$, going long the top 6 (on the traditional ``long'' side of that characteristic) and short the bottom 6, rebalancing with a 2-day lag and no transaction cost.",
        "Table~\\ref{tab:within-univ-corr} reports the daily-return Pearson correlation between each LLM's Q1$-$Q5 portfolio and these factor-mimicking portfolios (expected-return mode).",
        "",
        "\\begin{table}[H]",
        "\\centering",
        "\\caption{Correlation between LLM Q1$-$Q5 returns and within-universe factor-mimicking portfolios. Positive entries align with the factor's traditional long side; stars test zero correlation with Newey--West HAC SE: $^{*}p<0.10$, $^{**}p<0.05$, $^{***}p<0.01$.}",
        "\\label{tab:within-univ-corr}",
        "\\scriptsize",
        "\\setlength{\\tabcolsep}{3pt}",
        "\\resizebox{\\textwidth}{!}{",
        "\\begin{tabular}{l lllllll}",
        "\\toprule",
        "\\textbf{Model} & Mom & Rev & LowVol & E/P & ROE & LowInv & SmallSize \\\\",
        "\\midrule",
    ]
    labels = [label for label, _ascending_good in CHAR_SPECS.values()]
    last_vendor = None
    for _, row in corr.iterrows():
        if last_vendor is not None and row["Vendor"] != last_vendor:
            lines.append("\\midrule")
        last_vendor = row["Vendor"]
        vals = " & ".join(_fmt_corr(row[col], row[f"{col}_p"]) for col in labels)
        lines.append(f"{short_model(row['Model'])} & {vals} \\\\")
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "}",
            "\\end{table}",
            "",
            "The correlation structure is consistent with the FF6 loadings in Table~\\ref{tab:ff6} and the style fingerprint in Figure~\\ref{fig:style}: the strongest representative models---o3, Gemini 2.5 Pro, and Claude Opus 4---all show positive momentum correlation and negative low-volatility/value correlations, confirming a high-momentum, high-volatility growth tilt within the universe.",
            "GPT-4o mini remains the main counterexample, with negative momentum and positive reversal/low-volatility correlations, consistent with its anti-monotonic quintile pattern during the 2024--2025 AI rally.",
            "",
            "Figure~\\ref{fig:loadings} visualizes the FF6 factor loadings as a heatmap, offering a compact view of the cross-vendor style differences.",
            "",
            "\\begin{figure}[H]",
            "  \\centering",
            "  \\includegraphics[width=\\columnwidth]{figures/fig9_factor_loadings.pdf}",
            "  \\caption{Conservative-window FF6 factor loadings of expected-return Q1$-$Q5 portfolios. Red indicates positive loading and blue indicates negative loading.}",
            "  \\Description{Heatmap of Fama-French six-factor loadings for current expected-return long-short portfolios, with red for positive loadings and blue for negative loadings.}",
            "  \\label{fig:loadings}",
            "\\end{figure}",
            "",
            "The heatmap closes the appendix by linking the table-level style correlations back to the FF6 spanning tests.",
            "Across the stronger expected-return rankers, positive market and momentum exposure coexists with negative value/profitability/investment loadings, consistent with the growth--momentum implicit-prior interpretation developed in the main text.",
        ]
    )
    write_lines(path, lines)

