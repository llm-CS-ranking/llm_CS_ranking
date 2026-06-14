from __future__ import annotations

import argparse

import pandas as pd

from llm_ranking.paths import (
    APPENDIX_SECTION_DIR,
    APPENDIX_TABLE_DIR,
    COMMON_WINDOW_DIR,
    PAPER_TABLE_DIR,
    TABLE_DIR,
)
from llm_ranking.rendering.publish import write_text_to_many
from llm_ranking.rendering.tables.appendix_metrics import (
    render_extended_table,
    render_quintile_cagr,
    render_within_universe_corr,
)
from llm_ranking.rendering.tables.common_window import render_common_window_table, render_cost_table
from llm_ranking.rendering.tables.conservative import render_conservative_longshort
from llm_ranking.rendering.tables.factor_tables import (
    render_input_ablation_char_table,
    render_input_ablation_ff6_table,
)
from llm_ranking.rendering.tables.input_ablation import render_input_ablation_appendix_table


TABLE_GROUPS = [
    "table10",
    "common_window",
    "appendix_metrics",
    "factor_tables",
    "conservative",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render paper LaTeX tables from cached CSV artifacts.")
    parser.add_argument(
        "--only",
        choices=[*TABLE_GROUPS, "all"],
        default="all",
        help="Table family to render.",
    )
    return parser.parse_args()


def render_table10() -> None:
    df = pd.read_csv(TABLE_DIR / "input_ablation_appendix_results.csv")
    render_input_ablation_appendix_table(df, TABLE_DIR / "input_ablation_appendix_table.tex")
    render_input_ablation_appendix_table(df, APPENDIX_TABLE_DIR / "input_ablation_appendix_table.tex")


def render_common_window() -> None:
    common_df = pd.read_csv(COMMON_WINDOW_DIR / "common_window_results.csv")
    cost_df = pd.read_csv(COMMON_WINDOW_DIR / "transaction_cost_sensitivity.csv")
    render_common_window_table(common_df, COMMON_WINDOW_DIR / "common_window_longshort.tex")
    render_common_window_table(common_df, APPENDIX_TABLE_DIR / "common_window_longshort.tex")
    render_cost_table(cost_df, COMMON_WINDOW_DIR / "transaction_cost_sensitivity.tex")
    render_cost_table(cost_df, APPENDIX_TABLE_DIR / "transaction_cost_sensitivity.tex", scale_for_appendix=True)


def render_appendix_metrics() -> None:
    df = pd.read_csv(TABLE_DIR / "archive_appendix_gross_metrics.csv")
    corr = pd.read_csv(TABLE_DIR / "within_universe_correlation_current.csv")
    render_quintile_cagr(df, APPENDIX_SECTION_DIR / "02_quintile_cagr_breakdown.tex")
    render_extended_table(
        df,
        "Q1",
        APPENDIX_SECTION_DIR / "03_extended_q1_metrics.tex",
        "Extended Q1 Portfolio Metrics",
        "app:q1-extended",
        "Extended gross full-window Q1 portfolio metrics, before transaction costs. Bold indicates the best value in each metric column within each panel.",
        "Table~\\ref{tab:q1-extended} reports gross full-window performance metrics for the top-quintile (Q1) portfolio across all three prediction modes, including annualized volatility, Calmar ratio, and hit ratio.",
    )
    render_extended_table(
        df,
        "Q1_minus_Q5",
        APPENDIX_SECTION_DIR / "10_extended_longshort_metrics.tex",
        "Extended Long--Short Portfolio Metrics",
        "app:ls-extended",
        "Extended gross full-window Q1$-$Q5 long--short metrics, before transaction costs. Bold indicates the best value in each metric column within each panel.",
        "Table~\\ref{tab:ls-extended} reports gross full-window extended performance metrics for the long--short (Q1$-$Q5) portfolio, complementing the net 40\\,bps summary in Table~\\ref{tab:longshort}.",
    )
    render_within_universe_corr(corr, APPENDIX_SECTION_DIR / "11_within_universe_factor_mimicking.tex")


def render_factor_tables() -> None:
    ff6_df = pd.read_csv(TABLE_DIR / "input_ablation_ff6_loadings.csv")
    char_df = pd.read_csv(TABLE_DIR / "input_ablation_char_exposure.csv")
    render_input_ablation_ff6_table(ff6_df, TABLE_DIR / "input_ablation_ff6_loadings.tex")
    render_input_ablation_ff6_table(ff6_df, APPENDIX_TABLE_DIR / "input_ablation_ff6_loadings.tex")
    render_input_ablation_char_table(char_df, TABLE_DIR / "input_ablation_char_exposure.tex")
    render_input_ablation_char_table(char_df, APPENDIX_TABLE_DIR / "input_ablation_char_exposure.tex")


def render_conservative() -> None:
    longshort = pd.read_csv(TABLE_DIR / "conservative_filtered_llm_results.csv")
    tex = render_conservative_longshort(longshort)
    write_text_to_many(
        tex,
        [
            TABLE_DIR / "conservative_longshort.tex",
            PAPER_TABLE_DIR / "conservative_longshort.tex",
            APPENDIX_TABLE_DIR / "conservative_longshort.tex",
        ],
    )


def main() -> None:
    args = parse_args()
    selected = TABLE_GROUPS if args.only == "all" else [args.only]
    for group in selected:
        globals()[f"render_{group}"]()
        print(f"Rendered {group}")


if __name__ == "__main__":
    main()

