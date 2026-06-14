from __future__ import annotations

import argparse

from llm_ranking.evaluation.appendix_metrics import (
    build_within_universe_corr,
    collect_gross_metrics,
    write_gross_metrics_csv,
    write_within_universe_corr_csv,
)
from llm_ranking.evaluation.common_window import build_common_window_results, write_common_window_csvs
from llm_ranking.evaluation.factor_tests import build_input_ablation_factor_tables, write_input_ablation_factor_csvs
from llm_ranking.evaluation.input_ablation import (
    build_input_ablation_appendix_results,
    write_input_ablation_appendix_csv,
)


RESULT_GROUPS = [
    "common_window",
    "input_ablation",
    "appendix_metrics",
    "factor_tests",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build result CSV artifacts from cached rank panels.")
    parser.add_argument(
        "--only",
        choices=[*RESULT_GROUPS, "all"],
        default="all",
        help="Result family to build.",
    )
    return parser.parse_args()


def build_common_window() -> None:
    metadata, common_df, cost_df = build_common_window_results()
    write_common_window_csvs(metadata, common_df, cost_df)


def build_input_ablation() -> None:
    df = build_input_ablation_appendix_results()
    write_input_ablation_appendix_csv(df)


def build_appendix_metrics() -> None:
    df, panels = collect_gross_metrics()
    write_gross_metrics_csv(df)
    corr = build_within_universe_corr(panels)
    write_within_universe_corr_csv(corr)


def build_factor_tests() -> None:
    ff6_df, char_df, missing = build_input_ablation_factor_tables()
    write_input_ablation_factor_csvs(ff6_df, char_df)
    if missing:
        print("Missing panels:")
        for item in missing:
            print(f"  {item}")


def main() -> None:
    args = parse_args()
    selected = RESULT_GROUPS if args.only == "all" else [args.only]
    for group in selected:
        globals()[f"build_{group}"]()
        print(f"Built {group}")


if __name__ == "__main__":
    main()

