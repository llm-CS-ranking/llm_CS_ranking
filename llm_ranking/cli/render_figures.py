from __future__ import annotations

import argparse

from llm_ranking.evaluation.performance import compute_cutoff_daily_metrics
from llm_ranking.rendering.figures.factor_heatmaps import (
    render_conservative_ff6_heatmap,
    render_input_ablation_ff6_heatmap,
)
from llm_ranking.rendering.figures.performance import render_longshort_heatmap, render_q1_cumulative_returns


FIGURE_GROUPS = ["fig6", "fig8", "fig9", "fig10", "performance"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render paper figures from cached artifacts.")
    parser.add_argument(
        "--only",
        choices=[*FIGURE_GROUPS, "all"],
        default="all",
        help="Figure or figure family to render.",
    )
    return parser.parse_args()


def render_fig6() -> None:
    all_daily, _all_metrics = compute_cutoff_daily_metrics()
    render_q1_cumulative_returns(all_daily)


def render_fig8() -> None:
    _all_daily, all_metrics = compute_cutoff_daily_metrics()
    render_longshort_heatmap(all_metrics)


def render_performance() -> None:
    all_daily, all_metrics = compute_cutoff_daily_metrics()
    render_q1_cumulative_returns(all_daily)
    render_longshort_heatmap(all_metrics)


def render_fig9() -> None:
    render_conservative_ff6_heatmap()


def render_fig10() -> None:
    render_input_ablation_ff6_heatmap()


def main() -> None:
    args = parse_args()
    selected = ["performance", "fig9", "fig10"] if args.only == "all" else [args.only]
    for group in selected:
        globals()[f"render_{group}"]()
        print(f"Rendered {group}")


if __name__ == "__main__":
    main()

