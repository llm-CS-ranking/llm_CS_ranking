from __future__ import annotations

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from llm_ranking.config import MODE_LABELS, MODES, MODEL_CONFIG
from llm_ranking.paths import APPENDIX_FIGURE_DIR, PAPER_FIGURE_DIR
from llm_ranking.rendering.latex import short_model

# Display from the first month where the newer model families begin to overlap.
# GPT-4o variants have much earlier conservative windows and would otherwise
# compress the 2025 cross-model comparison.
PLOT_START = pd.Timestamp("2025-05-01")
PLOT_START_LABEL = PLOT_START.strftime("%Y-%m")


def vendor_color_fn():
    vendor_colors = {
        "OpenAI": plt.cm.Blues,
        "Google": plt.cm.Greens,
        "Anthropic": plt.cm.Oranges,
    }
    vendor_counts = {}
    for model in MODEL_CONFIG:
        vendor_counts.setdefault(MODEL_CONFIG[model]["vendor"], 0)
        vendor_counts[MODEL_CONFIG[model]["vendor"]] += 1

    def _get_color(model: str):
        vendor = MODEL_CONFIG[model]["vendor"]
        vendor_models = [m for m in MODEL_CONFIG if MODEL_CONFIG[m]["vendor"] == vendor]
        idx = vendor_models.index(model)
        return vendor_colors[vendor](0.35 + 0.55 * idx / max(vendor_counts[vendor] - 1, 1))

    return _get_color


def _format_cumulative_axis(ax, ylabel: str | None = None, right: pd.Timestamp | None = None) -> None:
    if right is not None:
        ax.set_xlim(left=PLOT_START, right=right)
    else:
        ax.set_xlim(left=PLOT_START)
    ax.set_xlabel(f"Date (shown from {PLOT_START_LABEL})", fontsize=20)
    ax.tick_params(axis="both", labelsize=18)
    right = pd.Timestamp(right) if right is not None else pd.Timestamp(mdates.num2date(ax.get_xlim()[1])).tz_localize(None)
    right_month = right.to_period("M").to_timestamp()
    ticks = [PLOT_START]
    ticks.extend(pd.date_range(PLOT_START + pd.DateOffset(months=2), right, freq="3MS"))
    if right_month > ticks[-1]:
        ticks.append(right_month)
    ax.set_xticks(ticks)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_minor_locator(mdates.MonthLocator())
    ax.grid(alpha=0.3)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=20)
    for label in ax.get_xticklabels():
        label.set_rotation(30)
        label.set_horizontalalignment("right")


def render_q1_cumulative_returns(all_daily: dict[tuple[str, str], pd.DataFrame]) -> None:
    get_color = vendor_color_fn()

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for ax_idx, mode in enumerate(MODES):
        ax = axes[ax_idx]
        axis_right = PLOT_START
        for model in MODEL_CONFIG:
            daily = all_daily.get((mode, model))
            if daily is None or "Q1" not in daily.columns:
                continue
            r = daily["Q1"].copy()
            r.index = pd.to_datetime(r.index)
            equity = (1.0 + r.fillna(0)).cumprod()
            visible_equity = equity.loc[equity.index >= PLOT_START]
            if visible_equity.empty:
                continue
            axis_right = max(axis_right, visible_equity.index.max())
            ax.plot(
                visible_equity.index,
                visible_equity.values,
                linewidth=1.2,
                label=short_model(MODEL_CONFIG[model]["display"]),
                color=get_color(model),
            )

        first_key = next((key for key in all_daily if key[0] == mode), None)
        if first_key:
            bench = all_daily[first_key]["UNIV_EW"].copy()
            bench.index = pd.to_datetime(bench.index)
            bench_eq = (1.0 + bench.fillna(0)).cumprod()
            visible_bench = bench_eq.loc[bench_eq.index >= PLOT_START]
            if not visible_bench.empty:
                axis_right = max(axis_right, visible_bench.index.max())
            ax.plot(visible_bench.index, visible_bench.values, "--", color="gray", linewidth=1.5, label="UNIV_EW", alpha=0.7)

        ax.set_title(f"Q1 - {MODE_LABELS[mode]} Mode", fontweight="bold", fontsize=21)
        _format_cumulative_axis(ax, right=axis_right)

    axes[0].set_ylabel("Cumulative Wealth", fontsize=20)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=6, frameon=True, fontsize=16, bbox_to_anchor=(0.5, -0.12))
    fig.suptitle("Q1 (Top Quintile) Cumulative Returns (Appendix)", fontsize=27, fontweight="bold", y=1.01)
    fig.tight_layout()
    APPENDIX_FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(APPENDIX_FIGURE_DIR / "fig6_q1_cumulative_returns.pdf", bbox_inches="tight")
    plt.close(fig)


def render_longshort_cumulative_returns(all_daily: dict[tuple[str, str], pd.DataFrame]) -> None:
    get_color = vendor_color_fn()

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for ax_idx, mode in enumerate(MODES):
        ax = axes[ax_idx]
        axis_right = PLOT_START
        for model in MODEL_CONFIG:
            daily = all_daily.get((mode, model))
            if daily is None or "Q1_minus_Q5" not in daily.columns:
                continue
            r = daily["Q1_minus_Q5"].copy()
            r.index = pd.to_datetime(r.index)
            wealth = (1.0 + r.fillna(0)).cumprod()
            visible_wealth = wealth.loc[wealth.index >= PLOT_START]
            if visible_wealth.empty:
                continue
            axis_right = max(axis_right, visible_wealth.index.max())
            ax.plot(
                visible_wealth.index,
                visible_wealth.values,
                linewidth=1.2,
                label=short_model(MODEL_CONFIG[model]["display"]),
                color=get_color(model),
            )

        ax.axhline(1.0, linestyle="--", color="gray", linewidth=1.5, alpha=0.7, label="Zero spread wealth")
        ax.set_title(f"Q1$-$Q5 - {MODE_LABELS[mode]} Mode", fontweight="bold", fontsize=21)
        _format_cumulative_axis(ax, right=axis_right)

    axes[0].set_ylabel("Cumulative Wealth", fontsize=20)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=6, frameon=True, fontsize=16, bbox_to_anchor=(0.5, -0.12))
    fig.suptitle("Long$-$Short (Q1$-$Q5) Cumulative Wealth", fontsize=27, fontweight="bold", y=1.01)
    fig.tight_layout()
    for out_dir in [PAPER_FIGURE_DIR, APPENDIX_FIGURE_DIR]:
        out_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_dir / "fig4_ls_cumulative_returns.pdf", bbox_inches="tight")
    plt.close(fig)


def render_longshort_heatmap(all_metrics: dict[tuple[str, str], pd.DataFrame]) -> None:
    heat_metrics = {"CAGR": "{:.1%}", "Sharpe": "{:.2f}", "MDD": "{:.1%}"}
    display_names = [short_model(cfg["display"]) for cfg in MODEL_CONFIG.values()]
    model_to_display = {model: short_model(cfg["display"]) for model, cfg in MODEL_CONFIG.items()}

    fig, axes = plt.subplots(1, 3, figsize=(20, 8))
    vendor_breaks = []
    prev_vendor = None
    for pos, model in enumerate(MODEL_CONFIG):
        vendor = MODEL_CONFIG[model]["vendor"]
        if prev_vendor is not None and vendor != prev_vendor:
            vendor_breaks.append(pos)
        prev_vendor = vendor

    for ax_idx, (metric, fmt) in enumerate(heat_metrics.items()):
        ax = axes[ax_idx]
        data = pd.DataFrame(index=display_names, columns=[MODE_LABELS[mode] for mode in MODES], dtype=float)
        for model in MODEL_CONFIG:
            for mode in MODES:
                met = all_metrics.get((mode, model))
                if met is not None:
                    data.loc[model_to_display[model], MODE_LABELS[mode]] = met.loc[metric, "Q1_minus_Q5"]

        annot = data.map(lambda v: fmt.format(v) if pd.notna(v) else "-")
        sns.heatmap(
            data.astype(float),
            ax=ax,
            annot=annot,
            fmt="",
            cmap="RdYlGn",
            linewidths=0.5,
            annot_kws={"fontsize": 18},
            cbar_kws={"shrink": 0.8},
        )
        for y in vendor_breaks:
            ax.axhline(y, color="black", linewidth=1.4)
        ax.set_title(f"Q1$-$Q5 {metric}", fontweight="bold", fontsize=24)
        ax.set_xlabel("Prediction Mode", fontsize=20)
        ax.tick_params(axis="x", labelsize=18)
        if ax_idx == 0:
            ax.set_ylabel("Model", fontsize=20)
            ax.tick_params(axis="y", labelsize=18)
        else:
            ax.set_ylabel("")
            ax.set_yticklabels([])
            ax.tick_params(axis="y", left=False)
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=16)

    fig.suptitle("Long$-$Short (Q1$-$Q5) Performance Heatmap", fontsize=28, fontweight="bold", y=1.02)
    fig.tight_layout()
    APPENDIX_FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(APPENDIX_FIGURE_DIR / "fig8_longshort_performance_heatmap.pdf", bbox_inches="tight")
    plt.close(fig)

