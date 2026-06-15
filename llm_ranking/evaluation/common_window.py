from __future__ import annotations

import pandas as pd

from llm_ranking.config import BASELINE_CONFIG, COST_BPS, COST_GRID, MODE_LABELS, MODEL_CONFIG, MODEL_CUTOFFS, MODES
from llm_ranking.evaluation.portfolio import evaluate_panel
from llm_ranking.io.market import load_market
from llm_ranking.io.panels import common_months_for_panels, first_full_month_after, load_llm_panels, load_pickle_panel, subset_panel
from llm_ranking.paths import COMMON_WINDOW_DIR, RESULT_DIR


def apply_conservative_boundaries(
    panels: dict[tuple[str, str], pd.DataFrame],
) -> dict[tuple[str, str], pd.DataFrame]:
    conservative = {}
    for (model_id, mode), panel in panels.items():
        start_month = first_full_month_after(MODEL_CUTOFFS[model_id])
        filtered = panel.loc[panel.index.astype(str) >= start_month].dropna(axis=0, how="all")
        if filtered.empty:
            raise RuntimeError(f"No conservative-window months for {model_id} / {mode}.")
        conservative[(model_id, mode)] = filtered
    return conservative


def build_common_window_results() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    market, _cid2ticker, _member = load_market()
    llm_panels = apply_conservative_boundaries(load_llm_panels())
    common_months = common_months_for_panels(llm_panels)

    metadata = pd.DataFrame(
        [
            {
                "CommonStart": common_months[0],
                "CommonEnd": common_months[-1],
                "Months": len(common_months),
                "Definition": "Intersection of all 11 LLM model-mode rank-panel month indexes after applying model-specific conservative release/update boundaries.",
            }
        ]
    )

    rows = []
    for order, (model_id, cfg) in enumerate(MODEL_CONFIG.items()):
        for mode in MODES:
            panel = subset_panel(llm_panels[(model_id, mode)], common_months)
            row = evaluate_panel(panel, market, cost_bps=COST_BPS, include_turnover=True)
            row.update(
                {
                    "Panel": "LLM",
                    "Ranker": cfg["display"],
                    "ModelID": model_id,
                    "Vendor": cfg["vendor"],
                    "Mode": mode,
                    "ModeLabel": MODE_LABELS[mode],
                    "CostBps": COST_BPS,
                    "Order": order,
                }
            )
            rows.append(row)

    for order, (filename, display) in enumerate(BASELINE_CONFIG.items()):
        panel = subset_panel(load_pickle_panel(RESULT_DIR / filename), common_months)
        row = evaluate_panel(panel, market, cost_bps=COST_BPS, include_turnover=True)
        row.update(
            {
                "Panel": "Baseline",
                "Ranker": display,
                "ModelID": filename.removesuffix("_rank_panel.pkl"),
                "Vendor": "Non-LLM",
                "Mode": "baseline",
                "ModeLabel": "Baseline",
                "CostBps": COST_BPS,
                "Order": order,
            }
        )
        rows.append(row)

    common_df = pd.DataFrame(rows)

    cost_panels = []
    for order, (filename, display) in enumerate(BASELINE_CONFIG.items()):
        cost_panels.append(
            {
                "Panel": "Baseline",
                "Ranker": display,
                "Vendor": "Non-LLM",
                "Mode": "baseline",
                "Order": order,
                "PanelData": subset_panel(load_pickle_panel(RESULT_DIR / filename), common_months),
            }
        )
    for order, (model_id, cfg) in enumerate(MODEL_CONFIG.items()):
        cost_panels.append(
            {
                "Panel": "LLM",
                "Ranker": cfg["display"],
                "Vendor": cfg["vendor"],
                "Mode": "return",
                "Order": order,
                "PanelData": subset_panel(llm_panels[(model_id, "return")], common_months),
            }
        )

    cost_rows = []
    for item in cost_panels:
        panel = item.pop("PanelData")
        for cost_bps in COST_GRID:
            row = evaluate_panel(panel, market, cost_bps=cost_bps, include_turnover=True)
            row.update({**item, "CostBps": cost_bps})
            cost_rows.append(row)
    cost_df = pd.DataFrame(cost_rows)
    return metadata, common_df, cost_df


def write_common_window_csvs(
    metadata: pd.DataFrame,
    common_df: pd.DataFrame,
    cost_df: pd.DataFrame,
) -> None:
    COMMON_WINDOW_DIR.mkdir(parents=True, exist_ok=True)
    metadata.to_csv(COMMON_WINDOW_DIR / "common_window_metadata.csv", index=False)
    common_df.to_csv(COMMON_WINDOW_DIR / "common_window_results.csv", index=False)
    cost_df.to_csv(COMMON_WINDOW_DIR / "transaction_cost_sensitivity.csv", index=False)

