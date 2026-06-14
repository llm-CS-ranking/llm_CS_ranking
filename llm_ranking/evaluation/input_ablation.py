from __future__ import annotations

import pandas as pd

from llm_ranking.config import INPUT_SETS, MODE_LABELS, MODEL_CONFIG, MODES
from llm_ranking.evaluation.portfolio import evaluate_panel
from llm_ranking.io.market import load_market
from llm_ranking.io.panels import input_prefix, load_ablation_panel
from llm_ranking.paths import TABLE_DIR


def build_input_ablation_appendix_results() -> pd.DataFrame:
    market, _cid2ticker, _member = load_market()
    existing_both = pd.read_csv(TABLE_DIR / "all_paper_models_results.csv")
    rows = []

    for model, cfg in MODEL_CONFIG.items():
        for mode in MODES:
            row = {
                "Vendor": cfg["vendor"],
                "Model": cfg["display"],
                "ModelID": model,
                "Mode": mode,
                "Objective": MODE_LABELS[mode],
            }
            for input_set, input_label in INPUT_SETS.items():
                if input_set == "both":
                    prev = existing_both[
                        (existing_both["ModelID"] == model) & (existing_both["Mode"] == mode)
                    ]
                    metrics = prev.iloc[0].to_dict() if not prev.empty else None
                else:
                    panel = load_ablation_panel(input_set, mode, model)
                    metrics = None if panel is None else evaluate_panel(panel, market)

                prefix = input_prefix(input_label)
                row[f"{prefix}_LS_CAGR"] = None if metrics is None else metrics["LS_CAGR"]
                row[f"{prefix}_LS_Sharpe"] = None if metrics is None else metrics["LS_Sharpe"]
                row[f"{prefix}_Mean_IC"] = None if metrics is None else metrics["Mean_IC"]
            rows.append(row)

    out = pd.DataFrame(rows)
    out["ModeOrder"] = out["Mode"].map({mode: i for i, mode in enumerate(MODES)})
    out["ModelOrder"] = out["ModelID"].map({model: i for i, model in enumerate(MODEL_CONFIG)})
    return out.sort_values(["ModeOrder", "ModelOrder"]).drop(columns=["ModeOrder", "ModelOrder"])


def write_input_ablation_appendix_csv(df: pd.DataFrame) -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(TABLE_DIR / "input_ablation_appendix_results.csv", index=False)

