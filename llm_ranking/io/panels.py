from __future__ import annotations

import pickle
from pathlib import Path

import pandas as pd

from llm_ranking.config import INPUT_SETS, MODEL_CONFIG, MODEL_CUTOFFS, MODES
from llm_ranking.paths import ABLATION_DIR, RESULT_DIR


def load_pickle_panel(path: Path) -> pd.DataFrame:
    with path.open("rb") as f:
        panel = pickle.load(f)
    panel.index = panel.index.astype(str)
    panel.columns = panel.columns.astype(str)
    return panel


def llm_panel_path(mode: str, model_id: str) -> Path:
    return RESULT_DIR / f"{mode}_{model_id}_result.pkl"


def ablation_panel_path(input_set: str, mode: str, model_id: str) -> Path:
    if input_set == "both":
        return llm_panel_path(mode, model_id)
    return ABLATION_DIR / f"{input_set}_{mode}_{model_id}_result.pkl"


def load_panel(mode: str, model_id: str) -> pd.DataFrame:
    return load_pickle_panel(llm_panel_path(mode, model_id))


def load_ablation_panel(input_set: str, mode: str, model_id: str) -> pd.DataFrame | None:
    path = ablation_panel_path(input_set, mode, model_id)
    if not path.exists():
        return None
    return load_pickle_panel(path)


def sort_months(months: set[str] | list[str]) -> list[str]:
    return [str(p) for p in sorted(pd.PeriodIndex(list(months), freq="M"))]


def first_full_month_after(cutoff: str) -> str:
    return str(pd.Period(pd.to_datetime(cutoff), freq="M") + 1)


def subset_panel(panel: pd.DataFrame, months: list[str]) -> pd.DataFrame:
    return panel.reindex(months).dropna(axis=0, how="all")


def load_llm_panels(
    model_config: dict[str, dict[str, str]] | None = None,
    modes: list[str] | None = None,
    require_all: bool = True,
) -> dict[tuple[str, str], pd.DataFrame]:
    model_config = MODEL_CONFIG if model_config is None else model_config
    modes = MODES if modes is None else modes
    panels = {}
    missing = []
    for model_id in model_config:
        for mode in modes:
            path = llm_panel_path(mode, model_id)
            if not path.exists():
                missing.append(str(path.relative_to(RESULT_DIR.parent)))
                continue
            panels[(model_id, mode)] = load_pickle_panel(path)
    if require_all and missing:
        raise FileNotFoundError("Missing rank panels:\n" + "\n".join(missing))
    return panels


def common_months_for_panels(panels: dict[tuple[str, str], pd.DataFrame]) -> list[str]:
    month_sets = [set(panel.index.astype(str)) for panel in panels.values()]
    common = set.intersection(*month_sets)
    if not common:
        raise RuntimeError("No common months across rank panels.")
    return sort_months(common)


def load_cutoff_filtered_panels() -> dict[tuple[str, str], pd.DataFrame]:
    panels = {}
    for model_id in MODEL_CONFIG:
        start_month = first_full_month_after(MODEL_CUTOFFS[model_id])
        for mode in MODES:
            path = llm_panel_path(mode, model_id)
            if not path.exists():
                continue
            panel = load_pickle_panel(path)
            panel = panel.loc[panel.index >= start_month].dropna(axis=0, how="all")
            if not panel.empty:
                panels[(mode, model_id)] = panel
    return panels


def input_prefix(input_label: str) -> str:
    return input_label.replace(" ", "_").replace(".", "")


def input_prefixes() -> list[str]:
    return [input_prefix(label) for label in INPUT_SETS.values()]

