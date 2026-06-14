from __future__ import annotations

import pandas as pd

from llm_ranking.cli.build_results import build_common_window
from llm_ranking.cli.render_tables import render_common_window
from llm_ranking.config import COST_BPS
from llm_ranking.evaluation.portfolio import evaluate_panel as _evaluate_panel
from llm_ranking.io.market import load_market


def evaluate_panel(rank_panel: pd.DataFrame, market: pd.DataFrame, cost_bps: float = COST_BPS) -> dict[str, float]:
    return _evaluate_panel(rank_panel, market, cost_bps=cost_bps, include_turnover=True)


def main() -> None:
    build_common_window()
    render_common_window()
    print("Wrote common-window CSV and LaTeX artifacts.")


if __name__ == "__main__":
    main()

