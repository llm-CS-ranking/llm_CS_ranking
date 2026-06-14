from __future__ import annotations

import pandas as pd

from llm_ranking.io.market import load_market
from llm_ranking.io.panels import load_cutoff_filtered_panels
from utils import backtest_quintiles_with_universe_ew_capw, portfolio_metrics


def compute_cutoff_daily_metrics() -> tuple[dict[tuple[str, str], pd.DataFrame], dict[tuple[str, str], pd.DataFrame]]:
    market, _cid2ticker, _member = load_market()
    panels = load_cutoff_filtered_panels()
    all_daily = {}
    all_metrics = {}
    for key, panel in panels.items():
        daily = backtest_quintiles_with_universe_ew_capw(
            rank_panel=panel,
            market_df=market,
            price_col="DIV_ADJ_CLOSE",
            n_quantiles=5,
            lag_days=2,
            cost_bps=40.0,
            min_names_per_bucket=1,
            add_long_short=True,
        )
        all_daily[key] = daily
        all_metrics[key] = portfolio_metrics(daily)
    return all_daily, all_metrics

