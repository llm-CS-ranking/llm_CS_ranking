from __future__ import annotations

import numpy as np
import pandas as pd

from llm_ranking.config import COST_BPS
from llm_ranking.io.panels import sort_months
from utils import (
    backtest_quintiles_with_universe_ew_capw,
    portfolio_metrics,
    rank_ic_by_month,
    rank_ic_summary,
)


def turnover_summary(rank_panel: pd.DataFrame, n_quantiles: int = 5) -> dict[str, float]:
    tickers = list(rank_panel.columns.astype(str))
    prev_q1 = pd.Series(0.0, index=tickers)
    prev_q5 = pd.Series(0.0, index=tickers)
    rows = []

    for month in sort_months(list(rank_panel.index.astype(str))):
        ranks = rank_panel.loc[month].dropna().astype(float).sort_values(kind="mergesort")
        names = [str(t) for t in ranks.index]
        buckets = [list(arr) for arr in np.array_split(names, n_quantiles)]
        q1_names = buckets[0] if buckets else []
        q5_names = buckets[-1] if buckets else []

        q1 = pd.Series(0.0, index=tickers)
        q5 = pd.Series(0.0, index=tickers)
        if q1_names:
            q1.loc[q1_names] = 1.0 / len(q1_names)
        if q5_names:
            q5.loc[q5_names] = 1.0 / len(q5_names)

        q1_turnover = float((q1 - prev_q1).abs().sum())
        q5_turnover = float((q5 - prev_q5).abs().sum())
        rows.append(
            {
                "MONTH": month,
                "Q1_Turnover": q1_turnover,
                "Q5_Turnover": q5_turnover,
                "LS_Turnover": q1_turnover + q5_turnover,
            }
        )
        prev_q1 = q1
        prev_q5 = q5

    turn = pd.DataFrame(rows)
    post_initial = turn.iloc[1:] if len(turn) > 1 else turn
    return {
        "Avg_Q1_Turnover": float(post_initial["Q1_Turnover"].mean()),
        "Avg_Q5_Turnover": float(post_initial["Q5_Turnover"].mean()),
        "Avg_LS_Turnover": float(post_initial["LS_Turnover"].mean()),
        "Max_LS_Turnover": float(post_initial["LS_Turnover"].max()),
    }


def evaluate_panel(
    rank_panel: pd.DataFrame,
    market: pd.DataFrame,
    cost_bps: float = COST_BPS,
    include_turnover: bool = False,
) -> dict[str, float]:
    rank_counts = rank_panel.notna().sum(axis=1)
    daily = backtest_quintiles_with_universe_ew_capw(
        rank_panel=rank_panel,
        market_df=market,
        price_col="DIV_ADJ_CLOSE",
        cap_col="MKTCAP",
        n_quantiles=5,
        lag_days=2,
        cost_bps=cost_bps,
        add_long_short=True,
    )
    metrics = portfolio_metrics(daily)
    ic = rank_ic_summary(rank_ic_by_month(rank_panel, market, lag_months=0, method="spearman"))
    out = {
        "Q1_CAGR": float(metrics.loc["CAGR", "Q1"]),
        "Q1_Sharpe": float(metrics.loc["Sharpe", "Q1"]),
        "Q1_MDD": float(metrics.loc["MDD", "Q1"]),
        "LS_CAGR": float(metrics.loc["CAGR", "Q1_minus_Q5"]),
        "LS_Sharpe": float(metrics.loc["Sharpe", "Q1_minus_Q5"]),
        "LS_Sortino": float(metrics.loc["Sortino", "Q1_minus_Q5"]),
        "LS_MDD": float(metrics.loc["MDD", "Q1_minus_Q5"]),
        "Mean_IC": float(ic["mean_IC"]),
        "ICIR": float(ic["ICIR_annual"]),
        "Months": int(ic["n_months"]),
        "Avg_Names": float(rank_counts.mean()),
        "Min_Names": int(rank_counts.min()),
    }
    if include_turnover:
        out.update(turnover_summary(rank_panel))
    return out


def gross_portfolio_metrics(rank_panel: pd.DataFrame, market: pd.DataFrame) -> pd.DataFrame:
    daily = backtest_quintiles_with_universe_ew_capw(
        rank_panel=rank_panel,
        market_df=market,
        price_col="DIV_ADJ_CLOSE",
        cap_col="MKTCAP",
        n_quantiles=5,
        lag_days=2,
        cost_bps=0.0,
        add_long_short=True,
    )
    return portfolio_metrics(daily)

