from __future__ import annotations

import numpy as np
import pandas as pd

from factor_utils import build_characteristic_panel, within_universe_factor_ls
from llm_ranking.config import CHAR_SPECS, MODE_LABELS, MODEL_CONFIG, MODES
from llm_ranking.evaluation.portfolio import gross_portfolio_metrics
from llm_ranking.io.market import load_fundamentals, load_market
from llm_ranking.io.panels import load_panel
from llm_ranking.paths import TABLE_DIR
from llm_ranking.rendering.latex import normal_pvalue_from_t
from utils import backtest_quintiles_with_universe_ew_capw


def hac_corr_pvalue(aligned: pd.DataFrame, nw_lags: int = 5) -> float:
    x = aligned["factor"].to_numpy(dtype=float)
    y = aligned["llm"].to_numpy(dtype=float)
    if len(x) <= 3 or np.nanstd(x) == 0 or np.nanstd(y) == 0:
        return np.nan

    x = (x - x.mean()) / x.std(ddof=0)
    y = (y - y.mean()) / y.std(ddof=0)
    design = np.column_stack([np.ones(len(x)), x])
    xtx_inv = np.linalg.pinv(design.T @ design)
    beta = xtx_inv @ design.T @ y
    resid = y - design @ beta

    n, k = design.shape
    s = design.T @ np.diag(resid**2) @ design
    max_lag = min(nw_lags, n - 1)
    for lag in range(1, max_lag + 1):
        weight = 1.0 - lag / (max_lag + 1.0)
        gamma = np.zeros((k, k))
        for t in range(lag, n):
            gamma += resid[t] * resid[t - lag] * np.outer(design[t], design[t - lag])
        s += weight * (gamma + gamma.T)
    cov = xtx_inv @ s @ xtx_inv
    se = np.sqrt(np.diag(cov))
    if se[1] <= 0 or not np.isfinite(se[1]):
        return np.nan
    return normal_pvalue_from_t(float(beta[1] / se[1]))


def collect_gross_metrics() -> tuple[pd.DataFrame, dict[tuple[str, str], pd.DataFrame]]:
    market, _cid2ticker, _member = load_market()
    rows = []
    panels = {}

    for model_id, cfg in MODEL_CONFIG.items():
        for mode in MODES:
            panel = load_panel(mode, model_id)
            panels[(mode, model_id)] = panel
            metrics = gross_portfolio_metrics(panel, market)
            row = {
                "ModelID": model_id,
                "Model": cfg["display"],
                "Vendor": cfg["vendor"],
                "Mode": mode,
                "ModeLabel": MODE_LABELS[mode],
            }
            for col in ["Q1", "Q2", "Q3", "Q4", "Q5", "Q1_minus_Q5"]:
                row[f"{col}_CAGR"] = float(metrics.loc["CAGR", col])
                row[f"{col}_Vol"] = float(metrics.loc["Ann.Vol", col])
                row[f"{col}_Sharpe"] = float(metrics.loc["Sharpe", col])
                row[f"{col}_Sortino"] = float(metrics.loc["Sortino", col])
                row[f"{col}_MDD"] = float(metrics.loc["MDD", col])
                row[f"{col}_Calmar"] = float(metrics.loc["Calmar", col])
                row[f"{col}_Hit"] = float(metrics.loc["HitRatio", col])
            rows.append(row)

    df = pd.DataFrame(rows)
    return df, panels


def write_gross_metrics_csv(df: pd.DataFrame) -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(TABLE_DIR / "archive_appendix_gross_metrics.csv", index=False)


def build_within_universe_corr(panels: dict[tuple[str, str], pd.DataFrame]) -> pd.DataFrame:
    market, cid2ticker, _member = load_market()
    fundamentals = load_fundamentals()
    rows = []

    for model_id, cfg in MODEL_CONFIG.items():
        panel = panels[("return", model_id)]
        daily = backtest_quintiles_with_universe_ew_capw(
            rank_panel=panel,
            market_df=market,
            price_col="DIV_ADJ_CLOSE",
            cap_col="MKTCAP",
            n_quantiles=5,
            lag_days=2,
            cost_bps=0.0,
            add_long_short=True,
        )["Q1_minus_Q5"]
        char_panels = build_characteristic_panel(
            mkt=market,
            fundamentals_long=fundamentals,
            companyid2ticker=cid2ticker,
            months=list(panel.index.astype(str)),
            universe=list(panel.columns.astype(str)),
        )
        row = {"ModelID": model_id, "Model": cfg["display"], "Vendor": cfg["vendor"]}
        for char_name, (label, ascending_good) in CHAR_SPECS.items():
            factor = within_universe_factor_ls(
                char_panels[char_name],
                market,
                n_quantiles=5,
                char_ascending_good=ascending_good,
                lag_days=2,
            )
            aligned = pd.concat([daily.rename("llm"), factor.rename("factor")], axis=1).dropna()
            row[label] = float(aligned["llm"].corr(aligned["factor"])) if len(aligned) > 2 else np.nan
            row[f"{label}_p"] = hac_corr_pvalue(aligned, nw_lags=5)
        rows.append(row)
    return pd.DataFrame(rows)


def write_within_universe_corr_csv(corr: pd.DataFrame) -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    corr.to_csv(TABLE_DIR / "within_universe_correlation_current.csv", index=False)

