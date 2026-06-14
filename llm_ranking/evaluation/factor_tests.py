from __future__ import annotations

import numpy as np
import pandas as pd

from factor_utils import build_characteristic_panel, quintile_char_exposure
from llm_ranking.config import CHAR_SPECS, INPUT_SETS, MODEL_CONFIG, REPRESENTATIVE_ABLATION_MODELS
from llm_ranking.io.market import load_fundamentals, load_market
from llm_ranking.io.panels import load_ablation_panel
from llm_ranking.paths import DATA_DIR, TABLE_DIR
from llm_ranking.rendering.latex import normal_pvalue_from_t
from utils import backtest_quintiles_with_universe_ew_capw

MODE = "return"
FF6 = ["Mkt_RF", "SMB", "HML", "RMW", "CMA", "MOM"]


def load_cached_ff_factors() -> pd.DataFrame:
    path = DATA_DIR / "ff_factors_daily.csv"
    df = pd.read_csv(path, parse_dates=["DATE"]).set_index("DATE").sort_index()
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df


def hac_mean_test(series: pd.Series, nw_lags: int = 5) -> tuple[float, float]:
    y = series.dropna().to_numpy(dtype=float)
    n = len(y)
    if n == 0:
        return np.nan, np.nan
    resid = y - y.mean()
    gamma0 = float(np.dot(resid, resid) / n)
    lrv = gamma0
    max_lag = min(nw_lags, n - 1)
    for lag in range(1, max_lag + 1):
        weight = 1.0 - lag / (max_lag + 1.0)
        gamma = float(np.dot(resid[lag:], resid[:-lag]) / n)
        lrv += 2.0 * weight * gamma
    if lrv <= 0:
        return np.nan, np.nan
    se_mean = np.sqrt(lrv / n)
    t = float(y.mean() / se_mean)
    return t, normal_pvalue_from_t(t)


def hac_ols_ff6(port_daily: pd.Series, factors: pd.DataFrame, nw_lags: int = 5) -> dict[str, float]:
    y = port_daily.dropna().copy()
    y.index = pd.to_datetime(y.index).tz_localize(None)

    df = factors.reindex(y.index).dropna(subset=FF6)
    y = y.reindex(df.index).to_numpy(dtype=float)
    x = df[FF6].to_numpy(dtype=float)
    x = np.column_stack([np.ones(len(x)), x])

    xtx_inv = np.linalg.inv(x.T @ x)
    beta = xtx_inv @ x.T @ y
    resid = y - x @ beta
    n, k = x.shape

    s = x.T @ np.diag(resid**2) @ x
    for lag in range(1, nw_lags + 1):
        weight = 1.0 - lag / (nw_lags + 1.0)
        gamma = np.zeros((k, k))
        for t in range(lag, n):
            gamma += resid[t] * resid[t - lag] * np.outer(x[t], x[t - lag])
        s += weight * (gamma + gamma.T)

    cov = xtx_inv @ s @ xtx_inv
    se = np.sqrt(np.diag(cov))
    tvals = beta / se

    y_hat = x @ beta
    ssr = float(((y - y_hat) ** 2).sum())
    tss = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ssr / tss
    r2_adj = 1.0 - (1.0 - r2) * (n - 1) / (n - k)

    alpha_daily = float(beta[0])
    out = {
        "alpha_ann": (1.0 + alpha_daily) ** 252 - 1.0,
        "alpha_t": float(tvals[0]),
        "alpha_p": normal_pvalue_from_t(float(tvals[0])),
        "r2_adj": r2_adj,
        "nobs": n,
        "beta_Mkt_RF": float(beta[1]),
        "beta_SMB": float(beta[2]),
        "beta_HML": float(beta[3]),
        "beta_RMW": float(beta[4]),
        "beta_CMA": float(beta[5]),
        "beta_MOM": float(beta[6]),
    }
    for i, name in enumerate(FF6, start=1):
        out[f"p_{name}"] = normal_pvalue_from_t(float(tvals[i]))
    return out


def build_input_ablation_factor_tables() -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    market, cid2ticker, _member = load_market()
    fundamentals = load_fundamentals()
    factors = load_cached_ff_factors()
    ff6_rows = []
    char_rows = []
    missing = []

    for input_set, input_label in INPUT_SETS.items():
        for model, cfg in MODEL_CONFIG.items():
            if model not in REPRESENTATIVE_ABLATION_MODELS:
                continue
            panel = load_ablation_panel(input_set, MODE, model)
            if panel is None:
                missing.append(f"{input_set}:{MODE}:{model}")
                continue

            months = list(panel.index.astype(str))
            universe = list(panel.columns)
            char_panels = build_characteristic_panel(
                mkt=market,
                fundamentals_long=fundamentals,
                companyid2ticker=cid2ticker,
                months=months,
                universe=universe,
            )
            daily = backtest_quintiles_with_universe_ew_capw(
                rank_panel=panel,
                market_df=market,
                price_col="DIV_ADJ_CLOSE",
                cap_col="MKTCAP",
                n_quantiles=5,
                lag_days=2,
                cost_bps=0.0,
                add_long_short=True,
            )
            ff6 = hac_ols_ff6(daily["Q1_minus_Q5"], factors, nw_lags=5)
            ff6_rows.append(
                {
                    "InputSet": input_set,
                    "InputLabel": input_label,
                    "ModelID": model,
                    "Model": cfg["display"],
                    "Vendor": cfg["vendor"],
                    "alpha_ann_pct": ff6["alpha_ann"] * 100,
                    "alpha_t": ff6["alpha_t"],
                    "alpha_p": ff6["alpha_p"],
                    "beta_Mkt_RF": ff6["beta_Mkt_RF"],
                    "beta_SMB": ff6["beta_SMB"],
                    "beta_HML": ff6["beta_HML"],
                    "beta_RMW": ff6["beta_RMW"],
                    "beta_CMA": ff6["beta_CMA"],
                    "beta_MOM": ff6["beta_MOM"],
                    "p_Mkt_RF": ff6["p_Mkt_RF"],
                    "p_SMB": ff6["p_SMB"],
                    "p_HML": ff6["p_HML"],
                    "p_RMW": ff6["p_RMW"],
                    "p_CMA": ff6["p_CMA"],
                    "p_MOM": ff6["p_MOM"],
                    "r2_adj": ff6["r2_adj"],
                    "nobs": ff6["nobs"],
                }
            )

            for char_name, (_label, ascending_good) in CHAR_SPECS.items():
                if char_name == "size":
                    continue
                exposure = quintile_char_exposure(
                    panel,
                    char_panels[char_name],
                    char_ascending_good=ascending_good,
                )
                spread = exposure["Q1_minus_Q5"].dropna()
                if spread.empty:
                    continue
                char_t, char_p = hac_mean_test(spread, nw_lags=5)
                char_rows.append(
                    {
                        "InputSet": input_set,
                        "InputLabel": input_label,
                        "ModelID": model,
                        "Model": cfg["display"],
                        "characteristic": char_name,
                        "Q1_minus_Q5": float(spread.mean()),
                        "Q1_minus_Q5_t": char_t,
                        "Q1_minus_Q5_p": char_p,
                        "n_months": int(len(spread)),
                    }
                )
    return pd.DataFrame(ff6_rows), pd.DataFrame(char_rows), missing


def write_input_ablation_factor_csvs(ff6_df: pd.DataFrame, char_df: pd.DataFrame) -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    ff6_df.to_csv(TABLE_DIR / "input_ablation_ff6_loadings.csv", index=False)
    char_df.to_csv(TABLE_DIR / "input_ablation_char_exposure.csv", index=False)

