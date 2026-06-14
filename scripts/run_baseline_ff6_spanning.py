from __future__ import annotations

import sys
import math
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from run_non_llm_baselines import (
    DATA_DIR,
    build_characteristic_panel,
    build_factor_baselines,
    build_ml_baselines,
    common_rank_template,
    load_market,
    load_rank_panels,
)
from utils import backtest_quintiles_with_universe_ew_capw


TABLE_DIR = ROOT / "result" / "tables"
OUT_CSV = TABLE_DIR / "baseline_spanning_ls_ff6.csv"
FF6 = ["Mkt_RF", "SMB", "HML", "RMW", "CMA", "MOM"]


DISPLAY_NAMES = {
    "RandomForest": "Random Forest",
    "Ridge": "Ridge",
    "Mom12-1": "12--1 Momentum",
    "QualityROE": "ROE Quality",
    "ValueEP": "Value (E/P)",
    "Composite6": "Composite",
}

ORDER = ["RandomForest", "Ridge", "Mom12-1", "QualityROE", "ValueEP", "Composite6"]


def load_cached_ff_factors() -> pd.DataFrame:
    path = ROOT / "data" / "ff_factors_daily.csv"
    df = pd.read_csv(path, parse_dates=["DATE"]).set_index("DATE").sort_index()
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df


def normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


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

    s = x.T @ (np.diag(resid**2)) @ x
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
    alpha_t = float(tvals[0])
    return {
        "alpha_ann": (1.0 + alpha_daily) ** 252 - 1.0,
        "alpha_t": alpha_t,
        "alpha_p": 2.0 * (1.0 - normal_cdf(abs(alpha_t))),
        "r2_adj": r2_adj,
        "nobs": n,
        "beta_Mkt_RF": float(beta[1]),
        "beta_SMB": float(beta[2]),
        "beta_HML": float(beta[3]),
        "beta_RMW": float(beta[4]),
        "beta_CMA": float(beta[5]),
        "beta_MOM": float(beta[6]),
        "p_Mkt_RF": 2.0 * (1.0 - normal_cdf(abs(float(tvals[1])))),
        "p_SMB": 2.0 * (1.0 - normal_cdf(abs(float(tvals[2])))),
        "p_HML": 2.0 * (1.0 - normal_cdf(abs(float(tvals[3])))),
        "p_RMW": 2.0 * (1.0 - normal_cdf(abs(float(tvals[4])))),
        "p_CMA": 2.0 * (1.0 - normal_cdf(abs(float(tvals[5])))),
        "p_MOM": 2.0 * (1.0 - normal_cdf(abs(float(tvals[6])))),
    }


def main() -> None:
    market, cid2ticker, _member = load_market()
    rank_panels = load_rank_panels()
    mask = common_rank_template(rank_panels)

    fundamentals = pd.read_csv(DATA_DIR / "ndx_fundamental_data.csv")
    chars = build_characteristic_panel(
        mkt=market,
        fundamentals_long=fundamentals,
        companyid2ticker=cid2ticker,
        months=list(mask.index),
        universe=list(mask.columns),
    )

    baselines = {}
    baselines.update(build_factor_baselines(chars, mask))
    baselines.update(build_ml_baselines(chars, mask, market))

    factors = load_cached_ff_factors()
    rows = []
    for name in ORDER:
        panel = baselines[name]
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
        res = hac_ols_ff6(daily["Q1_minus_Q5"], factors, nw_lags=5)
        rows.append(
            {
                "Model": DISPLAY_NAMES[name],
                "alpha_ann_pct": res["alpha_ann"] * 100,
                "alpha_t": res["alpha_t"],
                "alpha_p": res["alpha_p"],
                "beta_MKT": res["beta_Mkt_RF"],
                "beta_SMB": res["beta_SMB"],
                "beta_HML": res["beta_HML"],
                "beta_RMW": res["beta_RMW"],
                "beta_CMA": res["beta_CMA"],
                "beta_MOM": res["beta_MOM"],
                "r2_adj": res["r2_adj"],
                "nobs": res["nobs"],
            }
        )

    out = pd.DataFrame(rows)
    TABLE_DIR.mkdir(exist_ok=True)
    out.to_csv(OUT_CSV, index=False)
    print(out.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"Wrote {OUT_CSV}")


if __name__ == "__main__":
    main()
