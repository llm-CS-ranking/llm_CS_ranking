"""
Factor analysis utilities for LLM asset-pricing experiments.

Provides:
  - Fama-French / momentum factor daily data loader (cached on disk).
  - Spanning regressions (CAPM / FF3 / Carhart-4 / FF5 / FF6) with HAC SE.
  - Cross-sectional characteristic panel (within the 30-stock universe).
  - Within-universe long-short factor-mimicking portfolios.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


# ────────────────────────────────────────────────────────────────────
# Fama-French factor data
# ────────────────────────────────────────────────────────────────────

def load_ff_factors_daily(
    cache_path: str | Path = "data/ff_factors_daily.csv",
    start: str = "2023-01-01",
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Load daily Fama-French 5-factor + momentum factors in **decimal** units.

    Columns: ``Mkt_RF, SMB, HML, RMW, CMA, MOM, RF``, index ``DATE`` (tz-naive).
    Values are decimals (e.g. 0.0047 for +0.47%).
    """
    cache_path = Path(cache_path)
    if cache_path.exists() and not force_refresh:
        df = pd.read_csv(cache_path, parse_dates=["DATE"]).set_index("DATE")
        return df.sort_index()

    import pandas_datareader.data as web

    ff5 = web.DataReader(
        "F-F_Research_Data_5_Factors_2x3_daily", "famafrench", start=start
    )[0]
    mom = web.DataReader("F-F_Momentum_Factor_daily", "famafrench", start=start)[0]
    mom = mom.rename(columns={mom.columns[0]: "MOM"})

    df = ff5.join(mom, how="outer").sort_index()
    df = df.rename(columns={"Mkt-RF": "Mkt_RF"})
    df = df / 100.0

    df.index.name = "DATE"
    df.index = pd.to_datetime(df.index).tz_localize(None)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.reset_index().to_csv(cache_path, index=False)
    return df


# ────────────────────────────────────────────────────────────────────
# Spanning test (factor regression with HAC SE)
# ────────────────────────────────────────────────────────────────────

FACTOR_SETS = {
    "CAPM":      ["Mkt_RF"],
    "FF3":       ["Mkt_RF", "SMB", "HML"],
    "Carhart4":  ["Mkt_RF", "SMB", "HML", "MOM"],
    "FF5":       ["Mkt_RF", "SMB", "HML", "RMW", "CMA"],
    "FF6":       ["Mkt_RF", "SMB", "HML", "RMW", "CMA", "MOM"],
}


def spanning_test(
    port_daily: pd.Series,
    factors: pd.DataFrame,
    factor_names: Iterable[str],
    nw_lags: int = 5,
    annualization: int = 252,
    is_excess: bool = True,
) -> dict:
    """Regress daily portfolio return on factor returns with Newey-West SE.

    Parameters
    ----------
    port_daily : daily portfolio return (decimal).
    factors    : daily factor DataFrame (decimal). Must include ``RF`` column
                 when ``is_excess=False``.
    factor_names : factors to use as regressors.
    is_excess  : if False, subtract ``RF`` from ``port_daily`` first (long-only
                 portfolios).  For long-short portfolios set ``True``.
    """
    import statsmodels.api as sm

    y = port_daily.dropna()
    y.index = pd.to_datetime(y.index).tz_localize(None)

    df = factors.reindex(y.index).dropna()
    y = y.reindex(df.index)

    if not is_excess:
        y = y - df["RF"]

    X = df[list(factor_names)].copy()
    X = sm.add_constant(X, has_constant="add")

    model = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": nw_lags})

    alpha_daily = float(model.params["const"])
    alpha_ann = (1.0 + alpha_daily) ** annualization - 1.0
    alpha_t = float(model.tvalues["const"])
    alpha_p = float(model.pvalues["const"])

    out = {
        "alpha_ann": alpha_ann,
        "alpha_daily": alpha_daily,
        "alpha_t": alpha_t,
        "alpha_p": alpha_p,
        "r2": float(model.rsquared),
        "r2_adj": float(model.rsquared_adj),
        "nobs": int(model.nobs),
    }
    for f in factor_names:
        out[f"beta_{f}"] = float(model.params[f])
        out[f"t_{f}"] = float(model.tvalues[f])
        out[f"p_{f}"] = float(model.pvalues[f])

    return out


def spanning_test_table(
    port_daily: pd.Series,
    factors: pd.DataFrame,
    factor_sets: dict | None = None,
    nw_lags: int = 5,
    is_excess: bool = True,
) -> pd.DataFrame:
    """Run spanning tests across multiple factor models and return one row per model."""
    if factor_sets is None:
        factor_sets = FACTOR_SETS
    rows = []
    for name, fs in factor_sets.items():
        res = spanning_test(port_daily, factors, fs, nw_lags=nw_lags, is_excess=is_excess)
        res["model"] = name
        rows.append(res)
    df = pd.DataFrame(rows).set_index("model")
    return df


# ────────────────────────────────────────────────────────────────────
# Characteristic panel (within 30-stock universe)
# ────────────────────────────────────────────────────────────────────

# S&P Capital IQ data item IDs used as proxies.
DATA_ITEM_IDS = {
    "total_assets": 1007,
    "total_equity": 1275,
    "net_income": 15,        # Net Income - (IS)
    "total_revenues": 28,
    "asset_growth_1y": 4203,  # Total Assets 1Y Growth %
    "roe_pct": 4128,          # Return On Equity %
}


def _monthly_price_panel(mkt: pd.DataFrame, price_col: str = "DIV_ADJ_CLOSE") -> pd.DataFrame:
    m = mkt.reset_index()
    m["DATE"] = pd.to_datetime(m["DATE"])
    prices = (
        m.pivot_table(index="DATE", columns="TICKERSYMBOL", values=price_col, aggfunc="last")
        .sort_index()
    )
    return prices


def _monthly_caps_panel(mkt: pd.DataFrame, cap_col: str = "MKTCAP") -> pd.DataFrame:
    m = mkt.reset_index()
    m["DATE"] = pd.to_datetime(m["DATE"])
    caps = (
        m.pivot_table(index="DATE", columns="TICKERSYMBOL", values=cap_col, aggfunc="last")
        .sort_index()
    )
    return caps


def build_characteristic_panel(
    mkt: pd.DataFrame,
    fundamentals_long: pd.DataFrame,
    companyid2ticker: dict[int, str],
    months: Iterable[str],
    universe: list[str],
    price_col: str = "DIV_ADJ_CLOSE",
    cap_col: str = "MKTCAP",
) -> dict[str, pd.DataFrame]:
    """Build a panel of stock characteristics observable *before* month m starts.

    Returns dict[char_name] -> DataFrame with index=MONTH (YYYY-MM), columns=ticker.

    Characteristics (all point-in-time, lookback-only):
      - size       : log(market cap on last trading day before month m)
      - mom_12_1   : past 12-month return ending 1 month before m (12-1 momentum)
      - rev_1m     : past 1-month return ending at month m-1 end (short-term reversal)
      - vol_60d    : annualized 60-day realized volatility ending before m
      - ep_ratio   : trailing 4Q net income / market cap at m-1 end
      - roe        : latest reported ROE % (filing_date < m start)
      - asset_gr   : latest reported asset growth % (filing_date < m start)
    """
    prices = _monthly_price_panel(mkt, price_col)
    caps = _monthly_caps_panel(mkt, cap_col)

    # Filter fundamentals to universe via COMPANYID mapping
    ticker2cid = {t: c for c, t in companyid2ticker.items() if t in universe}
    f = fundamentals_long.copy()
    f["FILINGDATE"] = pd.to_datetime(f["FILINGDATE"], errors="coerce")
    f = f.dropna(subset=["FILINGDATE"])

    # Build daily-return panel for momentum/vol
    rets_d = prices.pct_change(fill_method=None)

    month_list = sorted(set(months))

    chars = {k: {} for k in ["size", "mom_12_1", "rev_1m", "vol_60d",
                              "ep_ratio", "roe", "asset_gr"]}

    for mstr in month_list:
        m_start = pd.Period(mstr, freq="M").start_time  # 1st of month m
        m_prev_end = m_start - pd.Timedelta(days=1)

        # Locate last trading day strictly before m_start
        idx = prices.index.searchsorted(m_start, side="left") - 1
        if idx < 0:
            continue
        last_date = prices.index[idx]

        # ---- size: market cap at last trading day before m
        cap_row = caps.loc[last_date].reindex(universe)
        chars["size"][mstr] = np.log(cap_row.astype(float).replace({0: np.nan}))

        # ---- rev_1m: return over past ~21 trading days
        start_idx_1m = max(idx - 21, 0)
        base_price = prices.iloc[start_idx_1m].reindex(universe)
        rev_1m = prices.loc[last_date].reindex(universe) / base_price - 1.0
        chars["rev_1m"][mstr] = rev_1m

        # ---- mom_12_1: return from t-252 to t-21
        start_idx_12 = max(idx - 252, 0)
        end_idx_1 = max(idx - 21, 0)
        p_252 = prices.iloc[start_idx_12].reindex(universe)
        p_21 = prices.iloc[end_idx_1].reindex(universe)
        mom = p_21 / p_252 - 1.0
        chars["mom_12_1"][mstr] = mom

        # ---- vol_60d: annualized std of past 60 daily returns (ending last_date)
        window = rets_d.iloc[max(idx - 60, 0): idx + 1]
        vol = window.std(ddof=0) * np.sqrt(252)
        chars["vol_60d"][mstr] = vol.reindex(universe)

        # ---- fundamentals-based: select latest filing < m_start per company, per item
        f_ok = f.loc[f["FILINGDATE"] < m_start]

        # ep_ratio: trailing 4Q net income / current market cap
        ni = f_ok.loc[f_ok["DATAITEMID"] == DATA_ITEM_IDS["net_income"]].copy()
        # keep last 4 quarters per company
        if not ni.empty:
            ni_sorted = ni.sort_values(["COMPANYID", "QUARTER", "FILINGDATE"])
            ni_last = ni_sorted.drop_duplicates(subset=["COMPANYID", "QUARTER"], keep="last")
            # take the 4 most recent quarters
            ni_last = (
                ni_last.sort_values(["COMPANYID", "QUARTER"])
                .groupby("COMPANYID")
                .tail(4)
            )
            ttm_ni = ni_last.groupby("COMPANYID")["DATAITEMVALUE"].sum()
            ep_vals = {}
            for tkr in universe:
                cid = ticker2cid.get(tkr)
                if cid is None or cid not in ttm_ni.index:
                    ep_vals[tkr] = np.nan
                    continue
                mc = cap_row.get(tkr, np.nan)
                if pd.isna(mc) or mc == 0:
                    ep_vals[tkr] = np.nan
                else:
                    ep_vals[tkr] = float(ttm_ni.loc[cid]) / float(mc)
            chars["ep_ratio"][mstr] = pd.Series(ep_vals)
        else:
            chars["ep_ratio"][mstr] = pd.Series({t: np.nan for t in universe})

        # roe: latest reported ROE %
        for key, char_key in [("roe_pct", "roe"), ("asset_growth_1y", "asset_gr")]:
            ser = f_ok.loc[f_ok["DATAITEMID"] == DATA_ITEM_IDS[key]]
            vals = {}
            for tkr in universe:
                cid = ticker2cid.get(tkr)
                if cid is None:
                    vals[tkr] = np.nan
                    continue
                sub = ser.loc[ser["COMPANYID"] == cid]
                if sub.empty:
                    vals[tkr] = np.nan
                    continue
                # most recent filing, then most recent quarter
                sub = sub.sort_values(["FILINGDATE", "QUARTER"])
                vals[tkr] = float(sub.iloc[-1]["DATAITEMVALUE"])
            chars[char_key][mstr] = pd.Series(vals)

    result = {}
    for k, dct in chars.items():
        df = pd.DataFrame(dct).T  # index=MONTH, columns=ticker
        df = df.reindex(columns=universe)
        df.index.name = "MONTH"
        result[k] = df.sort_index()
    return result


def cross_sectional_rank(df: pd.DataFrame, ascending: bool = True) -> pd.DataFrame:
    """Per-row (per-month) rank across tickers. NaN-safe.

    ``ascending=True`` means higher value → higher rank number.
    For characteristics we rescale to [0, 1] after ranking so +1 = top in universe,
    0 = bottom.  Result is signed rank z-like score.
    """
    n = df.notna().sum(axis=1)
    r = df.rank(axis=1, method="average", ascending=ascending)
    norm = r.sub(1).div(n.sub(1), axis=0)  # in [0, 1]
    return norm


# ────────────────────────────────────────────────────────────────────
# Q1/Q5 characteristic exposure
# ────────────────────────────────────────────────────────────────────

def quintile_char_exposure(
    rank_panel: pd.DataFrame,
    char_panel: pd.DataFrame,
    n_quantiles: int = 5,
    char_ascending_good: bool = True,
) -> pd.DataFrame:
    """For each month, compute the average characteristic value (rank-normalized
    to [0, 1]) within Q1 (top-ranked) and Q5 (bottom-ranked) stocks.

    Returns wide DataFrame with index=MONTH and columns {"Q1", "Q5", "Q1_minus_Q5"}.
    """
    char_norm = cross_sectional_rank(char_panel, ascending=char_ascending_good)

    out = []
    common = rank_panel.index.intersection(char_norm.index)
    for mstr in common:
        pred = rank_panel.loc[mstr].dropna().astype(float)
        if pred.empty:
            continue
        order = pred.sort_values(kind="mergesort").index.tolist()
        order = [t for t in order if t in char_norm.columns]
        names = np.array(order)
        buckets = np.array_split(names, n_quantiles)
        q1 = buckets[0].tolist()
        q5 = buckets[-1].tolist()

        c_row = char_norm.loc[mstr]
        q1_val = c_row.reindex(q1).mean()
        q5_val = c_row.reindex(q5).mean()
        out.append({
            "MONTH": mstr,
            "Q1": q1_val,
            "Q5": q5_val,
            "Q1_minus_Q5": q1_val - q5_val,
        })
    return pd.DataFrame(out).set_index("MONTH")


# ────────────────────────────────────────────────────────────────────
# Within-universe long-short mimicking portfolios
# ────────────────────────────────────────────────────────────────────

def within_universe_factor_ls(
    char_panel: pd.DataFrame,
    mkt: pd.DataFrame,
    n_quantiles: int = 5,
    char_ascending_good: bool = True,
    price_col: str = "DIV_ADJ_CLOSE",
    lag_days: int = 2,
) -> pd.Series:
    """Build a daily long-short factor-mimicking portfolio from a characteristic panel.

    Monthly: sort the universe by the characteristic (higher = more "long"
    when ``char_ascending_good=True``); long top-1/n_quantiles, short
    bottom-1/n_quantiles; equal weighted; rebalanced on the ``lag_days``-th
    trading day after month start.
    """
    m = mkt.reset_index()
    m["DATE"] = pd.to_datetime(m["DATE"])
    prices = m.pivot_table(
        index="DATE", columns="TICKERSYMBOL", values=price_col, aggfunc="last"
    ).sort_index()
    rets = prices.pct_change(fill_method=None).fillna(0.0)

    all_dates = prices.index
    tickers = prices.columns

    months = sorted(char_panel.dropna(how="all").index.astype(str))
    sched = []
    for ms in months:
        m_start = pd.Period(ms, freq="M").start_time
        pos0 = all_dates.searchsorted(m_start, side="left") + lag_days
        if pos0 >= len(all_dates):
            continue
        sched.append((ms, all_dates[pos0]))
    sched = pd.DataFrame(sched, columns=["MONTH", "EXEC_DATE"]).drop_duplicates("EXEC_DATE")

    W = pd.DataFrame(0.0, index=all_dates, columns=tickers)

    for i in range(len(sched)):
        ms = sched.iloc[i]["MONTH"]
        d0 = sched.iloc[i]["EXEC_DATE"]
        d1 = sched.iloc[i + 1]["EXEC_DATE"] if i + 1 < len(sched) else (all_dates[-1] + pd.Timedelta(days=1))
        mask = (all_dates >= d0) & (all_dates < d1)

        c = char_panel.loc[ms] if ms in char_panel.index else None
        if c is None:
            continue
        c = c.dropna().astype(float)
        c = c[c.index.isin(tickers)]
        if c.empty:
            continue
        c_sorted = c.sort_values(ascending=not char_ascending_good)  # best at top
        names = c_sorted.index.tolist()
        buckets = np.array_split(names, n_quantiles)
        longs = buckets[0].tolist()
        shorts = buckets[-1].tolist()

        new_w = pd.Series(0.0, index=tickers)
        if longs:
            new_w.loc[longs] += 1.0 / len(longs)
        if shorts:
            new_w.loc[shorts] -= 1.0 / len(shorts)
        W.loc[mask, :] = new_w.values

    port = (W.shift(1).fillna(0.0) * rets).sum(axis=1)
    port.name = "LS"
    first = sched["EXEC_DATE"].iloc[0] if len(sched) else all_dates[0]
    return port.loc[first:]


# ────────────────────────────────────────────────────────────────────
# Pretty LaTeX helpers
# ────────────────────────────────────────────────────────────────────

def sig_stars(p: float) -> str:
    if pd.isna(p):
        return ""
    if p < 0.01:
        return "^{***}"
    if p < 0.05:
        return "^{**}"
    if p < 0.10:
        return "^{*}"
    return ""
