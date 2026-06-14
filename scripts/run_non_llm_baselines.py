from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils import backtest_quintiles_with_universe_ew_capw, portfolio_metrics, rank_ic_by_month, rank_ic_summary


DATA_DIR = ROOT / "data" / "ndx_rolling_20260107"
RESULT_DIR = ROOT / "result"
TABLE_DIR = ROOT / "result" / "tables"
OUT_CSV = TABLE_DIR / "non_llm_baselines.csv"
OUT_TEX = TABLE_DIR / "non_llm_baselines.tex"

MODES = ["return", "sharpe", "sortino"]
COST_BPS = 40.0
LAG_DAYS = 2

DATA_ITEM_IDS = {
    "net_income": 15,
    "asset_growth_1y": 4203,
    "roe_pct": 4128,
}


def build_characteristic_panel(
    mkt: pd.DataFrame,
    fundamentals_long: pd.DataFrame,
    companyid2ticker: dict[int, str],
    months: list[str],
    universe: list[str],
    price_col: str = "DIV_ADJ_CLOSE",
    cap_col: str = "MKTCAP",
) -> dict[str, pd.DataFrame]:
    """Build point-in-time characteristics used for non-LLM ranker baselines."""
    m = mkt.reset_index()
    prices = m.pivot_table(index="DATE", columns="TICKERSYMBOL", values=price_col, aggfunc="last").sort_index()
    caps = m.pivot_table(index="DATE", columns="TICKERSYMBOL", values=cap_col, aggfunc="last").sort_index()
    rets_d = prices.pct_change(fill_method=None)

    ticker2cid = {t: c for c, t in companyid2ticker.items() if t in universe}
    f = fundamentals_long.copy()
    f["FILINGDATE"] = pd.to_datetime(f["FILINGDATE"], errors="coerce")
    f = f.dropna(subset=["FILINGDATE"])

    chars = {k: {} for k in ["size", "mom_12_1", "rev_1m", "vol_60d", "ep_ratio", "roe", "asset_gr"]}
    for mstr in sorted(set(months)):
        m_start = pd.Period(mstr, freq="M").start_time
        idx = prices.index.searchsorted(m_start, side="left") - 1
        if idx < 0:
            continue
        last_date = prices.index[idx]

        cap_row = caps.loc[last_date].reindex(universe)
        chars["size"][mstr] = np.log(cap_row.astype(float).replace({0: np.nan}))

        p_last = prices.loc[last_date].reindex(universe)
        p_21 = prices.iloc[max(idx - 21, 0)].reindex(universe)
        p_252 = prices.iloc[max(idx - 252, 0)].reindex(universe)
        p_21_for_mom = prices.iloc[max(idx - 21, 0)].reindex(universe)
        chars["rev_1m"][mstr] = p_last / p_21 - 1.0
        chars["mom_12_1"][mstr] = p_21_for_mom / p_252 - 1.0
        chars["vol_60d"][mstr] = (rets_d.iloc[max(idx - 60, 0): idx + 1].std(ddof=0) * np.sqrt(252)).reindex(universe)

        f_ok = f.loc[f["FILINGDATE"] < m_start]
        ni = f_ok.loc[f_ok["DATAITEMID"] == DATA_ITEM_IDS["net_income"]].copy()
        if not ni.empty:
            ni_last = ni.sort_values(["COMPANYID", "QUARTER", "FILINGDATE"]).drop_duplicates(
                subset=["COMPANYID", "QUARTER"], keep="last"
            )
            ni_last = ni_last.sort_values(["COMPANYID", "QUARTER"]).groupby("COMPANYID").tail(4)
            ttm_ni = ni_last.groupby("COMPANYID")["DATAITEMVALUE"].sum()
        else:
            ttm_ni = pd.Series(dtype=float)

        ep_vals = {}
        for tkr in universe:
            cid = ticker2cid.get(tkr)
            mc = cap_row.get(tkr, np.nan)
            ep_vals[tkr] = np.nan if cid is None or cid not in ttm_ni.index or pd.isna(mc) or mc == 0 else float(ttm_ni.loc[cid]) / float(mc)
        chars["ep_ratio"][mstr] = pd.Series(ep_vals)

        for item_key, char_key in [("roe_pct", "roe"), ("asset_growth_1y", "asset_gr")]:
            item = f_ok.loc[f_ok["DATAITEMID"] == DATA_ITEM_IDS[item_key]]
            vals = {}
            for tkr in universe:
                cid = ticker2cid.get(tkr)
                sub = item.loc[item["COMPANYID"] == cid] if cid is not None else pd.DataFrame()
                if sub.empty:
                    vals[tkr] = np.nan
                else:
                    vals[tkr] = float(sub.sort_values(["FILINGDATE", "QUARTER"]).iloc[-1]["DATAITEMVALUE"])
            chars[char_key][mstr] = pd.Series(vals)

    return {
        name: pd.DataFrame(vals).T.reindex(index=months, columns=universe).rename_axis("MONTH")
        for name, vals in chars.items()
    }


def load_market() -> tuple[pd.DataFrame, dict[int, str], pd.DataFrame]:
    trading = pd.read_csv(DATA_DIR / "ndx_tradingiteminfo.csv")
    member = pd.read_csv(DATA_DIR / "ndx_data_member.csv", parse_dates=["DATE"])
    market = pd.read_csv(DATA_DIR / "ndx_market_data.csv", parse_dates=["DATE"])

    # Keep the latest mapping when a ticker has multiple trading item ids.
    latest_tid = (
        trading.sort_values(["TICKERSYMBOL", "TRADINGITEMID"])
        .drop_duplicates("TICKERSYMBOL", keep="last")
    )
    tid2ticker = dict(zip(latest_tid["TRADINGITEMID"], latest_tid["TICKERSYMBOL"]))
    cid2ticker = dict(zip(latest_tid["COMPANYID"], latest_tid["TICKERSYMBOL"]))

    market = market.loc[market["TRADINGITEMID"].isin(tid2ticker)].copy()
    market["TICKERSYMBOL"] = market["TRADINGITEMID"].map(tid2ticker)
    market = market.drop_duplicates(subset=["DATE", "TICKERSYMBOL"], keep="last")
    market = market.set_index(["DATE", "TICKERSYMBOL"]).sort_index()

    member = member.loc[member["TRADINGITEMID"].isin(tid2ticker)].copy()
    member["TICKERSYMBOL"] = member["TRADINGITEMID"].map(tid2ticker)
    return market, cid2ticker, member


def load_rank_panels() -> dict[str, pd.DataFrame]:
    panels: dict[str, pd.DataFrame] = {}
    for path in sorted(RESULT_DIR.glob("*_result.pkl")):
        with path.open("rb") as f:
            panels[path.stem.replace("_result", "")] = pickle.load(f)
    if not panels:
        raise RuntimeError(f"No rank-panel pickle files found in {RESULT_DIR}")
    return panels


def common_rank_template(rank_panels: dict[str, pd.DataFrame]) -> pd.DataFrame:
    # Use the union of all months/tickers where any existing LLM panel produced ranks.
    months = sorted({m for p in rank_panels.values() for m in p.index.astype(str)})
    tickers = sorted({t for p in rank_panels.values() for t in p.columns.astype(str)})
    mask = pd.DataFrame(False, index=months, columns=tickers)
    for p in rank_panels.values():
        pp = p.copy()
        pp.index = pp.index.astype(str)
        pp.columns = pp.columns.astype(str)
        mask.loc[pp.index, pp.columns] |= pp.notna()
    return mask


def score_to_rank(scores: pd.DataFrame, mask: pd.DataFrame, higher_is_better: bool = True) -> pd.DataFrame:
    aligned = scores.reindex(index=mask.index, columns=mask.columns)
    aligned = aligned.where(mask)
    ranks = aligned.rank(axis=1, method="first", ascending=not higher_is_better)
    return ranks


def composite_score(parts: list[pd.DataFrame], mask: pd.DataFrame) -> pd.DataFrame:
    normed = []
    for part in parts:
        x = part.reindex(index=mask.index, columns=mask.columns).where(mask)
        n = x.notna().sum(axis=1)
        r = x.rank(axis=1, method="average", ascending=True)
        normed.append(r.sub(1).div(n.sub(1), axis=0))
    return pd.concat(normed, keys=range(len(normed))).groupby(level=1).mean()


def build_factor_baselines(chars: dict[str, pd.DataFrame], mask: pd.DataFrame) -> dict[str, pd.DataFrame]:
    # Higher score is better for all constructed scores.
    mom = chars["mom_12_1"]
    reversal = -chars["rev_1m"]
    low_vol = -chars["vol_60d"]
    value = chars["ep_ratio"]
    quality = chars["roe"]
    conservative = -chars["asset_gr"]

    scores = {
        "Mom12-1": mom,
        "Reversal1M": reversal,
        "LowVol60D": low_vol,
        "ValueEP": value,
        "QualityROE": quality,
        "ConservativeInv": conservative,
        "Composite6": composite_score([mom, reversal, low_vol, value, quality, conservative], mask),
    }
    return {name: score_to_rank(score, mask, higher_is_better=True) for name, score in scores.items()}


def realized_monthly_returns(market: pd.DataFrame, months: list[str], tickers: list[str]) -> pd.DataFrame:
    m = market.reset_index()
    prices = m.pivot_table(index="DATE", columns="TICKERSYMBOL", values="DIV_ADJ_CLOSE", aggfunc="last").sort_index()
    month_end = prices.resample("ME").last()
    mret = month_end.pct_change(fill_method=None)
    mret.index = mret.index.to_period("M").astype(str)
    return mret.reindex(index=months, columns=tickers)


def build_ml_baselines(chars: dict[str, pd.DataFrame], mask: pd.DataFrame, market: pd.DataFrame) -> dict[str, pd.DataFrame]:
    features = ["mom_12_1", "rev_1m", "vol_60d", "ep_ratio", "roe", "asset_gr", "size"]
    months = list(mask.index)
    tickers = list(mask.columns)
    future_ret = realized_monthly_returns(market, months, tickers).shift(-1)

    feature_panel = {
        f: chars[f].reindex(index=months, columns=tickers).where(mask)
        for f in features
    }

    models = {
        "Ridge": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "RandomForest": RandomForestRegressor(
            n_estimators=300,
            max_depth=4,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1,
        ),
    }
    preds = {name: pd.DataFrame(np.nan, index=months, columns=tickers) for name in models}

    for i, month in enumerate(months):
        # Expanding-window training only on prior months, avoiding the target month.
        train_months = months[:i]
        rows = []
        y = []
        for tm in train_months:
            for t in tickers:
                if not bool(mask.loc[tm, t]):
                    continue
                vals = [feature_panel[f].loc[tm, t] for f in features]
                target = future_ret.loc[tm, t]
                if any(pd.isna(v) for v in vals) or pd.isna(target):
                    continue
                rows.append(vals)
                y.append(target)

        test_tickers = [t for t in tickers if bool(mask.loc[month, t])]
        if len(rows) < 60 or len(test_tickers) < 5:
            continue

        X_train = np.asarray(rows, dtype=float)
        y_train = np.asarray(y, dtype=float)
        X_test = []
        valid_tickers = []
        for t in test_tickers:
            vals = [feature_panel[f].loc[month, t] for f in features]
            if any(pd.isna(v) for v in vals):
                continue
            X_test.append(vals)
            valid_tickers.append(t)
        if len(valid_tickers) < 5:
            continue

        X_test_arr = np.asarray(X_test, dtype=float)
        for name, model in models.items():
            model.fit(X_train, y_train)
            preds[name].loc[month, valid_tickers] = model.predict(X_test_arr)

    return {name: score_to_rank(score, mask, higher_is_better=True) for name, score in preds.items()}


def evaluate(rank_panel: pd.DataFrame, market: pd.DataFrame) -> dict[str, float]:
    daily = backtest_quintiles_with_universe_ew_capw(
        rank_panel=rank_panel,
        market_df=market,
        price_col="DIV_ADJ_CLOSE",
        cap_col="MKTCAP",
        n_quantiles=5,
        lag_days=LAG_DAYS,
        cost_bps=COST_BPS,
        add_long_short=True,
    )
    metrics = portfolio_metrics(daily)
    ic = rank_ic_summary(rank_ic_by_month(rank_panel, market, lag_months=0, method="spearman"))
    return {
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
    }


def main() -> None:
    market, cid2ticker, _member = load_market()
    rank_panels = load_rank_panels()
    mask = common_rank_template(rank_panels)

    fundamentals = pd.read_csv(DATA_DIR / "ndx_fundamental_data.csv")
    months = list(mask.index)
    universe = list(mask.columns)
    chars = build_characteristic_panel(
        mkt=market,
        fundamentals_long=fundamentals,
        companyid2ticker=cid2ticker,
        months=months,
        universe=universe,
    )

    baselines: dict[str, pd.DataFrame] = {}
    baselines.update(build_factor_baselines(chars, mask))
    baselines.update(build_ml_baselines(chars, mask, market))

    rows = []
    for name, panel in baselines.items():
        try:
            row = evaluate(panel, market)
            row["Baseline"] = name
            rows.append(row)
            (RESULT_DIR / f"baseline_{name}_rank_panel.pkl").write_bytes(pickle.dumps(panel))
            print(f"OK {name}: LS Sharpe={row['LS_Sharpe']:.2f}, IC={row['Mean_IC']:.3f}")
        except Exception as exc:
            print(f"ERROR {name}: {exc}")

    out = pd.DataFrame(rows).set_index("Baseline").sort_values("LS_Sharpe", ascending=False)
    TABLE_DIR.mkdir(exist_ok=True)
    out.to_csv(OUT_CSV)

    pct_cols = ["Q1_CAGR", "Q1_MDD", "LS_CAGR", "LS_MDD", "Mean_IC"]
    tex = out.reset_index().copy()
    for col in pct_cols:
        tex[col] = tex[col].map(lambda x: f"{x*100:.1f}")
    for col in ["Q1_Sharpe", "LS_Sharpe", "LS_Sortino", "ICIR"]:
        tex[col] = tex[col].map(lambda x: f"{x:.2f}")
    tex["Months"] = tex["Months"].astype(int).astype(str)
    tex = tex[["Baseline", "Q1_CAGR", "Q1_Sharpe", "LS_CAGR", "LS_Sharpe", "LS_Sortino", "Mean_IC", "ICIR", "Months"]]
    headers = ["Baseline", "Q1 CAGR", "Q1 Shp", "LS CAGR", "LS Shp", "LS Sortino", "IC", "ICIR", "Months"]
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Non-LLM baseline performance on the same SRLLM information set. CAGR and IC are reported in percent.}",
        "\\label{tab:nonllm-baselines}",
        "\\small",
        "\\begin{tabular}{lrrrrrrrr}",
        "\\toprule",
        " & ".join(headers) + " \\\\",
        "\\midrule",
    ]
    for _, row in tex.iterrows():
        lines.append(" & ".join(str(row[h]) for h in tex.columns) + " \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])
    OUT_TEX.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {OUT_CSV}")
    print(f"Wrote {OUT_TEX}")


if __name__ == "__main__":
    main()
