from __future__ import annotations

import pickle
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils import backtest_quintiles_with_universe_ew_capw, portfolio_metrics, rank_ic_by_month, rank_ic_summary


RESULT_DIR = ROOT / "result"
TABLE_DIR = ROOT / "result" / "tables"
OUT_CSV = TABLE_DIR / "retained_paper_models_results.csv"
OUT_TEX = TABLE_DIR / "retained_paper_models_longshort.tex"

MODES = ["return", "sharpe", "sortino"]
MODE_LABELS = {"return": "Exp. Return", "sharpe": "Sharpe", "sortino": "Sortino"}

# Existing paper models retained after excluding Claude and currently deprecated/replaced models.
MODEL_CONFIG = {
    "gpt-4o-2024-05-13": {"display": "GPT-4o", "vendor": "OpenAI"},
    "gpt-4o-mini-2024-07-18": {"display": "GPT-4o mini", "vendor": "OpenAI"},
    "gpt-4.1-2025-04-14": {"display": "GPT-4.1", "vendor": "OpenAI"},
    "gpt-4.1-mini-2025-04-14": {"display": "GPT-4.1 mini", "vendor": "OpenAI"},
    "gemini-2.5-flash": {"display": "Gemini 2.5 Flash", "vendor": "Google"},
    "gemini-2.5-pro": {"display": "Gemini 2.5 Pro", "vendor": "Google"},
    "gemini-3-flash-preview": {"display": "Gemini 3 Flash", "vendor": "Google"},
    "gemini-3-pro-preview": {"display": "Gemini 3 Pro", "vendor": "Google"},
}

EXCLUDED = {
    "claude-haiku-4-5-20251001": "Claude API unavailable for this project",
    "claude-sonnet-4-20250514": "Claude API unavailable for this project",
    "claude-opus-4-20250514": "Claude API unavailable for this project",
}


def load_market() -> tuple[pd.DataFrame, dict[int, str], pd.DataFrame]:
    data_dir = ROOT / "data" / "ndx_rolling_20260107"
    trading = pd.read_csv(data_dir / "ndx_tradingiteminfo.csv")
    member = pd.read_csv(data_dir / "ndx_data_member.csv", parse_dates=["DATE"])
    market = pd.read_csv(data_dir / "ndx_market_data.csv", parse_dates=["DATE"])

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


def evaluate_panel(rank_panel: pd.DataFrame, market: pd.DataFrame) -> dict[str, float]:
    daily = backtest_quintiles_with_universe_ew_capw(
        rank_panel=rank_panel,
        market_df=market,
        price_col="DIV_ADJ_CLOSE",
        cap_col="MKTCAP",
        n_quantiles=5,
        lag_days=2,
        cost_bps=40.0,
        add_long_short=True,
    )
    metrics = portfolio_metrics(daily)
    ic_summary = rank_ic_summary(rank_ic_by_month(rank_panel, market, lag_months=0, method="spearman"))
    return {
        "Q1_CAGR": float(metrics.loc["CAGR", "Q1"]),
        "Q1_Sharpe": float(metrics.loc["Sharpe", "Q1"]),
        "Q1_MDD": float(metrics.loc["MDD", "Q1"]),
        "LS_CAGR": float(metrics.loc["CAGR", "Q1_minus_Q5"]),
        "LS_Sharpe": float(metrics.loc["Sharpe", "Q1_minus_Q5"]),
        "LS_Sortino": float(metrics.loc["Sortino", "Q1_minus_Q5"]),
        "LS_MDD": float(metrics.loc["MDD", "Q1_minus_Q5"]),
        "Mean_IC": float(ic_summary["mean_IC"]),
        "ICIR": float(ic_summary["ICIR_annual"]),
        "Months": int(ic_summary["n_months"]),
    }


def main() -> None:
    market, _cid2ticker, _member = load_market()
    rows = []
    missing = []

    for model, cfg in MODEL_CONFIG.items():
        for mode in MODES:
            path = RESULT_DIR / f"{mode}_{model}_result.pkl"
            if not path.exists():
                missing.append(f"{mode}_{model}")
                continue
            with path.open("rb") as f:
                panel = pickle.load(f)
            row = evaluate_panel(panel, market)
            row.update(
                {
                    "Model": cfg["display"],
                    "ModelID": model,
                    "Vendor": cfg["vendor"],
                    "Mode": mode,
                    "ModeLabel": MODE_LABELS[mode],
                }
            )
            rows.append(row)

    out = pd.DataFrame(rows)
    TABLE_DIR.mkdir(exist_ok=True)
    out.to_csv(OUT_CSV, index=False)

    # Compact long-short table for the manuscript.
    pivot = out.pivot(index="Model", columns="Mode", values=["LS_CAGR", "LS_Sharpe", "LS_MDD"])
    order = [cfg["display"] for cfg in MODEL_CONFIG.values()]
    pivot = pivot.reindex(order)

    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{Retained non-Claude, non-deprecated paper-model long--short performance.}",
        "\\label{tab:retained-longshort}",
        "\\small",
        "\\resizebox{0.8\\textwidth}{!}{",
        "\\begin{tabular}{l rrr rrr rrr}",
        "\\toprule",
        "& \\multicolumn{3}{c}{\\textbf{Exp. Return}} & \\multicolumn{3}{c}{\\textbf{Sharpe}} & \\multicolumn{3}{c}{\\textbf{Sortino}} \\\\",
        "\\cmidrule(lr){2-4} \\cmidrule(lr){5-7} \\cmidrule(lr){8-10}",
        "\\textbf{Model} & CAGR & Shp & MDD & CAGR & Shp & MDD & CAGR & Shp & MDD \\\\",
        "\\midrule",
    ]
    for model_name in pivot.index:
        cells = [model_name]
        for mode in MODES:
            cagr = pivot.loc[model_name, ("LS_CAGR", mode)] * 100
            shp = pivot.loc[model_name, ("LS_Sharpe", mode)]
            mdd = pivot.loc[model_name, ("LS_MDD", mode)] * 100
            cells.extend([f"{cagr:.1f}", f"{shp:.2f}", f"{mdd:.1f}"])
        lines.append(" & ".join(cells) + " \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", "}", "\\end{table*}", ""])
    OUT_TEX.write_text("\n".join(lines), encoding="utf-8")

    excluded_path = TABLE_DIR / "retained_paper_models_excluded.txt"
    excluded_path.write_text("\n".join(f"{k}: {v}" for k, v in EXCLUDED.items()), encoding="utf-8")

    print(f"Wrote {OUT_CSV}")
    print(f"Wrote {OUT_TEX}")
    print(f"Missing panels: {missing}")
    print("Excluded:")
    for key, reason in EXCLUDED.items():
        print(f"- {key}: {reason}")


if __name__ == "__main__":
    main()
