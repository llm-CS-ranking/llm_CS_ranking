from __future__ import annotations

import pickle
import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from conf.model_list import MODEL_KNOWLEDGE_CUTOFF
from evaluate_input_ablation_models import INPUT_SETS, MODE_LABELS, MODEL_CONFIG, MODES, result_path
from evaluate_common_window import evaluate_panel
from evaluate_retained_paper_models import load_market
from run_baseline_ff6_spanning import hac_ols_ff6, load_cached_ff_factors
from utils import backtest_quintiles_with_universe_ew_capw

TABLE_DIR = ROOT / "result" / "tables"
PAPER_TABLE_DIR = ROOT / "paper" / "tables"
APPENDIX_TABLE_DIR = ROOT / "paper_appendix" / "tables"

OUT_RESULTS = TABLE_DIR / "conservative_input_ablation_results.csv"
OUT_TESTS = TABLE_DIR / "conservative_input_ablation_tests.csv"
OUT_LOADINGS = TABLE_DIR / "conservative_input_ablation_factor_loadings.csv"
OUT_TEX = TABLE_DIR / "conservative_input_ablation.tex"

REPRESENTATIVE = {
    "o3": "o3",
    "gemini-2.5-pro": "Gem. 2.5 Pro",
    "claude-opus-4-20250514": "Cl. Opus 4",
}


def first_full_month_after(date_str: str) -> str:
    return str((pd.Timestamp(date_str) + pd.offsets.MonthBegin(1)).to_period("M"))


def model_cutoff(model_id: str) -> str:
    if model_id in MODEL_KNOWLEDGE_CUTOFF:
        return MODEL_KNOWLEDGE_CUTOFF[model_id]
    for key in sorted(MODEL_KNOWLEDGE_CUTOFF, key=len, reverse=True):
        if model_id == key or model_id.startswith(f"{key}-"):
            return MODEL_KNOWLEDGE_CUTOFF[key]
    raise KeyError(model_id)


def load_panel(input_set: str, mode: str, model_id: str) -> pd.DataFrame | None:
    path = result_path(input_set, mode, model_id)
    if not path.exists():
        return None
    with path.open("rb") as f:
        panel = pickle.load(f)
    panel.index = panel.index.astype(str)
    panel.columns = panel.columns.astype(str)
    start = first_full_month_after(model_cutoff(model_id))
    return panel.loc[panel.index >= start]


def fmt_pct(value: float) -> str:
    text = f"{value * 100:.1f}"
    return "$-$" + text[1:] if text.startswith("-") else text


def fmt_num(value: float, decimals: int = 2) -> str:
    text = f"{value:.{decimals}f}"
    return "$-$" + text[1:] if text.startswith("-") else text


def sig_stars(pvalue: float) -> str:
    if pvalue < 0.01:
        return "^{***}"
    if pvalue < 0.05:
        return "^{**}"
    if pvalue < 0.10:
        return "^{*}"
    return ""


def fmt_coef(value: float, pvalue: float) -> str:
    text = fmt_num(value)
    stars = sig_stars(pvalue)
    if stars:
        text = f"{text}${stars}$"
    return text


def fmt_p(value: float) -> str:
    if value < 0.001:
        return "$<0.001$"
    return f"{value:.3f}"


def paired_table(df: pd.DataFrame, left: str, right: str, metric: str) -> pd.DataFrame:
    wide = (
        df[df["InputSet"].isin([left, right])]
        .pivot(index=["ModelID", "Model", "Mode"], columns="InputSet", values=metric)
        .reset_index()
    )
    wide["diff"] = wide[left] - wide[right]
    return wide.dropna(subset=[left, right, "diff"])


def clustered_sign_flip_pvalue(wide: pd.DataFrame) -> float:
    observed = abs(wide["diff"].mean())
    models = list(wide["ModelID"].drop_duplicates())
    stats = []
    for signs in product([-1, 1], repeat=len(models)):
        sign_map = dict(zip(models, signs, strict=True))
        stats.append(abs((wide["diff"] * wide["ModelID"].map(sign_map)).mean()))
    stats_arr = np.asarray(stats)
    return float((np.count_nonzero(stats_arr >= observed - 1e-12) + 1) / (len(stats_arr) + 1))


def bootstrap_ci(wide: pd.DataFrame, rng: np.random.Generator, reps: int = 100_000) -> tuple[float, float]:
    models = wide["ModelID"].drop_duplicates().to_numpy()
    grouped = {model: group["diff"].to_numpy() for model, group in wide.groupby("ModelID", sort=False)}
    boot = np.empty(reps)
    for i in range(reps):
        sampled = rng.choice(models, size=len(models), replace=True)
        boot[i] = np.concatenate([grouped[model] for model in sampled]).mean()
    lo, hi = np.quantile(boot, [0.025, 0.975])
    return float(lo), float(hi)


def build_results() -> pd.DataFrame:
    market, _cid2ticker, _member = load_market()
    rows = []
    missing = []
    for input_set, input_label in INPUT_SETS.items():
        for model_id, cfg in MODEL_CONFIG.items():
            for mode in MODES:
                panel = load_panel(input_set, mode, model_id)
                if panel is None or panel.empty:
                    missing.append(f"{input_set}:{mode}:{model_id}")
                    continue
                metrics = evaluate_panel(panel, market, cost_bps=40.0)
                metrics.update(
                    {
                        "InputSet": input_set,
                        "InputLabel": input_label,
                        "ModelID": model_id,
                        "Model": cfg["display"],
                        "Vendor": cfg["vendor"],
                        "Mode": mode,
                        "ModeLabel": MODE_LABELS[mode],
                    }
                )
                rows.append(metrics)
    out = pd.DataFrame(rows)
    out.to_csv(OUT_RESULTS, index=False)
    print(f"Missing panels: {missing}")
    return out


def build_tests(df: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(20260521)
    rows = []
    comparisons = [
        ("both", "market_only", "Both - Market"),
        ("both", "fundamentals_only", "Both - Fund."),
    ]
    metrics = [
        ("LS_CAGR", "LS CAGR", True),
        ("LS_Sharpe", "LS Shp", False),
        ("Mean_IC", "IC", True),
    ]
    for left, right, comparison in comparisons:
        for metric, label, as_percent in metrics:
            wide = paired_table(df, left, right, metric)
            lo, hi = bootstrap_ci(wide, rng)
            rows.append(
                {
                    "Comparison": comparison,
                    "Metric": label,
                    "BothMean": wide[left].mean(),
                    "AltMean": wide[right].mean(),
                    "Delta": wide["diff"].mean(),
                    "CI95Low": lo,
                    "CI95High": hi,
                    "PValue": clustered_sign_flip_pvalue(wide),
                    "NModels": wide["ModelID"].nunique(),
                    "NCells": len(wide),
                    "AsPercent": as_percent,
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(OUT_TESTS, index=False)
    return out


def build_loadings() -> pd.DataFrame:
    market, _cid2ticker, _member = load_market()
    factors = load_cached_ff_factors()
    rows = []
    for model_id, model_short in REPRESENTATIVE.items():
        for input_set, input_label in INPUT_SETS.items():
            panel = load_panel(input_set, "return", model_id)
            if panel is None or panel.empty:
                continue
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
            rows.append(
                {
                    "ModelID": model_id,
                    "Model": model_short,
                    "InputSet": input_set,
                    "Input": input_label,
                    "alpha": ff6["alpha_ann"] * 100,
                    "HML": ff6["beta_HML"],
                    "CMA": ff6["beta_CMA"],
                    "MOM": ff6["beta_MOM"],
                    "p_HML": ff6["p_HML"],
                    "p_CMA": ff6["p_CMA"],
                    "p_MOM": ff6["p_MOM"],
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(OUT_LOADINGS, index=False)
    return out


def write_table(results: pd.DataFrame, tests: pd.DataFrame, loadings: pd.DataFrame) -> None:
    panel_a = (
        results[results["Mode"].eq("return")]
        .groupby(["InputSet", "InputLabel"], sort=False)
        .agg(
            Mean_LS_CAGR=("LS_CAGR", "mean"),
            Mean_LS_Sharpe=("LS_Sharpe", "mean"),
            Mean_IC=("Mean_IC", "mean"),
            N=("LS_CAGR", "count"),
        )
        .reset_index()
    )
    input_order = {"both": 0, "market_only": 1, "fundamentals_only": 2}
    panel_a["Order"] = panel_a["InputSet"].map(input_order)
    panel_a = panel_a.sort_values("Order")

    lines = [
        "\\begin{table}[!t]",
        "\\centering",
        "\\caption{Input ablation diagnostics under conservative model windows. CAGR and IC are reported in percent; Panel C stars use Newey--West HAC SE: $^{*}p<0.10$, $^{**}p<0.05$, $^{***}p<0.01$.}",
        "\\label{tab:input-ablation-main}",
        "\\scriptsize",
        "\\setlength{\\tabcolsep}{3pt}",
        "\\resizebox{\\columnwidth}{!}{",
        "\\begin{tabular}{llrlll}",
        "\\toprule",
        "\\multicolumn{6}{l}{\\textbf{Panel A: Expected-return performance}}  \\\\",
        "\\textbf{Input} & & \\textbf{Mean LS CAGR} & \\textbf{Mean LS Shp} & \\textbf{Mean IC} & \\textbf{N}  \\\\",
        "\\midrule",
    ]
    for _, row in panel_a.iterrows():
        lines.append(
            f"{row['InputLabel']} & & {fmt_pct(row['Mean_LS_CAGR'])} & "
            f"{fmt_num(row['Mean_LS_Sharpe'])} & {fmt_pct(row['Mean_IC'])} & {int(row['N'])} \\\\"
        )

    lines.extend(
        [
            "\\midrule",
            "\\multicolumn{6}{l}{\\textbf{Panel B: Paired tests}}  \\\\",
            "\\textbf{Comparison} & \\textbf{Metric} & \\textbf{Both} & \\textbf{Ablated} & \\textbf{$\\Delta$} & \\textbf{$p$}  \\\\",
            "\\midrule",
        ]
    )
    for _, row in tests.iterrows():
        as_percent = bool(row["AsPercent"])
        fmt = fmt_pct if as_percent else fmt_num
        lines.append(
            f"{row['Comparison']} & {row['Metric']} & {fmt(row['BothMean'])} & "
            f"{fmt(row['AltMean'])} & {fmt(row['Delta'])} & {fmt_p(row['PValue'])} \\\\"
        )

    lines.extend(
        [
            "\\midrule",
            "\\multicolumn{6}{l}{\\textbf{Panel C: Representative FF6 loadings}}  \\\\",
            "\\textbf{Model} & \\textbf{Input} & $\\alpha$ & $\\beta_{\\text{HML}}$ & $\\beta_{\\text{CMA}}$ & $\\beta_{\\text{MOM}}$  \\\\",
            "\\midrule",
        ]
    )
    model_order = {model_id: i for i, model_id in enumerate(REPRESENTATIVE)}
    loadings["ModelOrder"] = loadings["ModelID"].map(model_order)
    loadings["InputOrder"] = loadings["InputSet"].map(input_order)
    for _, row in loadings.sort_values(["ModelOrder", "InputOrder"]).iterrows():
        lines.append(
            f"{row['Model']} & {row['Input']} & {fmt_num(row['alpha'], 1)} & "
            f"{fmt_coef(row['HML'], row['p_HML'])} & "
            f"{fmt_coef(row['CMA'], row['p_CMA'])} & "
            f"{fmt_coef(row['MOM'], row['p_MOM'])} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", "}", "\\end{table}", ""])

    tex = "\n".join(lines)
    OUT_TEX.write_text(tex, encoding="utf-8")
    PAPER_TABLE_DIR.mkdir(parents=True, exist_ok=True)
    APPENDIX_TABLE_DIR.mkdir(parents=True, exist_ok=True)
    (PAPER_TABLE_DIR / "conservative_input_ablation.tex").write_text(tex, encoding="utf-8")
    (APPENDIX_TABLE_DIR / "conservative_input_ablation.tex").write_text(tex, encoding="utf-8")


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    results = build_results()
    tests = build_tests(results)
    loadings = build_loadings()
    write_table(results, tests, loadings)
    print(f"Wrote {OUT_RESULTS}")
    print(f"Wrote {OUT_TESTS}")
    print(f"Wrote {OUT_LOADINGS}")
    print(f"Wrote {OUT_TEX}")


if __name__ == "__main__":
    main()
