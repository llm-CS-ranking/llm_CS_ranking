from __future__ import annotations

import pickle
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from conf.model_list import MODEL_KNOWLEDGE_CUTOFF
from evaluate_input_ablation_models import MODEL_CONFIG
from evaluate_retained_paper_models import load_market
from run_baseline_ff6_spanning import hac_ols_ff6, load_cached_ff_factors
from utils import backtest_quintiles_with_universe_ew_capw

RESULT_DIR = ROOT / "result"
TABLE_DIR = ROOT / "result" / "tables"
PAPER_TABLE_DIR = ROOT / "paper" / "tables"
APPENDIX_TABLE_DIR = ROOT / "paper_appendix" / "tables"
OUT_CSV = TABLE_DIR / "conservative_filtered_ff6.csv"
OUT_TEX = TABLE_DIR / "conservative_ff6.tex"


def first_full_month_after(date_str: str) -> str:
    return str((pd.Timestamp(date_str) + pd.offsets.MonthBegin(1)).to_period("M"))


def model_cutoff(model_id: str) -> str:
    if model_id in MODEL_KNOWLEDGE_CUTOFF:
        return MODEL_KNOWLEDGE_CUTOFF[model_id]
    for key in sorted(MODEL_KNOWLEDGE_CUTOFF, key=len, reverse=True):
        if model_id == key or model_id.startswith(f"{key}-"):
            return MODEL_KNOWLEDGE_CUTOFF[key]
    raise KeyError(model_id)


def load_panel(model_id: str) -> pd.DataFrame:
    path = RESULT_DIR / f"return_{model_id}_result.pkl"
    with path.open("rb") as f:
        panel = pickle.load(f)
    panel.index = panel.index.astype(str)
    panel.columns = panel.columns.astype(str)
    start = first_full_month_after(model_cutoff(model_id))
    return panel.loc[panel.index >= start]


def sig_stars(pvalue: float) -> str:
    if pvalue < 0.01:
        return "^{***}"
    if pvalue < 0.05:
        return "^{**}"
    if pvalue < 0.10:
        return "^{*}"
    return ""


def fmt_value(value: float, pvalue: float | None = None) -> str:
    text = f"{value:.2f}"
    if text.startswith("-"):
        text = "$-$" + text[1:]
    if pvalue is not None:
        stars = sig_stars(pvalue)
        if stars:
            text = f"{text}${stars}$"
    return text


def fmt_alpha(value: float, pvalue: float) -> str:
    text = f"{value:.1f}"
    if text.startswith("-"):
        text = "$-$" + text[1:]
    stars = sig_stars(pvalue)
    if stars:
        text = f"{text}${stars}$"
    return text


def build_table() -> pd.DataFrame:
    market, _cid2ticker, _member = load_market()
    factors = load_cached_ff_factors()
    rows = []

    for model_id, cfg in MODEL_CONFIG.items():
        panel = load_panel(model_id)
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
                "ModelID": model_id,
                "Model": cfg["display"],
                "Vendor": cfg["vendor"],
                "alpha": res["alpha_ann"] * 100,
                "t": res["alpha_t"],
                "p": res["alpha_p"],
                "MKT": res["beta_Mkt_RF"],
                "SMB": res["beta_SMB"],
                "HML": res["beta_HML"],
                "RMW": res["beta_RMW"],
                "CMA": res["beta_CMA"],
                "MOM": res["beta_MOM"],
                "p_MKT": res["p_Mkt_RF"],
                "p_SMB": res["p_SMB"],
                "p_HML": res["p_HML"],
                "p_RMW": res["p_RMW"],
                "p_CMA": res["p_CMA"],
                "p_MOM": res["p_MOM"],
                "R2": res["r2_adj"],
                "nobs": res["nobs"],
            }
        )
    out = pd.DataFrame(rows)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False)
    return out


def write_tex(df: pd.DataFrame) -> None:
    lines = [
        "\\begin{table*}[!t]",
        "\\centering",
        "\\caption{FF6 spanning tests for conservative-window Q1$-$Q5 returns. $\\alpha$ is annualized percent; stars use Newey--West HAC SE: $^{*}p<0.10$, $^{**}p<0.05$, $^{***}p<0.01$.}",
        "\\label{tab:ff6}",
        "\\scriptsize",
        "\\setlength{\\tabcolsep}{2.2pt}",
        "\\resizebox{1.5\\columnwidth}{!}{",
        "\\begin{tabular}{l l r llllll r}",
        "\\toprule",
        "\\textbf{Model} & $\\alpha$ (\\%) & $t(\\alpha)$ & $\\beta_{\\text{MKT}}$ & $\\beta_{\\text{SMB}}$ & $\\beta_{\\text{HML}}$ & $\\beta_{\\text{RMW}}$ & $\\beta_{\\text{CMA}}$ & $\\beta_{\\text{MOM}}$ & $R^2_{\\text{adj}}$  \\\\",
        "\\midrule",
    ]

    last_vendor = None
    for _, row in df.iterrows():
        if last_vendor is not None and row["Vendor"] != last_vendor:
            lines.append("\\midrule")
        last_vendor = row["Vendor"]
        lines.append(
            f"{row['Model'].replace('Gemini', 'Gem.').replace('Claude', 'Cl.')} & "
            f"{fmt_alpha(row['alpha'], row['p'])} & {fmt_value(row['t'])} & "
            f"{fmt_value(row['MKT'], row['p_MKT'])} & "
            f"{fmt_value(row['SMB'], row['p_SMB'])} & "
            f"{fmt_value(row['HML'], row['p_HML'])} & "
            f"{fmt_value(row['RMW'], row['p_RMW'])} & "
            f"{fmt_value(row['CMA'], row['p_CMA'])} & "
            f"{fmt_value(row['MOM'], row['p_MOM'])} & "
            f"{fmt_value(row['R2'])} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", "}", "\\end{table*}", ""])
    tex = "\n".join(lines)
    OUT_TEX.write_text(tex, encoding="utf-8")
    PAPER_TABLE_DIR.mkdir(parents=True, exist_ok=True)
    APPENDIX_TABLE_DIR.mkdir(parents=True, exist_ok=True)
    (PAPER_TABLE_DIR / "conservative_ff6.tex").write_text(tex, encoding="utf-8")
    (APPENDIX_TABLE_DIR / "conservative_ff6.tex").write_text(tex, encoding="utf-8")


def main() -> None:
    df = build_table()
    write_tex(df)
    print(f"Wrote {OUT_CSV}")
    print(f"Wrote {OUT_TEX}")


if __name__ == "__main__":
    main()
