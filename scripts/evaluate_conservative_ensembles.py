from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate_common_window import evaluate_panel, load_market
from utils import rank_ic_by_month


RESULT_DIR = ROOT / "result"
TABLE_DIR = RESULT_DIR / "tables"
PAPER_TABLE_DIR = ROOT / "paper" / "tables"
APPENDIX_TABLE_DIR = ROOT / "paper_appendix" / "tables"
RNG_SEED = 20260613

MODEL_STARTS = {
    "gpt-4.1-2025-04-14": "2025-05",
    "gpt-4.1-mini-2025-04-14": "2025-05",
    "gpt-4.1-nano": "2025-05",
    "o4-mini": "2025-05",
    "o3": "2025-05",
    "gemini-2.5-flash": "2025-07",
    "gemini-2.5-pro": "2025-07",
    "claude-sonnet-4-20250514": "2025-06",
    "claude-opus-4-20250514": "2025-06",
}

ENSEMBLES = {
    "Nine-model consensus": [
        "gpt-4.1-2025-04-14",
        "gpt-4.1-mini-2025-04-14",
        "gpt-4.1-nano",
        "o4-mini",
        "o3",
        "gemini-2.5-flash",
        "gemini-2.5-pro",
        "claude-sonnet-4-20250514",
        "claude-opus-4-20250514",
    ],
    "OpenAI 4.1 consensus": [
        "gpt-4.1-2025-04-14",
        "gpt-4.1-mini-2025-04-14",
        "gpt-4.1-nano",
    ],
    "OpenAI reasoning consensus": [
        "o4-mini",
        "o3",
    ],
    "Gemini 2.5 consensus": [
        "gemini-2.5-flash",
        "gemini-2.5-pro",
    ],
    "Claude 4 consensus": [
        "claude-sonnet-4-20250514",
        "claude-opus-4-20250514",
    ],
    "Low-cost consensus": [
        "gpt-4.1-nano",
        "gpt-4.1-mini-2025-04-14",
        "gemini-2.5-flash",
    ],
}


def load_panel(model_id: str) -> pd.DataFrame:
    path = RESULT_DIR / f"return_{model_id}_result.pkl"
    with path.open("rb") as f:
        panel = pickle.load(f)
    panel.index = panel.index.astype(str)
    panel.columns = panel.columns.astype(str)
    return panel


def ensemble_months(model_ids: list[str], panels: dict[str, pd.DataFrame]) -> list[str]:
    start = max(MODEL_STARTS[model_id] for model_id in model_ids)
    common = set.intersection(*(set(panels[model_id].index) for model_id in model_ids))
    return sorted(month for month in common if month >= start)


def average_rank_panel(model_ids: list[str], panels: dict[str, pd.DataFrame], months: list[str]) -> pd.DataFrame:
    tickers = sorted(set().union(*(set(panels[model_id].columns) for model_id in model_ids)))
    stacked = []
    for model_id in model_ids:
        aligned = panels[model_id].reindex(index=months, columns=tickers).astype(float)
        stacked.append(aligned)
    avg_rank = sum(panel.fillna(0.0) for panel in stacked)
    counts = sum(panel.notna().astype(int) for panel in stacked)
    avg_rank = avg_rank.where(counts > 0) / counts.where(counts > 0)
    ranks = avg_rank.rank(axis=1, method="first", ascending=True)
    return ranks.where(avg_rank.notna()).dropna(axis=0, how="all")


def ic_bootstrap_ci(rank_panel: pd.DataFrame, market: pd.DataFrame) -> tuple[float, float]:
    ic = rank_ic_by_month(rank_panel, market, lag_months=0, method="spearman")["IC"].dropna()
    if ic.empty:
        return np.nan, np.nan
    rng = np.random.default_rng(RNG_SEED)
    draws = rng.choice(ic.to_numpy(), size=(5000, len(ic)), replace=True).mean(axis=1)
    return tuple(np.quantile(draws, [0.025, 0.975]).astype(float))


def write_tex(df: pd.DataFrame, path: Path) -> None:
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Conservative-window average-rank ensemble robustness. Consensus ranks are evaluated as the same net 40\\,bps Q1$-$Q5 portfolio.}",
        "\\label{tab:ensemble-robustness}",
        "\\resizebox{\\columnwidth}{!}{%",
        "\\begin{tabular}{lrrrrrr}",
        "\\toprule",
        "\\textbf{Consensus} & \\textbf{Models} & \\textbf{Start} & \\textbf{Mo.} & \\textbf{LS CAGR} & \\textbf{LS Shp} & \\textbf{IC} \\\\",
        "\\midrule",
    ]
    for _, row in df.iterrows():
        lines.append(
            f"{row['Ranker']} & {int(row['ModelCount'])} & {row['Start']} & {int(row['Months'])} & "
            f"{row['LS_CAGR'] * 100:.1f} & {row['LS_Sharpe']:.2f} & {row['Mean_IC']:.3f} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", "}%", "\\end{table}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    market, _cid2ticker, _member = load_market()
    panels = {model_id: load_panel(model_id) for model_ids in ENSEMBLES.values() for model_id in model_ids}
    rows = []
    for name, model_ids in ENSEMBLES.items():
        months = ensemble_months(model_ids, panels)
        panel = average_rank_panel(model_ids, panels, months)
        metrics = evaluate_panel(panel, market, cost_bps=40.0)
        ci_low, ci_high = ic_bootstrap_ci(panel, market)
        rows.append(
            {
                "Ranker": name,
                "ModelCount": len(model_ids),
                "Start": months[0],
                "End": months[-1],
                "IC_CI95_Low": ci_low,
                "IC_CI95_High": ci_high,
                **metrics,
            }
        )

    out = pd.DataFrame(rows)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    PAPER_TABLE_DIR.mkdir(parents=True, exist_ok=True)
    APPENDIX_TABLE_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(TABLE_DIR / "conservative_ensemble_robustness.csv", index=False)
    write_tex(out, PAPER_TABLE_DIR / "conservative_ensemble_robustness.tex")
    write_tex(out, APPENDIX_TABLE_DIR / "conservative_ensemble_robustness.tex")
    print(out[["Ranker", "Start", "End", "Months", "LS_CAGR", "LS_Sharpe", "Mean_IC"]].to_string(index=False))


if __name__ == "__main__":
    main()
