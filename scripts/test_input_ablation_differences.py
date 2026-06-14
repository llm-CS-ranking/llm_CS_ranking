from __future__ import annotations

from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TABLE_DIR = ROOT / "result" / "tables"
PAPER_APPENDIX_TABLE_DIR = ROOT / "paper_appendix" / "tables"
INPUT_CSV = TABLE_DIR / "input_ablation_results.csv"
OUT_CSV = TABLE_DIR / "input_ablation_hypothesis_tests.csv"
OUT_TEX = TABLE_DIR / "input_ablation_hypothesis_tests.tex"
OUT_APPENDIX_TEX = PAPER_APPENDIX_TABLE_DIR / "input_ablation_hypothesis_tests.tex"

COMPARISONS = [
    ("both", "market_only", "Both - Market only"),
    ("both", "fundamentals_only", "Both - Fund. only"),
]
METRICS = [
    ("LS_CAGR", "LS CAGR", True),
    ("LS_Sharpe", "LS Sharpe", False),
    ("Mean_IC", "Mean IC", True),
]
BOOTSTRAP_REPS = 100_000
RNG_SEED = 20260521


def fmt_p(value: float) -> str:
    if value < 0.001:
        return "$<0.001$"
    return f"{value:.3f}"


def fmt_value(value: float, as_percent: bool) -> str:
    if as_percent:
        return f"{value * 100:.1f}"
    return f"{value:.2f}"


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
        flipped = wide["diff"] * wide["ModelID"].map(sign_map)
        stats.append(abs(flipped.mean()))
    stats_arr = np.asarray(stats)
    return float((np.count_nonzero(stats_arr >= observed - 1e-12) + 1) / (len(stats_arr) + 1))


def clustered_bootstrap_ci(wide: pd.DataFrame, rng: np.random.Generator) -> tuple[float, float]:
    models = wide["ModelID"].drop_duplicates().to_numpy()
    grouped = {model: group["diff"].to_numpy() for model, group in wide.groupby("ModelID", sort=False)}
    boot = np.empty(BOOTSTRAP_REPS)
    for i in range(BOOTSTRAP_REPS):
        sampled_models = rng.choice(models, size=len(models), replace=True)
        boot[i] = np.concatenate([grouped[model] for model in sampled_models]).mean()
    lo, hi = np.quantile(boot, [0.025, 0.975])
    return float(lo), float(hi)


def main() -> None:
    df = pd.read_csv(INPUT_CSV)
    rng = np.random.default_rng(RNG_SEED)
    rows = []

    for left, right, comparison in COMPARISONS:
        for metric, label, as_percent in METRICS:
            wide = paired_table(df, left, right, metric)
            ci_low, ci_high = clustered_bootstrap_ci(wide, rng)
            rows.append(
                {
                    "Comparison": comparison,
                    "Metric": label,
                    "BothMean": wide[left].mean(),
                    "AltMean": wide[right].mean(),
                    "Delta": wide["diff"].mean(),
                    "CI95Low": ci_low,
                    "CI95High": ci_high,
                    "PValue": clustered_sign_flip_pvalue(wide),
                    "NModels": wide["ModelID"].nunique(),
                    "NCells": len(wide),
                    "AsPercent": as_percent,
                }
            )

    out = pd.DataFrame(rows)
    TABLE_DIR.mkdir(exist_ok=True)
    out.to_csv(OUT_CSV, index=False)

    lines = [
        "\\begin{table}[H]",
        "\\centering",
        "\\caption{Paired input-ablation tests. Differences are full input minus the ablated input; p-values use model-clustered sign-flip tests.}",
        "\\label{tab:input-ablation-tests}",
        "\\small",
        "\\setlength{\\tabcolsep}{4pt}",
        "\\resizebox{\\textwidth}{!}{",
        "\\begin{tabular}{llrrrrr}",
        "\\toprule",
        "\\textbf{Comparison} & \\textbf{Metric} & \\textbf{Both} & \\textbf{Ablated} & \\textbf{$\\Delta$} & \\textbf{95\\% CI} & \\textbf{$p$} \\\\",
        "\\midrule",
    ]
    current_comparison = None
    for _, row in out.iterrows():
        if current_comparison is not None and current_comparison != row["Comparison"]:
            lines.append("\\midrule")
        current_comparison = row["Comparison"]
        as_percent = bool(row["AsPercent"])
        ci = f"[{fmt_value(row['CI95Low'], as_percent)}, {fmt_value(row['CI95High'], as_percent)}]"
        lines.append(
            f"{row['Comparison']} & {row['Metric']} & "
            f"{fmt_value(row['BothMean'], as_percent)} & {fmt_value(row['AltMean'], as_percent)} & "
            f"{fmt_value(row['Delta'], as_percent)} & {ci} & {fmt_p(row['PValue'])} \\\\"
        )
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "}",
            "\\end{table}",
            "",
        ]
    )
    PAPER_APPENDIX_TABLE_DIR.mkdir(parents=True, exist_ok=True)
    tex = "\n".join(lines)
    OUT_TEX.write_text(tex, encoding="utf-8")
    OUT_APPENDIX_TEX.write_text(tex, encoding="utf-8")
    print(f"Wrote {OUT_CSV}")
    print(f"Wrote {OUT_TEX}")
    print(f"Wrote {OUT_APPENDIX_TEX}")


if __name__ == "__main__":
    main()
