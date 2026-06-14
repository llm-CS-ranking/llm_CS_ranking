from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from llm_ranking.paths import APPENDIX_FIGURE_DIR, TABLE_DIR
from llm_ranking.rendering.latex import short_model
from llm_ranking.rendering.tables.factor_tables import representative_ff6_rows


def render_conservative_ff6_heatmap(
    csv_path=None,
    output_path=None,
) -> None:
    csv_path = TABLE_DIR / "conservative_filtered_ff6.csv" if csv_path is None else csv_path
    output_path = APPENDIX_FIGURE_DIR / "fig9_factor_loadings.pdf" if output_path is None else output_path

    df = pd.read_csv(csv_path)
    value_cols = ["MKT", "SMB", "HML", "RMW", "CMA", "MOM"]
    heat = df.set_index("Model")[value_cols].copy()
    labels = value_cols
    values = heat.to_numpy(dtype=float)
    vmax = float(np.nanmax(np.abs(values)))

    fig, ax = plt.subplots(figsize=(8.3, 5.2))
    im = ax.imshow(values, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticks(np.arange(len(heat.index)))
    ax.set_yticklabels([short_model(m) for m in heat.index])
    ax.set_xlabel("Factor")
    ax.set_ylabel("Model")
    ax.set_title("Conservative-window FF6 loadings (expected-return mode, gross)", fontsize=14, fontweight="bold")

    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            val = values[i, j]
            color = "white" if abs(val) > 0.55 * vmax else "#222222"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=color, fontsize=9)

    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.04)
    cbar.set_label("Loading")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def render_input_ablation_ff6_heatmap(
    ff6_df: pd.DataFrame | None = None,
    csv_path=None,
    output_path=None,
) -> None:
    csv_path = TABLE_DIR / "input_ablation_ff6_loadings.csv" if csv_path is None else csv_path
    output_path = APPENDIX_FIGURE_DIR / "fig10_ablation_ff6_loadings.pdf" if output_path is None else output_path
    ff6_df = pd.read_csv(csv_path) if ff6_df is None else ff6_df
    rep = representative_ff6_rows(ff6_df)
    heat = rep.set_index(["ModelShort", "InputLabel"])[
        ["beta_Mkt_RF", "beta_SMB", "beta_HML", "beta_RMW", "beta_CMA", "beta_MOM"]
    ]
    heat.index = [f"{m} / {i}" for m, i in heat.index]
    heat.columns = ["MKT", "SMB", "HML", "RMW", "CMA", "MOM"]
    plt.figure(figsize=(8.5, 4.2))
    sns.heatmap(heat, annot=True, fmt=".2f", cmap="RdBu_r", center=0.0, linewidths=0.5)
    plt.title("FF6 Loadings under Input Ablations (Expected Return)")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()

