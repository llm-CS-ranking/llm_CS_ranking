from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np
import pandas as pd


def short_model(name: str) -> str:
    return str(name).replace("Gemini", "Gem.").replace("Claude", "Cl.")


def latex_minus(text: str) -> str:
    return "$-$" + text[1:] if text.startswith("-") else text


def maybe_bold(text: str, bold: bool = False) -> str:
    return f"\\textbf{{{text}}}" if bold else text


def fmt_pct(value: float | None, bold: bool = False, decimals: int = 1, latex_sign: bool = False) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    text = f"{value * 100:.{decimals}f}"
    if latex_sign:
        text = latex_minus(text)
    return maybe_bold(text, bold)


def fmt_num(value: float | None, bold: bool = False, decimals: int = 2, latex_sign: bool = False) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    text = f"{value:.{decimals}f}"
    if latex_sign:
        text = latex_minus(text)
    return maybe_bold(text, bold)


def fmt_signed(value: float | None, decimals: int = 2) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"{value:+.{decimals}f}"


def sig_stars(p: float | None) -> str:
    if p is None or pd.isna(p):
        return ""
    if p < 0.01:
        return "***"
    if p < 0.05:
        return "**"
    if p < 0.10:
        return "*"
    return ""


def normal_pvalue_from_t(t: float) -> float:
    if not np.isfinite(t):
        return np.nan
    return 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(t) / math.sqrt(2.0))))


def is_close(value: float | None, best: float | None) -> bool:
    return (
        value is not None
        and best is not None
        and not pd.isna(value)
        and not pd.isna(best)
        and bool(np.isclose(value, best, equal_nan=False))
    )


def best_value(values: Iterable[float | None], direction: str = "max") -> float | None:
    clean = [float(v) for v in values if v is not None and not pd.isna(v)]
    if not clean:
        return None
    return min(clean) if direction == "min" else max(clean)


def write_lines(path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")

