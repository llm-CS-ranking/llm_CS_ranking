from __future__ import annotations


MODES = ["return", "sharpe", "sortino"]
MODE_LABELS = {"return": "Exp. Return", "sharpe": "Sharpe", "sortino": "Sortino"}

MODEL_CONFIG = {
    "gpt-4o-2024-05-13": {"display": "GPT-4o", "vendor": "OpenAI"},
    "gpt-4o-mini-2024-07-18": {"display": "GPT-4o mini", "vendor": "OpenAI"},
    "gpt-4.1-2025-04-14": {"display": "GPT-4.1", "vendor": "OpenAI"},
    "gpt-4.1-mini-2025-04-14": {"display": "GPT-4.1 mini", "vendor": "OpenAI"},
    "gpt-4.1-nano": {"display": "GPT-4.1 nano", "vendor": "OpenAI"},
    "o4-mini": {"display": "o4-mini", "vendor": "OpenAI"},
    "o3": {"display": "o3", "vendor": "OpenAI"},
    "gemini-2.5-flash": {"display": "Gemini 2.5 Flash", "vendor": "Google"},
    "gemini-2.5-pro": {"display": "Gemini 2.5 Pro", "vendor": "Google"},
    "claude-sonnet-4-20250514": {"display": "Claude Sonnet 4", "vendor": "Anthropic"},
    "claude-opus-4-20250514": {"display": "Claude Opus 4", "vendor": "Anthropic"},
}

RETAINED_MODEL_CONFIG = {
    "gpt-4o-2024-05-13": MODEL_CONFIG["gpt-4o-2024-05-13"],
    "gpt-4o-mini-2024-07-18": MODEL_CONFIG["gpt-4o-mini-2024-07-18"],
    "gpt-4.1-2025-04-14": MODEL_CONFIG["gpt-4.1-2025-04-14"],
    "gpt-4.1-mini-2025-04-14": MODEL_CONFIG["gpt-4.1-mini-2025-04-14"],
    "gemini-2.5-flash": MODEL_CONFIG["gemini-2.5-flash"],
    "gemini-2.5-pro": MODEL_CONFIG["gemini-2.5-pro"],
    "gemini-3-flash-preview": {"display": "Gemini 3 Flash", "vendor": "Google"},
    "gemini-3-pro-preview": {"display": "Gemini 3 Pro", "vendor": "Google"},
}

BASELINE_CONFIG = {
    "baseline_RandomForest_rank_panel.pkl": "Random Forest",
    "baseline_Ridge_rank_panel.pkl": "Ridge",
    "baseline_Mom12-1_rank_panel.pkl": "12--1 Momentum",
    "baseline_QualityROE_rank_panel.pkl": "ROE Quality",
    "baseline_ValueEP_rank_panel.pkl": "Value (E/P)",
    "baseline_Composite6_rank_panel.pkl": "Composite",
}

MODEL_CUTOFFS = {
    "gpt-4o-2024-05-13": "2024-05-13",
    "gpt-4o-mini-2024-07-18": "2024-07-18",
    "gpt-4.1-2025-04-14": "2025-04-14",
    "gpt-4.1-mini-2025-04-14": "2025-04-14",
    "gpt-4.1-nano": "2025-04-14",
    "o4-mini": "2025-04-16",
    "o3": "2025-04-16",
    "gemini-2.5-flash": "2025-06-01",
    "gemini-2.5-pro": "2025-06-01",
    "claude-sonnet-4-20250514": "2025-05-22",
    "claude-opus-4-20250514": "2025-05-22",
}

INPUT_SETS = {
    "both": "Fund. + Market",
    "fundamentals_only": "Fund. only",
    "market_only": "Market only",
}

INPUT_SET_ORDER = ["both", "market_only", "fundamentals_only"]

REPRESENTATIVE_ABLATION_MODELS = {
    "o3": "o3",
    "gemini-2.5-pro": "Gem. 2.5 Pro",
    "claude-opus-4-20250514": "Cl. Opus 4",
}

CHAR_SPECS = {
    "mom_12_1": ("Mom", True),
    "rev_1m": ("Rev", False),
    "vol_60d": ("LowVol", False),
    "ep_ratio": ("E/P", True),
    "roe": ("ROE", True),
    "asset_gr": ("LowInv", False),
    "size": ("SmallSize", False),
}

COST_BPS = 40.0
COST_GRID = [0.0, 20.0, 40.0, 100.0]

