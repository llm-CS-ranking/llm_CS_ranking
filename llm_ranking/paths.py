from __future__ import annotations

from pathlib import Path


PACKAGE_DIR = Path(__file__).resolve().parent
ROOT = PACKAGE_DIR.parent

DATA_DIR = ROOT / "data"
NDX_DATA_DIR = DATA_DIR / "ndx_rolling_20260107"

RESULT_DIR = ROOT / "result"
ABLATION_DIR = RESULT_DIR / "ablations"
TABLE_DIR = RESULT_DIR / "tables"
COMMON_WINDOW_DIR = RESULT_DIR / "common_window"

PAPER_DIR = ROOT / "paper"
PAPER_TABLE_DIR = PAPER_DIR / "tables"
PAPER_FIGURE_DIR = PAPER_DIR / "figures"

APPENDIX_DIR = ROOT / "paper_appendix"
APPENDIX_TABLE_DIR = APPENDIX_DIR / "tables"
APPENDIX_FIGURE_DIR = APPENDIX_DIR / "figures"
APPENDIX_SECTION_DIR = APPENDIX_DIR / "appendices"

