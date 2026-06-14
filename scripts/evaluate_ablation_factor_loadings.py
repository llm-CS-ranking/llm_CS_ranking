from __future__ import annotations

from llm_ranking.cli.build_results import build_factor_tests
from llm_ranking.cli.render_figures import render_fig10
from llm_ranking.cli.render_tables import render_factor_tables


def main() -> None:
    build_factor_tests()
    render_factor_tables()
    render_fig10()
    print("Wrote input-ablation factor CSV, LaTeX, and figure artifacts.")


if __name__ == "__main__":
    main()

