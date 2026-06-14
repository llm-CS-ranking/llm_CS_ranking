from __future__ import annotations

from llm_ranking.cli.build_results import build_input_ablation
from llm_ranking.cli.render_tables import render_table10


def main() -> None:
    build_input_ablation()
    render_table10()
    print("Wrote input-ablation appendix CSV and LaTeX artifacts.")


if __name__ == "__main__":
    main()

