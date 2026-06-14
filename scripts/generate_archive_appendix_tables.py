from __future__ import annotations

from llm_ranking.cli.build_results import build_appendix_metrics
from llm_ranking.cli.render_tables import render_appendix_metrics


def main() -> None:
    build_appendix_metrics()
    render_appendix_metrics()
    print("Wrote archive appendix metric CSV and LaTeX artifacts.")


if __name__ == "__main__":
    main()

