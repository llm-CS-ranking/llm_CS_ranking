from __future__ import annotations

from llm_ranking.cli.render_tables import render_conservative


def main() -> None:
    render_conservative()
    print("Wrote conservative long-short LaTeX artifacts.")


if __name__ == "__main__":
    main()

