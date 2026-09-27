#!/usr/bin/env python
"""
Assemble the MkDocs source tree from repository files.

The site's landing page is README.md, and CONTRIBUTING.md / CHANGELOG.md are
included as pages, so the documentation never drifts from the files people
read on GitHub. Relative links are rewritten to work inside the site.

    python scripts/build_docs.py && mkdocs build --strict
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DOCS = REPO / "docs"

REWRITES = [
    (r"(src=\")results/plots/", r"\1plots/"),  # README images -> copied into the site
    (r"\]\(docs/", "]("),  # docs/methods.md -> methods.md
    (r"\]\(CONTRIBUTING\.md\)", "](contributing.md)"),
    (r"\]\(CHANGELOG\.md\)", "](changelog.md)"),
    (
        r"\]\(LICENSE\)",
        "](https://github.com/SatvikPraveen/Optimal-Demo-Selection-ICL/blob/main/LICENSE)",
    ),
    (
        r"\]\(CITATION\.cff\)",
        "](https://github.com/SatvikPraveen/Optimal-Demo-Selection-ICL/blob/main/CITATION.cff)",
    ),
    (
        r"\]\(((?:configs|scripts|results)/[^)]+)\)",
        r"](https://github.com/SatvikPraveen/Optimal-Demo-Selection-ICL/blob/main/\1)",
    ),
]


def rewrite(text: str) -> str:
    for pattern, repl in REWRITES:
        text = re.sub(pattern, repl, text)
    return text


def main() -> int:
    DOCS.mkdir(exist_ok=True)
    (DOCS / "index.md").write_text(rewrite((REPO / "README.md").read_text()))
    (DOCS / "contributing.md").write_text(rewrite((REPO / "CONTRIBUTING.md").read_text()))
    (DOCS / "changelog.md").write_text(rewrite((REPO / "CHANGELOG.md").read_text()))
    plots = DOCS / "plots"
    if plots.exists():
        shutil.rmtree(plots)
    if (REPO / "results" / "plots").exists():
        shutil.copytree(REPO / "results" / "plots", plots)
    figures = DOCS / "Figures"
    if figures.exists():
        shutil.rmtree(figures)
    shutil.copytree(REPO / "Figures", figures)
    print("docs tree assembled")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
