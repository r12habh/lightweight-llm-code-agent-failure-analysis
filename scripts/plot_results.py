#!/usr/bin/env python3
"""Print compact tables from the archived result files.

The manuscript figures are included as PDFs under ``figures/``. This helper is
kept intentionally lightweight so artifact users can sanity-check the key
numbers without rerunning the full Docker harness.
"""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_json(path: str) -> dict:
    return json.loads((ROOT / path).read_text())


def main() -> None:
    comparisons = [
        ("gpt-4o-mini 300", "results/swebench_300/gpt4omini_300_comparison.json"),
        ("Qwen 32B AWQ 300", "results/swebench_300/qwen32b_300_comparison.json"),
        ("Qwen 7B 200", "results/swebench_200/qwen7b_200_comparison.json"),
    ]
    print("backend,direct_resolved,agentless_resolved,n_paired,mcnemar_exact_p")
    for label, relpath in comparisons:
        data = load_json(relpath)
        print(
            f"{label},{data['direct_resolved']},{data['agentless_resolved']},"
            f"{data['n_paired']},{data['mcnemar_exact_p']}"
        )


if __name__ == "__main__":
    main()
