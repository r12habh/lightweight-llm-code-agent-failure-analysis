#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiment_extension_utils import (
    DATASET_NAME,
    SPLIT,
    load_rows,
    merge_predictions_for,
    print_summary_table,
)


BACKENDS = ["gpt4omini", "qwen7b", "qwen32b"]
MODES = ["direct", "agentless"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge old 50-instance predictions with extension predictions.")
    parser.add_argument("--outputs_dir", default="outputs")
    parser.add_argument("--dataset_name", default=DATASET_NAME)
    parser.add_argument("--split", default=SPLIT)
    args = parser.parse_args()

    rows = load_rows(args.dataset_name, args.split)
    summaries = []
    for backend in BACKENDS:
        for mode in MODES:
            summaries.append(merge_predictions_for(backend, mode, rows, Path(args.outputs_dir)))

    print_summary_table(summaries)
    summary_path = Path(args.outputs_dir) / "merge_predictions_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    print(f"Wrote merge summary to {summary_path}")


if __name__ == "__main__":
    main()
