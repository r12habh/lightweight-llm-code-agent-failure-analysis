#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from experiment_extension_utils import EXPECTED_COUNTS, FINAL_PREDICTION_NAMES, final_predictions_path


BACKENDS = ["gpt4omini", "qwen7b", "qwen32b"]
MODES = ["direct", "agentless"]
EXPECTED_FIELDS = {"instance_id", "model_patch", "model_name_or_path"}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        rows.append(json.loads(line))
    return rows


def validate_file(path: Path, expected: int) -> dict[str, Any]:
    rows = read_jsonl(path)
    ids = [str(row.get("instance_id") or "") for row in rows]
    counts = Counter(ids)
    duplicates = sorted(iid for iid, count in counts.items() if iid and count > 1)
    exact_field_errors = [
        str(row.get("instance_id") or f"row_{index}")
        for index, row in enumerate(rows, start=1)
        if set(row) != EXPECTED_FIELDS
    ]
    nonempty = sum(bool(str(row.get("model_patch") or "").strip()) for row in rows)
    has_diff = sum("diff --git" in str(row.get("model_patch") or "") for row in rows)
    malformed = len(rows) - has_diff
    return {
        "file": str(path),
        "expected": expected,
        "total": len(rows),
        "nonempty_patch": nonempty,
        "valid_diff_git_header": has_diff,
        "empty_or_malformed_patch": malformed,
        "duplicate_instance_ids": duplicates,
        "field_shape_errors": exact_field_errors,
        "ok": len(rows) >= expected and not duplicates and not exact_field_errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate final merged SWE-bench prediction files.")
    parser.add_argument("--outputs_dir", default="outputs")
    args = parser.parse_args()

    results = []
    for backend in BACKENDS:
        for mode in MODES:
            expected = EXPECTED_COUNTS[backend]
            path = final_predictions_path(backend, mode, Path(args.outputs_dir))
            results.append(validate_file(path, expected))

    for row in results:
        print(f"\n{row['file']}")
        print(f"  Total instances: {row['total']} (expected >= {row['expected']})")
        print(f"  Instances with non-empty patch: {row['nonempty_patch']}")
        print(f"  Instances with valid diff --git header: {row['valid_diff_git_header']}")
        print(f"  Instances with empty or malformed patch: {row['empty_or_malformed_patch']}")
        print(f"  Duplicate instance_ids: {row['duplicate_instance_ids'] or []}")
        if row["field_shape_errors"]:
            print(f"  Field shape errors: {row['field_shape_errors']}")

    out = Path(args.outputs_dir) / "validate_predictions_summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nWrote validation summary to {out}")

    if not all(row["ok"] for row in results):
        sys.exit(1)


if __name__ == "__main__":
    main()
