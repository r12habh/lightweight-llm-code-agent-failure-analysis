#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from math import comb
from pathlib import Path
from typing import Any


def iter_json_files(path: Path):
    if path.is_file():
        yield path
    else:
        for pat in ["**/*instance*result*.jsonl", "**/*report*.json", "**/*result*.json"]:
            yield from path.glob(pat)


def read_jsonl(path: Path):
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if line:
            try:
                yield json.loads(line)
            except Exception:
                pass


def flatten_records(obj: Any):
    if isinstance(obj, dict):
        if {"completed_ids", "resolved_ids"} <= set(obj):
            resolved = set(obj.get("resolved_ids") or [])
            submitted = set(obj.get("completed_ids") or [])
            submitted.update(obj.get("error_ids") or [])
            submitted.update(obj.get("empty_patch_ids") or [])
            for iid in submitted:
                yield {"instance_id": iid, "resolved": iid in resolved}
        if "instance_id" in obj or "instance" in obj:
            yield obj
        for key, v in obj.items():
            if isinstance(v, dict) and any(
                result_key in v
                for result_key in ["resolved", "success", "solved", "passed", "test_result"]
            ):
                # SWE-bench per-instance reports are often shaped as
                # {"repo__issue": {"resolved": false, ...}}.
                rec = dict(v)
                rec.setdefault("instance_id", key)
                yield rec
            yield from flatten_records(v)
    elif isinstance(obj, list):
        for x in obj:
            yield from flatten_records(x)


def load_results(path: str) -> dict[str, bool]:
    p = Path(path)
    found: dict[str, bool] = {}
    for jf in iter_json_files(p):
        try:
            if jf.suffix == ".jsonl":
                records = list(read_jsonl(jf))
            else:
                records = list(flatten_records(json.loads(jf.read_text(encoding="utf-8", errors="replace"))))
        except Exception:
            continue
        for r in records:
            iid = r.get("instance_id") or r.get("instance") or r.get("id")
            if not iid:
                continue
            val = None
            for key in ["resolved", "success", "solved", "passed", "test_result"]:
                if key in r:
                    val = r[key]
                    break
            if isinstance(val, str):
                val = val.lower() in {"true", "resolved", "passed", "success", "pass"}
            if isinstance(val, (bool, int)):
                found[str(iid)] = bool(val)
    if p.is_dir():
        for marker in [*p.glob("**/run_instance.log"), *p.glob("**/patch.diff")]:
            child = marker.parent
            if child.is_dir():
                # Patch-apply errors may create an instance directory without a
                # report.json. They are still submitted paired failures.
                found.setdefault(child.name, False)
    if not found:
        raise SystemExit(f"Could not find parseable result records under {path}. Inspect SWE-bench output manually.")
    return found


def mcnemar_exact_pvalue(b: int, c: int) -> float:
    # Two-sided exact binomial test with p=0.5 over discordant pairs.
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = 2 * sum(comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, p)


def main() -> None:
    p = argparse.ArgumentParser(description="Compare two paired SWE-bench result directories.")
    p.add_argument("--a", required=True, help="Result directory/file for system A")
    p.add_argument("--b", required=True, help="Result directory/file for system B")
    p.add_argument("--name_a", default="direct")
    p.add_argument("--name_b", default="agentless")
    p.add_argument("--out", default="outputs/comparison.json")
    args = p.parse_args()

    a = load_results(args.a)
    b = load_results(args.b)
    ids = sorted(set(a) & set(b))
    if not ids:
        raise SystemExit("No overlapping instance IDs between result sets.")

    both_fail = sum((not a[i]) and (not b[i]) for i in ids)
    a_only = sum(a[i] and (not b[i]) for i in ids)
    b_only = sum((not a[i]) and b[i] for i in ids)
    both_pass = sum(a[i] and b[i] for i in ids)
    result = {
        "n_paired": len(ids),
        args.name_a + "_resolved": sum(a[i] for i in ids),
        args.name_b + "_resolved": sum(b[i] for i in ids),
        args.name_a + "_rate": sum(a[i] for i in ids) / len(ids),
        args.name_b + "_rate": sum(b[i] for i in ids) / len(ids),
        "paired_table": {
            "both_fail": both_fail,
            args.name_a + "_only": a_only,
            args.name_b + "_only": b_only,
            "both_pass": both_pass,
        },
        "mcnemar_exact_p": mcnemar_exact_pvalue(a_only, b_only),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
