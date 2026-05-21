#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from experiment_extension_utils import estimate_openai_cost, log_path


OPENAI_INPUT_PER_M = 0.15
OPENAI_OUTPUT_PER_M = 0.60
HF_HOURLY = {
    "qwen7b": 1.50,
    "qwen32b": 1.80,
}

HF_JOB_WALLCLOCK = {
    "qwen7b": {
        "job_id": "6a0d490a3aba298b21d146dc",
        "flavor": "a10g-large",
        "running_seconds": 6171,
        "total_seconds": 6323,
    },
    "qwen32b": {
        "job_id": "6a0d28603aba298b21d14685",
        "flavor": "l40sx1",
        "running_seconds": 7729,
        "total_seconds": 7953,
    },
}

BACKENDS = ["gpt4omini", "qwen7b", "qwen32b"]
MODES = ["direct", "agentless"]


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return {}


def read_logs(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def baseline_costs() -> dict[tuple[str, str], str]:
    out: dict[tuple[str, str], str] = {}
    openai = read_json(Path("outputs/openai_gpt4omini_50_20260507_0028/detailed_summary.json"))
    for mode in MODES:
        usage = (((openai.get("summary") or {}).get(mode) or {}).get("token_usage") or {})
        cost = usage.get("estimated_usd_gpt4o_mini_standard")
        out[("gpt4omini", mode)] = f"${cost:.3f}" if isinstance(cost, (int, float)) else "n/a"

    qwen7 = read_json(Path("outputs/hf_qwen25_coder7b_50_20260508_a10g/final_results.json"))
    qwen7_cost = ((qwen7.get("hf_generation_job") or {}).get("estimated_generation_cost_usd_total_secs"))
    qwen32 = read_json(Path("outputs/hf_qwen25_coder32b_awq_50_20260508_l40s/final_results.json"))
    qwen32_cost = (((qwen32.get("experiment") or {}).get("hf_job") or {}).get("estimated_cost_usd_total_seconds"))
    for mode in MODES:
        out[("qwen7b", mode)] = f"${qwen7_cost:.3f} total job" if isinstance(qwen7_cost, (int, float)) else "n/a"
        out[("qwen32b", mode)] = f"${qwen32_cost:.3f} total job" if isinstance(qwen32_cost, (int, float)) else "n/a"
    return out


def estimate_cost(backend: str, tokens_in: int, tokens_out: int, elapsed_seconds: float) -> float:
    if backend == "gpt4omini":
        return estimate_openai_cost(tokens_in, tokens_out)
    return elapsed_seconds / 3600 * HF_HOURLY[backend]


def table(rows: list[dict[str, Any]]) -> None:
    headers = [
        "backend",
        "mode",
        "instances",
        "tokens_in",
        "tokens_out",
        "elapsed_s",
        "estimated_usd",
        "baseline_50_usd",
    ]
    widths = {key: max(len(key), *(len(str(row.get(key, ""))) for row in rows)) for key in headers}
    print(" | ".join(key.ljust(widths[key]) for key in headers))
    print("-+-".join("-" * widths[key] for key in headers))
    for row in rows:
        print(" | ".join(str(row.get(key, "")).ljust(widths[key]) for key in headers))


def main() -> None:
    parser = argparse.ArgumentParser(description="Estimate extension run costs from sidecar logs.")
    parser.add_argument("--outputs_dir", default="outputs")
    args = parser.parse_args()

    base = Path(args.outputs_dir)
    baseline = baseline_costs()
    rows = []
    for backend in BACKENDS:
        for mode in MODES:
            logs = read_logs(log_path(backend, mode, base))
            tokens_in = sum(int(row.get("tokens_in") or 0) for row in logs)
            tokens_out = sum(int(row.get("tokens_out") or 0) for row in logs)
            per_instance_elapsed = sum(float(row.get("elapsed_seconds") or 0.0) for row in logs)
            job = HF_JOB_WALLCLOCK.get(backend)
            elapsed = float(job["total_seconds"]) if job else per_instance_elapsed
            cost = estimate_cost(backend, tokens_in, tokens_out, elapsed)
            row = {
                "backend": backend,
                "mode": mode,
                "instances": len(logs),
                "tokens_in": tokens_in,
                "tokens_out": tokens_out,
                "elapsed_s": round(elapsed, 1),
                "estimated_usd": f"${cost:.3f}",
                "baseline_50_usd": baseline.get((backend, mode), "n/a"),
            }
            if job:
                row.update(
                    {
                        "job_id": job["job_id"],
                        "flavor": job["flavor"],
                        "running_seconds": job["running_seconds"],
                        "per_instance_elapsed_s": round(per_instance_elapsed, 1),
                        "cost_basis": "job total wall-clock seconds",
                    }
                )
            rows.append(row)

    table(rows)
    out = base / "cost_report_summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"Wrote cost summary to {out}")
    print(
        f"Pricing used: gpt-4o-mini ${OPENAI_INPUT_PER_M}/1M input, "
        f"${OPENAI_OUTPUT_PER_M}/1M output; a10g-large $1.50/hr; l40sx1 $1.80/hr."
    )


if __name__ == "__main__":
    main()
