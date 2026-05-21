#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


def read_predictions(path: Path) -> tuple[list[str], str, set[str]]:
    ids: list[str] = []
    model_name = ""
    empty_patch_ids: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        instance_id = row["instance_id"]
        ids.append(instance_id)
        model_name = model_name or row["model_name_or_path"].replace("/", "__")
        if not (row.get("model_patch") or "").strip():
            empty_patch_ids.add(instance_id)
    return sorted(set(ids)), model_name, empty_patch_ids


def report_exists(run_id: str, model_name: str, instance_id: str) -> bool:
    return (
        Path("logs")
        / "run_evaluation"
        / run_id
        / model_name
        / instance_id
        / "report.json"
    ).exists()


def expected_swebench_image(instance_id: str) -> str:
    repo = instance_id.split("__", 1)[0]
    image_id = instance_id.replace("__", "_1776_")
    return f"swebench/sweb.eval.x86_64.{repo}_1776_{image_id.split('_1776_', 1)[1]}:latest"


def epoch_ghcr_image(instance_id: str) -> str:
    return f"ghcr.io/epoch-research/swe-bench.eval.x86_64.{instance_id}:latest"


def ensure_epoch_image(instance_id: str) -> None:
    dst = expected_swebench_image(instance_id)
    inspect = subprocess.run(
        ["docker", "image", "inspect", dst],
        text=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if inspect.returncode == 0:
        return

    src = epoch_ghcr_image(instance_id)
    run_with_retries(["docker", "pull", src], attempts=5, initial_delay=10)
    run(["docker", "tag", src, dst])


def run(cmd: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    print("+ " + " ".join(cmd), flush=True)
    return subprocess.run(cmd, text=True, check=check)


def run_with_retries(
    cmd: list[str], *, attempts: int, initial_delay: int
) -> subprocess.CompletedProcess[str]:
    delay = initial_delay
    last_error: subprocess.CalledProcessError | None = None
    for attempt in range(1, attempts + 1):
        try:
            return run(cmd)
        except subprocess.CalledProcessError as exc:
            last_error = exc
            if attempt == attempts:
                break
            print(
                f"Command failed on attempt {attempt}/{attempts}; retrying in {delay}s.",
                flush=True,
            )
            time.sleep(delay)
            delay *= 2
    assert last_error is not None
    raise last_error


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run SWE-bench evaluation in resumable chunks and prune Docker images between chunks."
    )
    parser.add_argument("--predictions_path", required=True, type=Path)
    parser.add_argument("--run_id", required=True)
    parser.add_argument("--dataset_name", default="princeton-nlp/SWE-bench_Lite")
    parser.add_argument("--split", default="test")
    parser.add_argument("--chunk_size", type=int, default=10)
    parser.add_argument("--max_workers", type=int, default=1)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--no_prune", action="store_true")
    parser.add_argument(
        "--prepull_epoch_ghcr",
        action="store_true",
        help="Pull public Epoch GHCR SWE-bench images and tag them to the harness namespace before evaluation.",
    )
    args = parser.parse_args()

    instance_ids, model_name, empty_patch_ids = read_predictions(args.predictions_path)
    if args.limit:
        instance_ids = instance_ids[: args.limit]
    if args.start_index:
        instance_ids = instance_ids[args.start_index :]
    if not instance_ids:
        raise SystemExit("No prediction rows found.")

    total = len(instance_ids)
    print(
        f"Evaluating {total} instances for {model_name} in chunks of {args.chunk_size}.",
        flush=True,
    )

    for offset in range(0, total, args.chunk_size):
        chunk = instance_ids[offset : offset + args.chunk_size]
        done = [
            iid
            for iid in chunk
            if iid in empty_patch_ids or report_exists(args.run_id, model_name, iid)
        ]
        if len(done) == len(chunk):
            print(
                f"[chunk {offset // args.chunk_size + 1}] {len(chunk)} already complete; skipping.",
                flush=True,
            )
            continue

        pending = [iid for iid in chunk if iid not in done]
        if args.prepull_epoch_ghcr:
            for instance_id in pending:
                ensure_epoch_image(instance_id)

        print(
            f"[chunk {offset // args.chunk_size + 1}] running {len(pending)}/{len(chunk)} pending "
            f"({chunk[0]} .. {chunk[-1]})",
            flush=True,
        )
        cmd = [
            sys.executable,
            "-m",
            "swebench.harness.run_evaluation",
            "--dataset_name",
            args.dataset_name,
            "--split",
            args.split,
            "--predictions_path",
            str(args.predictions_path),
            "--max_workers",
            str(args.max_workers),
            "--timeout",
            str(args.timeout),
            "--run_id",
            args.run_id,
            "--cache_level",
            "none",
            "--clean",
            "True",
            "--instance_ids",
            *pending,
        ]
        run(cmd)
        if not args.no_prune:
            run(["docker", "system", "prune", "-af"], check=False)
            run(["docker", "system", "df"], check=False)
            run(["df", "-h", "."], check=False)

    print("Rewriting aggregate SWE-bench report from completed instance reports.", flush=True)
    run(
        [
            sys.executable,
            "-m",
            "swebench.harness.run_evaluation",
            "--dataset_name",
            args.dataset_name,
            "--split",
            args.split,
            "--predictions_path",
            str(args.predictions_path),
            "--max_workers",
            str(args.max_workers),
            "--run_id",
            args.run_id,
            "--rewrite_reports",
            "True",
        ]
    )


if __name__ == "__main__":
    main()
