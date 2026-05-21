#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from datasets import load_dataset
from tqdm import tqdm

from generate_predictions import clone_checkout


PATCH_BIN = shutil.which("gpatch") or "patch"

APPLY_COMMANDS = [
    ["git", "apply", "--verbose"],
    ["git", "apply", "--verbose", "--reject"],
    [PATCH_BIN, "--batch", "--fuzz=5", "-p1", "-i"],
]


def run(cmd: list[str], cwd: Path, timeout: int = 300) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=cwd, text=True, capture_output=True, timeout=timeout)


def reset_repo(repo_dir: Path, base_commit: str) -> None:
    subprocess.run(["git", "reset", "--hard", base_commit], cwd=repo_dir, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.run(["git", "clean", "-xfd"], cwd=repo_dir, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def git_diff(repo_dir: Path) -> str:
    proc = run(["git", "-c", "core.fileMode=false", "diff", "--binary"], repo_dir)
    return proc.stdout.strip() + "\n" if proc.stdout.strip() else ""


def swebench_style_repair(patch: str) -> list[tuple[str, str]]:
    repaired: list[tuple[str, str]] = []
    try:
        from swebench.harness.utils import extract_minimal_patch, repair_patch
    except Exception:
        return repaired

    for name, fn in [("swebench_repair_patch", repair_patch), ("swebench_extract_minimal_patch", extract_minimal_patch)]:
        try:
            candidate = fn(patch)
        except Exception:
            continue
        if candidate and candidate.strip() and candidate.strip() != patch.strip():
            repaired.append((name, candidate.strip() + "\n"))
    return repaired


def patch_candidates(patch: str) -> list[tuple[str, str]]:
    out = [("original", patch.strip() + "\n" if patch.strip() else "")]
    out.extend(swebench_style_repair(patch))
    seen: set[str] = set()
    unique: list[tuple[str, str]] = []
    for name, candidate in out:
        key = candidate.strip()
        if not key or key in seen:
            continue
        seen.add(key)
        unique.append((name, candidate))
    return unique


def apply_and_canonicalize(repo_dir: Path, base_commit: str, patch: str) -> dict[str, Any]:
    errors: list[dict[str, str]] = []
    with tempfile.NamedTemporaryFile("w", suffix=".diff", delete=False, encoding="utf-8") as tmp:
        tmp.write(patch)
        patch_path = Path(tmp.name)
    try:
        reset_repo(repo_dir, base_commit)
        for command in APPLY_COMMANDS:
            cmd = command + [str(patch_path)]
            proc = run(cmd, repo_dir)
            if proc.returncode == 0:
                canonical = git_diff(repo_dir)
                reset_repo(repo_dir, base_commit)
                if canonical:
                    return {
                        "ok": True,
                        "apply_command": " ".join(command),
                        "canonical_patch": canonical,
                        "errors": errors,
                    }
                errors.append({
                    "command": " ".join(command),
                    "stderr": proc.stderr,
                    "stdout": proc.stdout,
                    "note": "command succeeded but produced an empty git diff",
                })
            else:
                errors.append({
                    "command": " ".join(command),
                    "stderr": proc.stderr,
                    "stdout": proc.stdout,
                })
        reset_repo(repo_dir, base_commit)
        return {"ok": False, "errors": errors}
    finally:
        patch_path.unlink(missing_ok=True)


def read_predictions(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        rows.append(json.loads(line))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate and canonicalize SWE-bench prediction patches before Docker evaluation.")
    parser.add_argument("--dataset_name", default="princeton-nlp/SWE-bench_Lite")
    parser.add_argument("--split", default="test")
    parser.add_argument("--input_jsonl", required=True)
    parser.add_argument("--output_jsonl", required=True)
    parser.add_argument("--raw_dir", required=True)
    parser.add_argument("--cache_dir", default=".cache/repos")
    parser.add_argument("--work_dir", default=".work/repair")
    parser.add_argument("--keep_worktrees", action="store_true")
    args = parser.parse_args()

    predictions = read_predictions(Path(args.input_jsonl))
    ids = {row["instance_id"] for row in predictions}
    ds = load_dataset(args.dataset_name, split=args.split)
    rows_by_id = {row["instance_id"]: dict(row) for row in ds if row["instance_id"] in ids}

    out_path = Path(args.output_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    raw_dir = Path(args.raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as out:
        for pred in tqdm(predictions, desc="Repairing patches"):
            instance_id = pred["instance_id"]
            row = rows_by_id[instance_id]
            repo_dir = clone_checkout(
                row["repo"],
                row["base_commit"],
                Path(args.cache_dir),
                Path(args.work_dir) / instance_id,
                force=True,
            )
            status: dict[str, Any] = {
                "instance_id": instance_id,
                "input_patch_chars": len(pred.get("model_patch") or ""),
                "candidate_attempts": [],
                "repaired": False,
                "valid": False,
            }
            output_pred = dict(pred)
            try:
                for candidate_name, candidate in patch_candidates(pred.get("model_patch") or ""):
                    result = apply_and_canonicalize(repo_dir, row["base_commit"], candidate)
                    status["candidate_attempts"].append({
                        "candidate": candidate_name,
                        "ok": bool(result.get("ok")),
                        "apply_command": result.get("apply_command"),
                        "errors": result.get("errors", [])[-2:],
                    })
                    if result.get("ok"):
                        canonical = result["canonical_patch"]
                        output_pred["model_patch"] = canonical
                        status.update({
                            "valid": True,
                            "repaired": canonical.strip() != (pred.get("model_patch") or "").strip(),
                            "selected_candidate": candidate_name,
                            "apply_command": result.get("apply_command"),
                            "output_patch_chars": len(canonical),
                        })
                        break
            finally:
                if not args.keep_worktrees:
                    shutil.rmtree(Path(args.work_dir) / instance_id, ignore_errors=True)

            (raw_dir / f"{instance_id}.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
            out.write(json.dumps(output_pred) + "\n")
            out.flush()

    print(f"Wrote repaired predictions to {out_path}")


if __name__ == "__main__":
    main()
