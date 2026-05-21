#!/usr/bin/env python3
from __future__ import annotations

import base64
import gzip
import json
import os
import random
import shutil
import subprocess
import tarfile
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

DATASET_NAME = "princeton-nlp/SWE-bench_Lite"
SPLIT = "test"
OUTPUTS_DIR = Path("outputs")

CONTEXT_ARGS = {
    "max_files": 18,
    "max_file_chars": 12000,
    "max_context_chars": 90000,
}

EMPTY_DIFF_PATCH = (
    "diff --git a/.swebench_empty_patch b/.swebench_empty_patch\n"
    "new file mode 100644\n"
    "index 0000000..e69de29\n"
    "--- /dev/null\n"
    "+++ b/.swebench_empty_patch\n"
)

MODEL_NAME_BY_KEY = {
    ("gpt4omini", "direct"): "gpt-4o-mini__direct",
    ("gpt4omini", "agentless"): "gpt-4o-mini__agentless",
    ("qwen7b", "direct"): "qwen2.5-coder-7b-instruct__direct",
    ("qwen7b", "agentless"): "qwen2.5-coder-7b-instruct__agentless",
    ("qwen32b", "direct"): "qwen2.5-coder-32b-instruct-awq__direct",
    ("qwen32b", "agentless"): "qwen2.5-coder-32b-instruct-awq__agentless",
}

OLD_PREDICTIONS = {
    ("gpt4omini", "direct"): Path("outputs/openai_gpt4omini_50_20260507_0028/direct_predictions.jsonl"),
    ("gpt4omini", "agentless"): Path("outputs/openai_gpt4omini_50_20260507_0028/agentless_predictions.jsonl"),
    ("qwen7b", "direct"): Path("outputs/hf_qwen25_coder7b_50_20260508_a10g/direct_predictions.jsonl"),
    ("qwen7b", "agentless"): Path("outputs/hf_qwen25_coder7b_50_20260508_a10g/agentless_predictions.jsonl"),
    ("qwen32b", "direct"): Path("outputs/hf_qwen25_coder32b_awq_50_20260508_l40s/direct_predictions.jsonl"),
    ("qwen32b", "agentless"): Path("outputs/hf_qwen25_coder32b_awq_50_20260508_l40s/agentless_predictions.jsonl"),
}

NEW_PREDICTION_NAMES = {
    ("gpt4omini", "direct"): "new_predictions_gpt4omini_direct.jsonl",
    ("gpt4omini", "agentless"): "new_predictions_gpt4omini_agentless.jsonl",
    ("qwen7b", "direct"): "new_predictions_qwen7b_direct.jsonl",
    ("qwen7b", "agentless"): "new_predictions_qwen7b_agentless.jsonl",
    ("qwen32b", "direct"): "new_predictions_qwen32b_direct.jsonl",
    ("qwen32b", "agentless"): "new_predictions_qwen32b_agentless.jsonl",
}

FINAL_PREDICTION_NAMES = {
    ("gpt4omini", "direct"): "final_predictions_gpt4omini_direct.jsonl",
    ("gpt4omini", "agentless"): "final_predictions_gpt4omini_agentless.jsonl",
    ("qwen7b", "direct"): "final_predictions_qwen7b_direct.jsonl",
    ("qwen7b", "agentless"): "final_predictions_qwen7b_agentless.jsonl",
    ("qwen32b", "direct"): "final_predictions_qwen32b_direct.jsonl",
    ("qwen32b", "agentless"): "final_predictions_qwen32b_agentless.jsonl",
}

EXPECTED_COUNTS = {
    "gpt4omini": 300,
    "qwen32b": 300,
    "qwen7b": 200,
}


def outputs_dir(path: Path | str | None = None) -> Path:
    return Path(path) if path is not None else OUTPUTS_DIR


def new_predictions_path(backend: str, mode: str, base: Path | str | None = None) -> Path:
    return outputs_dir(base) / NEW_PREDICTION_NAMES[(backend, mode)]


def final_predictions_path(backend: str, mode: str, base: Path | str | None = None) -> Path:
    return outputs_dir(base) / FINAL_PREDICTION_NAMES[(backend, mode)]


def log_path(backend: str, mode: str, base: Path | str | None = None) -> Path:
    return outputs_dir(base) / f"log_{backend}_{mode}.jsonl"


def raw_dir(backend: str, mode: str, base: Path | str | None = None) -> Path:
    return outputs_dir(base) / f"raw_{backend}_{mode}"


def load_rows(dataset_name: str = DATASET_NAME, split: str = SPLIT) -> list[dict[str, Any]]:
    from datasets import load_dataset

    return [dict(row) for row in load_dataset(dataset_name, split=split)]


def target_rows(rows: list[dict[str, Any]], backend: str) -> list[dict[str, Any]]:
    if backend == "qwen7b":
        keep = set(sorted(row["instance_id"] for row in rows)[:200])
        selected = [row for row in rows if row["instance_id"] in keep]
    else:
        selected = rows
    return sorted(selected, key=lambda row: row["instance_id"])


def legacy_50_ids(rows: list[dict[str, Any]]) -> set[str]:
    ids = [row["instance_id"] for row in rows]
    random.Random(13).shuffle(ids)
    return set(sorted(ids[:50]))


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            yield json.loads(line)
        except json.JSONDecodeError:
            continue


def normalize_prediction(record: dict[str, Any], backend: str, mode: str) -> dict[str, str]:
    patch = ensure_patch(str(record.get("model_patch") or record.get("patch") or ""))
    return {
        "instance_id": str(record["instance_id"]),
        "model_patch": patch,
        "model_name_or_path": MODEL_NAME_BY_KEY[(backend, mode)],
    }


def prediction_map(path: Path, backend: str, mode: str, target_ids: set[str] | None = None) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    for record in iter_jsonl(path):
        iid = str(record.get("instance_id") or "")
        if not iid or (target_ids is not None and iid not in target_ids):
            continue
        out.setdefault(iid, normalize_prediction(record, backend, mode))
    return out


def existing_ids_for_generation(backend: str, mode: str, rows: list[dict[str, Any]], base: Path | str | None = None) -> set[str]:
    target_ids = {row["instance_id"] for row in target_rows(rows, backend)}
    old = prediction_map(OLD_PREDICTIONS[(backend, mode)], backend, mode, target_ids)
    if not old:
        old = {iid: {} for iid in legacy_50_ids(rows) if iid in target_ids}
    new = prediction_map(new_predictions_path(backend, mode, base), backend, mode, target_ids)
    final = prediction_map(final_predictions_path(backend, mode, base), backend, mode, target_ids)
    return set(old) | set(new) | set(final)


def append_prediction(path: Path, prediction: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(prediction, ensure_ascii=False) + "\n")


def append_log(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def ensure_patch(patch: str) -> str:
    patch = (patch or "").strip()
    if not patch or "diff --git" not in patch:
        return EMPTY_DIFF_PATCH
    return patch + "\n"


def patch_flags(patch: str) -> tuple[bool, bool]:
    raw = patch or ""
    return bool(raw.strip()), "diff --git" in raw


def log_record(
    *,
    instance_id: str,
    mode: str,
    tokens_in: int,
    tokens_out: int,
    elapsed_seconds: float,
    raw_patch: str,
    error: str | None,
) -> dict[str, Any]:
    nonempty, has_header = patch_flags(raw_patch)
    return {
        "instance_id": instance_id,
        "mode": mode,
        "tokens_in": int(tokens_in),
        "tokens_out": int(tokens_out),
        "elapsed_seconds": round(elapsed_seconds, 3),
        "patch_nonempty": nonempty,
        "patch_has_diff_header": has_header,
        "error": error,
    }


def write_raw(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def merge_predictions_for(
    backend: str,
    mode: str,
    rows: list[dict[str, Any]],
    base: Path | str | None = None,
) -> dict[str, int]:
    selected = target_rows(rows, backend)
    target_ids = {row["instance_id"] for row in selected}
    old = prediction_map(OLD_PREDICTIONS[(backend, mode)], backend, mode, target_ids)
    new = prediction_map(new_predictions_path(backend, mode, base), backend, mode, target_ids)

    merged: dict[str, dict[str, str]] = {}
    old_count = 0
    new_count = 0
    for iid in sorted(target_ids):
        if iid in old:
            merged[iid] = old[iid]
            old_count += 1
        elif iid in new:
            merged[iid] = new[iid]
            new_count += 1

    out_path = final_predictions_path(backend, mode, base)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for iid in sorted(merged):
            handle.write(json.dumps(merged[iid], ensure_ascii=False) + "\n")
    return {
        "backend": backend,
        "mode": mode,
        "target": len(target_ids),
        "old": old_count,
        "new": new_count,
        "merged": len(merged),
        "missing": len(target_ids) - len(merged),
    }


def print_summary_table(rows: list[dict[str, Any]]) -> None:
    headers = ["backend", "mode", "target", "old", "new", "merged", "missing"]
    widths = {key: max(len(key), *(len(str(row.get(key, ""))) for row in rows)) for key in headers}
    print(" | ".join(key.ljust(widths[key]) for key in headers))
    print("-+-".join("-" * widths[key] for key in headers))
    for row in rows:
        print(" | ".join(str(row.get(key, "")).ljust(widths[key]) for key in headers))


def top_context_files(repo_dir: Path, problem_statement: str, limit: int = 5) -> list[str]:
    from scripts.generate_predictions import list_files, read_file, score_file, tokenize

    problem_tokens = tokenize(problem_statement)
    scored: list[tuple[float, str]] = []
    for rel in list_files(repo_dir):
        content = read_file(repo_dir, rel, CONTEXT_ARGS["max_file_chars"])
        if not content:
            continue
        scored.append((score_file(rel, content, problem_tokens), rel))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [rel for _, rel in scored[:limit]]


def repo_work_dir(row: dict[str, Any], root: Path = Path("/tmp/swebench_repos")) -> Path:
    repo = str(row["repo"]).replace("/", "__")
    commit = str(row["base_commit"])[:12]
    return root / "worktrees" / f"{repo}__{commit}"


def repo_cache_dir(root: Path = Path("/tmp/swebench_repos")) -> Path:
    return root / "bare"


def openai_usage_tokens(usage: Any) -> tuple[int, int]:
    if hasattr(usage, "model_dump"):
        usage = usage.model_dump()
    usage = usage or {}
    if not isinstance(usage, dict):
        return 0, 0
    tokens_in = usage.get("input_tokens") or usage.get("prompt_tokens") or 0
    tokens_out = usage.get("output_tokens") or usage.get("completion_tokens") or 0
    return int(tokens_in or 0), int(tokens_out or 0)


def estimate_openai_cost(tokens_in: int, tokens_out: int) -> float:
    return (tokens_in / 1_000_000) * 0.15 + (tokens_out / 1_000_000) * 0.60


def create_bundle(base: Path, bundle_name: str, paths: list[Path]) -> Path:
    archive = base / f"{bundle_name}.tar.gz"
    archive.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "w:gz") as tar:
        for path in paths:
            if path.exists():
                tar.add(path, arcname=path.name)
    return archive


def upload_archive(archive: Path) -> str | None:
    curl = shutil.which("curl")
    if not curl:
        print("BUNDLE_UPLOAD_SKIPPED curl_not_found", flush=True)
        return None
    try:
        proc = subprocess.run(
            [
                curl,
                "-fsS",
                "-F",
                "reqtype=fileupload",
                "-F",
                f"fileToUpload=@{archive}",
                "https://catbox.moe/user/api.php",
            ],
            text=True,
            capture_output=True,
            timeout=120,
            check=False,
        )
    except Exception as exc:
        print(f"BUNDLE_UPLOAD_FAILED {exc!r}", flush=True)
        return None
    if proc.returncode == 0 and proc.stdout.strip().startswith("https://"):
        url = proc.stdout.strip()
        print(f"BUNDLE_UPLOAD_URL {url}", flush=True)
        return url
    detail = (proc.stderr or proc.stdout or "").replace("\n", " ")[:500]
    print(f"BUNDLE_UPLOAD_FAILED returncode={proc.returncode} detail={detail}", flush=True)
    if os.environ.get("BUNDLE_PASTE_FALLBACK", "0") == "1":
        payload = base64.b64encode(archive.read_bytes()).decode("ascii")
        try:
            paste_proc = subprocess.run(
                [curl, "-fsS", "--data-binary", "@-", "https://paste.rs"],
                input=payload,
                text=True,
                capture_output=True,
                timeout=120,
                check=False,
            )
        except Exception as exc:
            print(f"BUNDLE_PASTE_FAILED {exc!r}", flush=True)
            return None
        if paste_proc.returncode == 0 and paste_proc.stdout.strip().startswith("https://"):
            paste_url = paste_proc.stdout.strip()
            print(f"BUNDLE_PASTE_URL {paste_url}", flush=True)
            print(f"BUNDLE_PASTE_BASE64_CHARS {len(payload)}", flush=True)
            return paste_url
        paste_detail = (paste_proc.stderr or paste_proc.stdout or "").replace("\n", " ")[:500]
        print(f"BUNDLE_PASTE_FAILED returncode={paste_proc.returncode} detail={paste_detail}", flush=True)
    return None


def upload_bundle(base: Path, bundle_name: str, paths: list[Path]) -> str | None:
    archive = create_bundle(base, bundle_name, paths)
    print(f"BUNDLE_CHECKPOINT {archive.name} bytes={archive.stat().st_size}", flush=True)
    url = upload_archive(archive)
    if url:
        return url
    if os.environ.get("BUNDLE_CHECKPOINT_BASE64", "1") == "1":
        payload = base64.b64encode(archive.read_bytes()).decode("ascii")
        print(f"BEGIN_CHECKPOINT_BUNDLE {archive.name}", flush=True)
        for index in range(0, len(payload), 4000):
            print(payload[index : index + 4000], flush=True)
        print(f"END_CHECKPOINT_BUNDLE {archive.name}", flush=True)
        print(f"CHECKPOINT_BUNDLE_BASE64_CHARS {len(payload)}", flush=True)
    return None


def emit_bundle(base: Path, bundle_name: str, paths: list[Path]) -> None:
    archive = create_bundle(base, bundle_name, paths)
    payload = base64.b64encode(archive.read_bytes()).decode("ascii")
    print(f"BEGIN_BUNDLE {archive.name}", flush=True)
    for index in range(0, len(payload), 4000):
        print(payload[index : index + 4000], flush=True)
    print(f"END_BUNDLE {archive.name}", flush=True)
    print(f"BUNDLE_BASE64_CHARS {len(payload)}", flush=True)
    print(f"BUNDLE_BYTES {archive.stat().st_size}", flush=True)
    upload_archive(archive)


def gzip_jsonl(path: Path) -> Path:
    out = path.with_suffix(path.suffix + ".gz")
    with path.open("rb") as src, gzip.open(out, "wb") as dst:
        dst.write(src.read())
    return out


def now() -> float:
    return time.perf_counter()
