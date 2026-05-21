#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from datasets import load_dataset
from openai import OpenAI
from tqdm import tqdm

TEXT_EXTS = {
    ".py", ".pyi", ".toml", ".cfg", ".ini", ".md", ".rst", ".txt",
    ".yaml", ".yml", ".json", ".js", ".ts", ".tsx", ".jsx", ".java",
}


def sh(cmd: list[str], cwd: str | Path | None = None, timeout: int = 1800) -> str:
    proc = subprocess.run(cmd, cwd=cwd, text=True, capture_output=True, timeout=timeout)
    if proc.returncode != 0:
        raise RuntimeError(
            f"Command failed: {' '.join(cmd)}\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )
    return proc.stdout


def safe_repo_dir(repo: str) -> str:
    return repo.replace("/", "__")


def clone_checkout(repo: str, base_commit: str, cache_dir: Path, work_dir: Path, force: bool = False) -> Path:
    """Clone a GitHub repo at base_commit into an instance-specific worktree."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    work_dir.parent.mkdir(parents=True, exist_ok=True)

    bare = cache_dir / f"{safe_repo_dir(repo)}.git"
    url = f"https://github.com/{repo}.git"
    if bare.exists():
        # Local clones from a partial bare cache can fail when checkout needs
        # promised blobs. Keep the cache boring and complete for reliability.
        promisor = subprocess.run(
            ["git", "config", "--get", "remote.origin.promisor"],
            cwd=bare,
            text=True,
            capture_output=True,
        )
        if promisor.returncode == 0 and promisor.stdout.strip().lower() == "true":
            shutil.rmtree(bare)

    if not bare.exists():
        sh(["git", "clone", "--bare", url, str(bare)], timeout=3600)

    try:
        sh(["git", "cat-file", "-e", f"{base_commit}^{{commit}}"], cwd=bare)
    except Exception:
        sh(["git", "fetch", "origin", base_commit], cwd=bare, timeout=3600)

    if work_dir.exists() and force:
        shutil.rmtree(work_dir)
    if not work_dir.exists():
        sh(["git", "clone", str(bare), str(work_dir)], timeout=3600)

    # Ensure commit exists locally.
    try:
        sh(["git", "cat-file", "-e", f"{base_commit}^{{commit}}"], cwd=work_dir)
    except Exception:
        sh(["git", "fetch", "origin", base_commit], cwd=work_dir, timeout=3600)

    sh(["git", "checkout", "--force", base_commit], cwd=work_dir)
    sh(["git", "clean", "-xfd"], cwd=work_dir)
    return work_dir


def list_files(repo_dir: Path) -> list[str]:
    out = sh(["git", "ls-files"], cwd=repo_dir)
    files = []
    for line in out.splitlines():
        path = line.strip()
        if not path:
            continue
        if Path(path).suffix.lower() in TEXT_EXTS:
            files.append(path)
    return files


def read_file(repo_dir: Path, rel: str, max_chars: int) -> str:
    path = repo_dir / rel
    try:
        data = path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""
    if len(data) > max_chars:
        return data[:max_chars] + "\n# ... [TRUNCATED]\n"
    return data


def tokenize(text: str) -> set[str]:
    toks = re.findall(r"[A-Za-z_][A-Za-z0-9_]{2,}", text.lower())
    stop = {"the", "and", "for", "with", "that", "this", "from", "when", "into", "should", "would", "could"}
    return {t for t in toks if t not in stop}


def score_file(path: str, content: str, problem_tokens: set[str]) -> float:
    path_tokens = tokenize(path.replace("/", " ").replace("_", " "))
    content_tokens = tokenize(content[:20000])
    return 4.0 * len(problem_tokens & path_tokens) + 1.0 * len(problem_tokens & content_tokens)


def build_context(repo_dir: Path, problem_statement: str, *, max_files: int, max_file_chars: int, max_context_chars: int, preferred: list[str] | None = None) -> str:
    files = list_files(repo_dir)
    problem_tokens = tokenize(problem_statement)

    scored: list[tuple[float, str, str]] = []
    preferred_set = set(preferred or [])
    for rel in files:
        content = read_file(repo_dir, rel, max_file_chars)
        if not content:
            continue
        score = score_file(rel, content, problem_tokens)
        if rel in preferred_set:
            score += 1000.0
        scored.append((score, rel, content))

    scored.sort(key=lambda x: (-x[0], x[1]))
    chunks = []
    total = 0
    for score, rel, content in scored[: max_files * 3]:
        chunk = f"\n--- FILE: {rel} ---\n{content}\n"
        if total + len(chunk) > max_context_chars:
            break
        chunks.append(chunk)
        total += len(chunk)
        if len(chunks) >= max_files:
            break
    return "".join(chunks)


def load_prompt(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def retry_sleep_seconds(error: Exception, attempt: int) -> float:
    text = str(error)
    match = re.search(r"try again in ([0-9.]+)s", text, flags=re.IGNORECASE)
    if match:
        return min(float(match.group(1)) + 2.0, 180.0)
    return min(2 ** attempt, 120.0)


def is_non_retryable_openai_error(error: Exception) -> bool:
    text = str(error).lower()
    return (
        "does not have access to model" in text
        or "model_not_found" in text
        or "invalid api key" in text
    )


def call_openai(client: OpenAI, *, model: str, instructions: str, user_input: str, max_output_tokens: int, temperature: float | None, reasoning_effort: str | None, retries: int = 8) -> tuple[str, dict[str, Any]]:
    kwargs: dict[str, Any] = {
        "model": model,
        "instructions": instructions,
        "input": user_input,
        "max_output_tokens": max_output_tokens,
        "store": False,
    }
    if temperature is not None:
        kwargs["temperature"] = temperature
    if reasoning_effort:
        kwargs["reasoning"] = {"effort": reasoning_effort}

    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            resp = client.responses.create(**kwargs)
            text = getattr(resp, "output_text", None)
            if not text:
                # Fallback for older SDK response shapes.
                text_parts = []
                for item in getattr(resp, "output", []) or []:
                    for c in getattr(item, "content", []) or []:
                        if getattr(c, "type", "") in {"output_text", "text"}:
                            text_parts.append(getattr(c, "text", ""))
                text = "\n".join(text_parts)
            usage = getattr(resp, "usage", None)
            usage_dict = usage.model_dump() if hasattr(usage, "model_dump") else (usage or {})
            return text or "", usage_dict
        except Exception as e:
            last_err = e
            if is_non_retryable_openai_error(e):
                raise
            sleep = retry_sleep_seconds(e, attempt)
            print(f"OpenAI call failed on attempt {attempt+1}/{retries}: {e}; sleeping {sleep}s", file=sys.stderr)
            time.sleep(sleep)
    raise RuntimeError(f"OpenAI call failed after retries: {last_err}")


def extract_patch(text: str) -> str:
    fenced = re.search(r"```(?:diff|patch)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        return fenced.group(1).strip() + "\n"
    idx = text.find("diff --git")
    if idx >= 0:
        return text[idx:].strip() + "\n"
    # Last resort: return raw text. SWE-bench will mark invalid patches as failures.
    return text.strip() + "\n" if text.strip() else ""


def parse_localized_files(text: str) -> list[str]:
    try:
        data = json.loads(text)
        files = data.get("files", [])
        out = []
        for item in files:
            if isinstance(item, dict) and item.get("path"):
                out.append(str(item["path"]))
        return out[:5]
    except Exception:
        # Fallback: scan for paths ending in common extensions.
        return re.findall(r"[\w./-]+\.(?:py|pyi|js|ts|java)", text)[:5]


def build_user_input(row: dict[str, Any], context: str, extra: str = "") -> str:
    return f"""INSTANCE_ID: {row['instance_id']}
REPOSITORY: {row.get('repo')}
BASE_COMMIT: {row.get('base_commit')}

PROBLEM_STATEMENT:
{row.get('problem_statement','')}

{extra}

REPOSITORY_CONTEXT:
{context}
"""


def generate_for_row(args: argparse.Namespace, client: OpenAI, row: dict[str, Any]) -> dict[str, Any]:
    instance_id = row["instance_id"]
    repo = row["repo"]
    base_commit = row["base_commit"]
    repo_dir = clone_checkout(repo, base_commit, Path(args.cache_dir), Path(args.work_dir) / instance_id, force=args.force_checkout)

    if args.mode == "direct":
        context = build_context(repo_dir, row["problem_statement"], max_files=args.max_files, max_file_chars=args.max_file_chars, max_context_chars=args.max_context_chars)
        instructions = load_prompt(args.direct_prompt)
        user_input = build_user_input(row, context)
        raw, usage = call_openai(
            client, model=args.model, instructions=instructions, user_input=user_input,
            max_output_tokens=args.max_output_tokens, temperature=args.temperature,
            reasoning_effort=args.reasoning_effort,
        )
        patch = extract_patch(raw)
        return {"instance_id": instance_id, "raw": raw, "patch": patch, "usage": usage, "localized_files": []}

    if args.mode == "agentless":
        loc_context = build_context(repo_dir, row["problem_statement"], max_files=args.max_files, max_file_chars=args.max_file_chars, max_context_chars=args.max_context_chars)
        loc_raw, loc_usage = call_openai(
            client, model=args.model, instructions=load_prompt(args.localize_prompt),
            user_input=build_user_input(row, loc_context), max_output_tokens=1500,
            temperature=args.temperature, reasoning_effort=args.reasoning_effort,
        )
        localized = parse_localized_files(loc_raw)
        patch_context = build_context(repo_dir, row["problem_statement"], max_files=args.max_files, max_file_chars=args.max_file_chars, max_context_chars=args.max_context_chars, preferred=localized)
        extra = "LOCALIZED_FILES:\n" + "\n".join(f"- {f}" for f in localized)
        patch_raw, patch_usage = call_openai(
            client, model=args.model, instructions=load_prompt(args.patch_prompt),
            user_input=build_user_input(row, patch_context, extra=extra),
            max_output_tokens=args.max_output_tokens, temperature=args.temperature,
            reasoning_effort=args.reasoning_effort,
        )
        patch = extract_patch(patch_raw)
        return {
            "instance_id": instance_id,
            "raw": patch_raw,
            "patch": patch,
            "usage": {"localize": loc_usage, "patch": patch_usage},
            "localized_files": localized,
            "localization_raw": loc_raw,
        }

    raise ValueError(f"Unknown mode: {args.mode}")


def main() -> None:
    p = argparse.ArgumentParser(description="Generate SWE-bench predictions with OpenAI Responses API.")
    p.add_argument("--dataset_name", default="princeton-nlp/SWE-bench_Lite")
    p.add_argument("--split", default="test")
    p.add_argument("--instance_ids_file", required=True)
    p.add_argument("--mode", choices=["direct", "agentless"], required=True)
    p.add_argument("--model", default=os.environ.get("OPENAI_MODEL", "gpt-4.1"))
    p.add_argument("--output_jsonl", required=True)
    p.add_argument("--raw_dir", default="outputs/raw")
    p.add_argument("--cache_dir", default=".cache/repos")
    p.add_argument("--work_dir", default=".work/repos")
    p.add_argument("--max_files", type=int, default=18)
    p.add_argument("--max_file_chars", type=int, default=12000)
    p.add_argument("--max_context_chars", type=int, default=90000)
    p.add_argument("--max_output_tokens", type=int, default=12000)
    p.add_argument("--temperature", type=float, default=None)
    p.add_argument("--reasoning_effort", default=None, help="Optional: low, medium, or high for reasoning models that support it.")
    p.add_argument("--direct_prompt", default="prompts/direct.txt")
    p.add_argument("--localize_prompt", default="prompts/agentless_localize.txt")
    p.add_argument("--patch_prompt", default="prompts/agentless_patch.txt")
    p.add_argument("--force_checkout", action="store_true")
    p.add_argument("--keep_worktrees", action="store_true", help="Keep per-instance repository worktrees after generation.")
    p.add_argument("--resume", action="store_true", help="Append only missing instance IDs if the output JSONL already exists.")
    args = p.parse_args()

    ids = {x.strip() for x in Path(args.instance_ids_file).read_text().splitlines() if x.strip()}
    ds = load_dataset(args.dataset_name, split=args.split)
    rows = [dict(row) for row in ds if row["instance_id"] in ids]
    rows.sort(key=lambda r: r["instance_id"])
    if not rows:
        raise SystemExit("No matching instances found.")

    out_path = Path(args.output_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    raw_dir = Path(args.raw_dir) / args.mode
    raw_dir.mkdir(parents=True, exist_ok=True)

    client = OpenAI()
    model_name_for_swebench = f"{args.model}-{args.mode}"

    existing_ids: set[str] = set()
    if args.resume and out_path.exists():
        for line in out_path.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                rec = json.loads(line)
            except Exception:
                continue
            if rec.get("instance_id"):
                existing_ids.add(str(rec["instance_id"]))

    mode = "a" if args.resume else "w"
    with out_path.open(mode, encoding="utf-8") as pred_f:
        for row in tqdm(rows, desc=f"Generating {args.mode}"):
            if row["instance_id"] in existing_ids:
                continue
            try:
                result = generate_for_row(args, client, row)
                raw_record = {k: v for k, v in result.items() if k != "patch"}
                (raw_dir / f"{row['instance_id']}.json").write_text(json.dumps(raw_record, indent=2), encoding="utf-8")
                pred = {
                    "instance_id": row["instance_id"],
                    "model_name_or_path": model_name_for_swebench,
                    "model_patch": result["patch"],
                }
            except Exception as e:
                if is_non_retryable_openai_error(e):
                    raise
                err = {"instance_id": row["instance_id"], "error": repr(e)}
                (raw_dir / f"{row['instance_id']}.error.json").write_text(json.dumps(err, indent=2), encoding="utf-8")
                pred = {"instance_id": row["instance_id"], "model_name_or_path": model_name_for_swebench, "model_patch": ""}
            finally:
                if not args.keep_worktrees:
                    shutil.rmtree(Path(args.work_dir) / row["instance_id"], ignore_errors=True)
            pred_f.write(json.dumps(pred) + "\n")
            pred_f.flush()

    print(f"Wrote predictions to {out_path}")


if __name__ == "__main__":
    main()
