#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any
from collections import Counter

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from experiment_extension_utils import (
    CONTEXT_ARGS,
    DATASET_NAME,
    SPLIT,
    MODEL_NAME_BY_KEY,
    append_log,
    append_prediction,
    emit_bundle,
    ensure_patch,
    existing_ids_for_generation,
    final_predictions_path,
    load_rows,
    log_path,
    log_record,
    merge_predictions_for,
    new_predictions_path,
    now,
    raw_dir,
    repo_cache_dir,
    repo_work_dir,
    target_rows,
    top_context_files,
    upload_bundle,
    write_raw,
)
from scripts.generate_predictions import build_context, build_user_input, clone_checkout, extract_patch, load_prompt


PATCH_PARAMS = SamplingParams(temperature=0.0, max_tokens=1600)
LOCALIZE_PARAMS = SamplingParams(temperature=0.0, max_tokens=700)
MAX_MODEL_LEN = 32768
BATCH_SIZE = int(os.environ.get("QWEN_BATCH_SIZE", "4"))
CHECKPOINT_EVERY_BATCHES = int(os.environ.get("QWEN_CHECKPOINT_EVERY_BATCHES", "4"))


def output_base() -> Path:
    return Path(os.environ.get("OUTPUTS_DIR", "/outputs"))


def token_count(tokenizer: AutoTokenizer, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


def chat_prompt(tokenizer: AutoTokenizer, system: str, user: str) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        tokenize=False,
        add_generation_prompt=True,
    )


def generate_text(llm: LLM, prompt: str, params: SamplingParams) -> str:
    return llm.generate([prompt], params)[0].outputs[0].text


def generate_texts_resilient(llm: LLM, prompts: list[str], params: SamplingParams) -> list[tuple[str, str | None]]:
    if not prompts:
        return []
    try:
        outputs = llm.generate(prompts, params)
        return [(item.outputs[0].text, None) for item in outputs]
    except Exception as batch_exc:
        maybe_clear_cuda(batch_exc)
        results = []
        for prompt in prompts:
            try:
                results.append((generate_text(llm, prompt, params), None))
            except Exception as exc:
                maybe_clear_cuda(exc)
                results.append(("", repr(exc)))
        return results


def require_git() -> None:
    if shutil.which("git"):
        version = subprocess.run(["git", "--version"], text=True, capture_output=True, check=False)
        print((version.stdout or "git available").strip(), flush=True)
        return
    raise RuntimeError("git is required for SWE-bench repository checkout but is not installed in this container")


def parse_localized_files_strict(text: str) -> list[str]:
    data = json.loads(text)
    files = data.get("files", [])
    out = []
    for item in files:
        if isinstance(item, dict) and item.get("path"):
            out.append(str(item["path"]))
    return out[:5]


def maybe_clear_cuda(exc: Exception) -> None:
    text = repr(exc).lower()
    if "out of memory" not in text and "cuda" not in text and "oom" not in text:
        return
    try:
        import torch

        torch.cuda.empty_cache()
    except Exception:
        pass


def run_direct(
    *,
    backend: str,
    row: dict[str, Any],
    repo_dir: Path,
    tokenizer: AutoTokenizer,
    llm: LLM,
) -> tuple[str, int, int, dict[str, Any]]:
    context = build_context(repo_dir, row["problem_statement"], **CONTEXT_ARGS)
    prompt = chat_prompt(tokenizer, load_prompt("prompts/direct.txt"), build_user_input(row, context))
    raw = generate_text(llm, prompt, PATCH_PARAMS)
    return (
        extract_patch(raw),
        token_count(tokenizer, prompt),
        token_count(tokenizer, raw),
        {"raw": raw, "context_chars": len(context), "backend": backend},
    )


def run_agentless(
    *,
    backend: str,
    row: dict[str, Any],
    repo_dir: Path,
    tokenizer: AutoTokenizer,
    llm: LLM,
) -> tuple[str, int, int, dict[str, Any], str | None]:
    loc_context = build_context(repo_dir, row["problem_statement"], **CONTEXT_ARGS)
    loc_prompt = chat_prompt(tokenizer, load_prompt("prompts/agentless_localize.txt"), build_user_input(row, loc_context))
    loc_raw = generate_text(llm, loc_prompt, LOCALIZE_PARAMS)
    fallback_error = None
    try:
        localized = parse_localized_files_strict(loc_raw)
        if not localized:
            raise ValueError("localization JSON contained no files")
    except Exception as exc:
        localized = top_context_files(repo_dir, row["problem_statement"], limit=5)
        fallback_error = f"localization_json_parse_failed: {exc!r}; used context-score top files"

    patch_context = build_context(repo_dir, row["problem_statement"], preferred=localized, **CONTEXT_ARGS)
    extra = "LOCALIZED_FILES:\n" + "\n".join(f"- {path}" for path in localized)
    patch_prompt = chat_prompt(
        tokenizer,
        load_prompt("prompts/agentless_patch.txt"),
        build_user_input(row, patch_context, extra=extra),
    )
    patch_raw = generate_text(llm, patch_prompt, PATCH_PARAMS)
    return (
        extract_patch(patch_raw),
        token_count(tokenizer, loc_prompt) + token_count(tokenizer, patch_prompt),
        token_count(tokenizer, loc_raw) + token_count(tokenizer, patch_raw),
        {
            "localization_raw": loc_raw,
            "raw": patch_raw,
            "localized_files": localized,
            "context_chars": len(loc_context) + len(patch_context),
            "backend": backend,
        },
        fallback_error,
    )


def write_failure(backend: str, mode: str, row: dict[str, Any], base: Path, started: float, error: str) -> None:
    iid = row["instance_id"]
    prediction = {
        "instance_id": iid,
        "model_patch": ensure_patch(""),
        "model_name_or_path": MODEL_NAME_BY_KEY[(backend, mode)],
    }
    append_prediction(new_predictions_path(backend, mode, base), prediction)
    write_raw(raw_dir(backend, mode, base) / f"{iid}.error.json", {"instance_id": iid, "mode": mode, "error": error})
    append_log(
        log_path(backend, mode, base),
        log_record(
            instance_id=iid,
            mode=mode,
            tokens_in=0,
            tokens_out=0,
            elapsed_seconds=now() - started,
            raw_patch="",
            error=error,
        ),
    )


def summarize_logs(backend: str, base: Path) -> None:
    for mode in ("direct", "agentless"):
        path = log_path(backend, mode, base)
        if not path.exists():
            print(f"log_summary {backend} {mode}: missing", flush=True)
            continue
        rows = []
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if not line.strip():
                continue
            rows.append(json.loads(line))
        errors = [row.get("error") for row in rows if row.get("error")]
        print(
            f"log_summary {backend} {mode}: rows={len(rows)} errors={len(errors)} "
            f"tokens_in={sum(int(row.get('tokens_in') or 0) for row in rows)} "
            f"tokens_out={sum(int(row.get('tokens_out') or 0) for row in rows)} "
            f"patch_nonempty={sum(bool(row.get('patch_nonempty')) for row in rows)} "
            f"diff_header={sum(bool(row.get('patch_has_diff_header')) for row in rows)}",
            flush=True,
        )
        for error, count in Counter(errors).most_common(5):
            compact = str(error).replace("\n", " | ")[:700]
            print(f"log_error_sample {backend} {mode}: count={count} error={compact}", flush=True)


def bundle_paths_for(backend: str, base: Path, *, include_final: bool = False) -> list[Path]:
    paths = []
    for mode in ("direct", "agentless"):
        paths.extend([new_predictions_path(backend, mode, base), log_path(backend, mode, base)])
        if include_final:
            paths.append(final_predictions_path(backend, mode, base))
    return paths


def checkpoint(backend: str, base: Path, batches_done: int, processed: dict[str, int]) -> None:
    if CHECKPOINT_EVERY_BATCHES <= 0 or batches_done % CHECKPOINT_EVERY_BATCHES != 0:
        return
    name = f"{backend}_checkpoint_b{batches_done}_d{processed['direct']}_a{processed['agentless']}"
    upload_bundle(base, name, bundle_paths_for(backend, base, include_final=False))


def write_success(
    backend: str,
    mode: str,
    row: dict[str, Any],
    base: Path,
    started: float,
    raw_patch: str,
    tokens_in: int,
    tokens_out: int,
    raw_payload: dict[str, Any],
    error: str | None,
) -> None:
    iid = row["instance_id"]
    prediction = {
        "instance_id": iid,
        "model_patch": ensure_patch(raw_patch),
        "model_name_or_path": MODEL_NAME_BY_KEY[(backend, mode)],
    }
    append_prediction(new_predictions_path(backend, mode, base), prediction)
    write_raw(raw_dir(backend, mode, base) / f"{iid}.json", {**raw_payload, "patch": prediction["model_patch"]})
    append_log(
        log_path(backend, mode, base),
        log_record(
            instance_id=iid,
            mode=mode,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            elapsed_seconds=now() - started,
            raw_patch=raw_patch,
            error=error,
        ),
    )


def process_batch(
    *,
    backend: str,
    entries: list[tuple[int, dict[str, Any], bool, bool]],
    base: Path,
    tokenizer: AutoTokenizer,
    llm: LLM,
    remaining: dict[str, set[str]],
    processed: dict[str, int],
) -> None:
    prepared: list[dict[str, Any]] = []
    for index, row, need_direct, need_agentless in entries:
        iid = row["instance_id"]
        started = now()
        try:
            repo_dir = clone_checkout(
                row["repo"],
                row["base_commit"],
                repo_cache_dir(),
                repo_work_dir(row),
                force=False,
            )
            context = build_context(repo_dir, row["problem_statement"], **CONTEXT_ARGS)
            item: dict[str, Any] = {
                "index": index,
                "row": row,
                "repo_dir": repo_dir,
                "context": context,
                "need_direct": need_direct,
                "need_agentless": need_agentless,
                "direct_started": started,
                "agentless_started": started,
            }
            if need_direct:
                direct_prompt = chat_prompt(tokenizer, load_prompt("prompts/direct.txt"), build_user_input(row, context))
                item["direct_prompt"] = direct_prompt
                item["direct_tokens_in"] = token_count(tokenizer, direct_prompt)
            if need_agentless:
                loc_prompt = chat_prompt(tokenizer, load_prompt("prompts/agentless_localize.txt"), build_user_input(row, context))
                item["loc_prompt"] = loc_prompt
                item["loc_tokens_in"] = token_count(tokenizer, loc_prompt)
            prepared.append(item)
        except Exception as exc:
            error = f"clone_or_context_failed: {exc!r}"
            if need_direct:
                write_failure(backend, "direct", row, base, started, error)
                remaining["direct"].add(iid)
                processed["direct"] += 1
            if need_agentless:
                write_failure(backend, "agentless", row, base, started, error)
                remaining["agentless"].add(iid)
                processed["agentless"] += 1
            shutil.rmtree(repo_work_dir(row), ignore_errors=True)

    direct_items = [item for item in prepared if item.get("need_direct")]
    direct_results = generate_texts_resilient(llm, [item["direct_prompt"] for item in direct_items], PATCH_PARAMS)
    for item, (raw, error) in zip(direct_items, direct_results, strict=True):
        row = item["row"]
        iid = row["instance_id"]
        if error:
            write_failure(backend, "direct", row, base, item["direct_started"], error)
        else:
            raw_patch = extract_patch(raw)
            write_success(
                backend,
                "direct",
                row,
                base,
                item["direct_started"],
                raw_patch,
                item["direct_tokens_in"],
                token_count(tokenizer, raw),
                {"raw": raw, "context_chars": len(item["context"]), "backend": backend, "batch_size": len(direct_items)},
                None,
            )
        remaining["direct"].add(iid)
        processed["direct"] += 1

    agent_items = [item for item in prepared if item.get("need_agentless")]
    loc_results = generate_texts_resilient(llm, [item["loc_prompt"] for item in agent_items], LOCALIZE_PARAMS)
    patch_items: list[dict[str, Any]] = []
    for item, (loc_raw, loc_error) in zip(agent_items, loc_results, strict=True):
        row = item["row"]
        iid = row["instance_id"]
        if loc_error:
            write_failure(backend, "agentless", row, base, item["agentless_started"], loc_error)
            remaining["agentless"].add(iid)
            processed["agentless"] += 1
            continue
        fallback_error = None
        try:
            localized = parse_localized_files_strict(loc_raw)
            if not localized:
                raise ValueError("localization JSON contained no files")
        except Exception as exc:
            localized = top_context_files(item["repo_dir"], row["problem_statement"], limit=5)
            fallback_error = f"localization_json_parse_failed: {exc!r}; used context-score top files"
        try:
            patch_context = build_context(item["repo_dir"], row["problem_statement"], preferred=localized, **CONTEXT_ARGS)
            extra = "LOCALIZED_FILES:\n" + "\n".join(f"- {path}" for path in localized)
            patch_prompt = chat_prompt(
                tokenizer,
                load_prompt("prompts/agentless_patch.txt"),
                build_user_input(row, patch_context, extra=extra),
            )
        except Exception as exc:
            write_failure(backend, "agentless", row, base, item["agentless_started"], f"patch_context_failed: {exc!r}")
            remaining["agentless"].add(iid)
            processed["agentless"] += 1
            continue
        item.update(
            {
                "loc_raw": loc_raw,
                "localized_files": localized,
                "fallback_error": fallback_error,
                "patch_context": patch_context,
                "patch_prompt": patch_prompt,
                "patch_tokens_in": token_count(tokenizer, patch_prompt),
                "loc_tokens_out": token_count(tokenizer, loc_raw),
            }
        )
        patch_items.append(item)

    patch_results = generate_texts_resilient(llm, [item["patch_prompt"] for item in patch_items], PATCH_PARAMS)
    for item, (patch_raw, patch_error) in zip(patch_items, patch_results, strict=True):
        row = item["row"]
        iid = row["instance_id"]
        if patch_error:
            write_failure(backend, "agentless", row, base, item["agentless_started"], patch_error)
        else:
            raw_patch = extract_patch(patch_raw)
            write_success(
                backend,
                "agentless",
                row,
                base,
                item["agentless_started"],
                raw_patch,
                item["loc_tokens_in"] + item["patch_tokens_in"],
                item["loc_tokens_out"] + token_count(tokenizer, patch_raw),
                {
                    "localization_raw": item["loc_raw"],
                    "raw": patch_raw,
                    "localized_files": item["localized_files"],
                    "context_chars": len(item["context"]) + len(item["patch_context"]),
                    "backend": backend,
                    "batch_size": len(patch_items),
                },
                item["fallback_error"],
            )
        remaining["agentless"].add(iid)
        processed["agentless"] += 1

    for item in prepared:
        shutil.rmtree(repo_work_dir(item["row"]), ignore_errors=True)


def run_qwen_job(*, backend: str, model_id: str, quantization: str | None = None) -> None:
    base = output_base()
    base.mkdir(parents=True, exist_ok=True)
    rows = load_rows(DATASET_NAME, SPLIT)
    selected = target_rows(rows, backend)
    remaining = {
        mode: existing_ids_for_generation(backend, mode, rows, base)
        for mode in ("direct", "agentless")
    }

    require_git()
    print(f"Loading tokenizer for {model_id}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    print(f"Loading model {model_id}", flush=True)
    llm_kwargs: dict[str, Any] = {
        "model": model_id,
        "max_model_len": MAX_MODEL_LEN,
        "gpu_memory_utilization": 0.90,
        "trust_remote_code": True,
    }
    if quantization:
        llm_kwargs["quantization"] = quantization
    llm = LLM(**llm_kwargs)

    processed = {"direct": 0, "agentless": 0}
    skipped = {"direct": 0, "agentless": 0}
    batch: list[tuple[int, dict[str, Any], bool, bool]] = []
    batches_done = 0
    for index, row in enumerate(selected, start=1):
        iid = row["instance_id"]
        need_direct = iid not in remaining["direct"]
        need_agentless = iid not in remaining["agentless"]
        if not need_direct and not need_agentless:
            skipped["direct"] += 1
            skipped["agentless"] += 1
            continue

        print(f"[{index}/{len(selected)}] {iid}", flush=True)
        batch.append((index, row, need_direct, need_agentless))
        if len(batch) >= BATCH_SIZE:
            process_batch(
                backend=backend,
                entries=batch,
                base=base,
                tokenizer=tokenizer,
                llm=llm,
                remaining=remaining,
                processed=processed,
            )
            batch = []
            batches_done += 1
            checkpoint(backend, base, batches_done, processed)

        if index % 10 == 0:
            print(
                f"progress rows={index}/{len(selected)} "
                f"direct_new={processed['direct']} agentless_new={processed['agentless']} "
                f"direct_skipped={skipped['direct']} agentless_skipped={skipped['agentless']}",
                flush=True,
            )

    if batch:
        process_batch(
            backend=backend,
            entries=batch,
            base=base,
            tokenizer=tokenizer,
            llm=llm,
            remaining=remaining,
            processed=processed,
        )
        batches_done += 1
        checkpoint(backend, base, batches_done, processed)

    summaries = [merge_predictions_for(backend, mode, rows, base) for mode in ("direct", "agentless")]
    print(json.dumps({"merge_summaries": summaries}, indent=2), flush=True)
    summarize_logs(backend, base)
    bundle_paths = bundle_paths_for(backend, base, include_final=True)
    emit_bundle(base, f"{backend}_extended_outputs", bundle_paths)
    shutil.rmtree(Path("/tmp/swebench_repos/worktrees"), ignore_errors=True)
