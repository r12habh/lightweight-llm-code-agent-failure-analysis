#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Any

from openai import OpenAI

from experiment_extension_utils import (
    CONTEXT_ARGS,
    DATASET_NAME,
    SPLIT,
    append_log,
    append_prediction,
    estimate_openai_cost,
    existing_ids_for_generation,
    final_predictions_path,
    ensure_patch,
    log_path,
    log_record,
    merge_predictions_for,
    new_predictions_path,
    now,
    openai_usage_tokens,
    raw_dir,
    repo_cache_dir,
    repo_work_dir,
    target_rows,
    top_context_files,
    load_rows,
    write_raw,
    MODEL_NAME_BY_KEY,
)
from scripts.generate_predictions import (
    build_context,
    build_user_input,
    clone_checkout,
    extract_patch,
    load_prompt,
)


BACKEND = "gpt4omini"
MODEL = "gpt-4o-mini"
MAX_OUTPUT_TOKENS = 12000
MAX_LOCALIZE_TOKENS = 1500
TEMPERATURE = None


def is_rate_limit_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return "rate limit" in text or "rate_limit" in text or "429" in text


def response_text(resp: Any) -> str:
    text = getattr(resp, "output_text", None)
    if text:
        return str(text)
    parts: list[str] = []
    for item in getattr(resp, "output", []) or []:
        for content in getattr(item, "content", []) or []:
            if getattr(content, "type", "") in {"output_text", "text"}:
                parts.append(str(getattr(content, "text", "")))
    return "\n".join(parts)


def call_openai_with_backoff(
    client: OpenAI,
    *,
    instructions: str,
    user_input: str,
    max_output_tokens: int,
) -> tuple[str, dict[str, Any]]:
    kwargs: dict[str, Any] = {
        "model": MODEL,
        "instructions": instructions,
        "input": user_input,
        "max_output_tokens": max_output_tokens,
        "store": False,
    }
    if TEMPERATURE is not None:
        kwargs["temperature"] = TEMPERATURE

    last_error: Exception | None = None
    for attempt in range(6):
        try:
            resp = client.responses.create(**kwargs)
            usage = getattr(resp, "usage", None)
            if hasattr(usage, "model_dump"):
                usage = usage.model_dump()
            return response_text(resp), dict(usage or {})
        except Exception as exc:
            last_error = exc
            if not is_rate_limit_error(exc) or attempt == 5:
                raise
            sleep_seconds = 10 * (2**attempt)
            print(
                f"OpenAI rate limit on attempt {attempt + 1}/5; sleeping {sleep_seconds}s",
                file=sys.stderr,
                flush=True,
            )
            time.sleep(sleep_seconds)
    raise RuntimeError(f"OpenAI call failed after retries: {last_error!r}")


def parse_localized_files_strict(text: str) -> list[str]:
    data = json.loads(text)
    files = data.get("files", [])
    out = []
    for item in files:
        if isinstance(item, dict) and item.get("path"):
            out.append(str(item["path"]))
    return out[:5]


def generate_direct(client: OpenAI, row: dict[str, Any], repo_dir: Path) -> tuple[str, int, int, dict[str, Any]]:
    context = build_context(repo_dir, row["problem_statement"], **CONTEXT_ARGS)
    raw, usage = call_openai_with_backoff(
        client,
        instructions=load_prompt("prompts/direct.txt"),
        user_input=build_user_input(row, context),
        max_output_tokens=MAX_OUTPUT_TOKENS,
    )
    tokens_in, tokens_out = openai_usage_tokens(usage)
    patch = extract_patch(raw)
    return patch, tokens_in, tokens_out, {"raw": raw, "usage": usage, "context_chars": len(context)}


def generate_agentless(client: OpenAI, row: dict[str, Any], repo_dir: Path) -> tuple[str, int, int, dict[str, Any], str | None]:
    loc_context = build_context(repo_dir, row["problem_statement"], **CONTEXT_ARGS)
    loc_raw, loc_usage = call_openai_with_backoff(
        client,
        instructions=load_prompt("prompts/agentless_localize.txt"),
        user_input=build_user_input(row, loc_context),
        max_output_tokens=MAX_LOCALIZE_TOKENS,
    )
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
    patch_raw, patch_usage = call_openai_with_backoff(
        client,
        instructions=load_prompt("prompts/agentless_patch.txt"),
        user_input=build_user_input(row, patch_context, extra=extra),
        max_output_tokens=MAX_OUTPUT_TOKENS,
    )
    loc_in, loc_out = openai_usage_tokens(loc_usage)
    patch_in, patch_out = openai_usage_tokens(patch_usage)
    return (
        extract_patch(patch_raw),
        loc_in + patch_in,
        loc_out + patch_out,
        {
            "localization_raw": loc_raw,
            "raw": patch_raw,
            "localized_files": localized,
            "usage": {"localize": loc_usage, "patch": patch_usage},
            "context_chars": len(loc_context) + len(patch_context),
        },
        fallback_error,
    )


def run_mode(client: OpenAI, rows: list[dict[str, Any]], mode: str) -> None:
    out_path = new_predictions_path(BACKEND, mode)
    mode_log_path = log_path(BACKEND, mode)
    mode_raw_dir = raw_dir(BACKEND, mode)
    existing_ids = existing_ids_for_generation(BACKEND, mode, rows)
    selected = target_rows(rows, BACKEND)

    tokens_in_total = 0
    tokens_out_total = 0
    processed = 0
    skipped = 0
    for index, row in enumerate(selected, start=1):
        iid = row["instance_id"]
        if iid in existing_ids:
            skipped += 1
            continue

        started = now()
        raw_patch = ""
        error = None
        tokens_in = 0
        tokens_out = 0
        raw_payload: dict[str, Any] = {"instance_id": iid, "mode": mode}
        try:
            repo_dir = clone_checkout(
                row["repo"],
                row["base_commit"],
                repo_cache_dir(),
                repo_work_dir(row),
                force=False,
            )
            if mode == "direct":
                raw_patch, tokens_in, tokens_out, raw_payload = generate_direct(client, row, repo_dir)
            else:
                raw_patch, tokens_in, tokens_out, raw_payload, fallback_error = generate_agentless(client, row, repo_dir)
                error = fallback_error
        except Exception as exc:
            error = repr(exc)
            raw_payload["error"] = error

        prediction = {
            "instance_id": iid,
            "model_patch": ensure_patch(raw_patch),
            "model_name_or_path": MODEL_NAME_BY_KEY[(BACKEND, mode)],
        }
        append_prediction(out_path, prediction)
        write_raw(mode_raw_dir / f"{iid}.json", {**raw_payload, "patch": prediction["model_patch"]})
        append_log(
            mode_log_path,
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
        shutil.rmtree(repo_work_dir(row), ignore_errors=True)
        existing_ids.add(iid)
        processed += 1
        tokens_in_total += tokens_in
        tokens_out_total += tokens_out

        if processed % 10 == 0:
            cost = estimate_openai_cost(tokens_in_total, tokens_out_total)
            print(
                f"{mode}: processed {processed} new instances, skipped {skipped}, "
                f"input_tokens={tokens_in_total}, output_tokens={tokens_out_total}, est_cost=${cost:.4f}",
                flush=True,
            )
        elif index % 25 == 0:
            print(f"{mode}: scanned {index}/{len(selected)} instances, processed={processed}, skipped={skipped}", flush=True)

    summary = merge_predictions_for(BACKEND, mode, rows)
    print(f"{mode}: wrote {out_path}")
    print(f"{mode}: merged final {final_predictions_path(BACKEND, mode)} -> {summary}")


def main() -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is not set in the environment.")
    rows = load_rows(DATASET_NAME, SPLIT)
    client = OpenAI()
    print(f"Loaded {len(rows)} SWE-bench Lite rows; running {MODEL} direct then agentless.")
    run_mode(client, rows, "direct")
    run_mode(client, rows, "agentless")


if __name__ == "__main__":
    main()
