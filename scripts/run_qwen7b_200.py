#!/usr/bin/env python3
from __future__ import annotations

from qwen_job_runner import run_qwen_job


def main() -> None:
    run_qwen_job(
        backend="qwen7b",
        model_id="Qwen/Qwen2.5-Coder-7B-Instruct",
        quantization=None,
    )


if __name__ == "__main__":
    main()
