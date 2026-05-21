#!/usr/bin/env python3
from __future__ import annotations

from qwen_job_runner import run_qwen_job


def main() -> None:
    run_qwen_job(
        backend="qwen32b",
        model_id="Qwen/Qwen2.5-Coder-32B-Instruct-AWQ",
        quantization="awq",
    )


if __name__ == "__main__":
    main()
