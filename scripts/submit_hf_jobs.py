#!/usr/bin/env python3
from __future__ import annotations

import os
import subprocess
import tarfile
import tempfile
from pathlib import Path

from huggingface_hub import HfApi


IMAGE = "vllm/vllm-openai:latest"
DEPENDENCIES = [
    "datasets>=2.19.0",
    "openai>=2.0.0",
    "tqdm>=4.66.0",
    "huggingface_hub>=0.34.0",
]
COMMON_ENV = {
    "OUTPUTS_DIR": "/outputs",
    "HF_HUB_ENABLE_HF_TRANSFER": "1",
    "QWEN_BATCH_SIZE": "8",
    "QWEN_CHECKPOINT_EVERY_BATCHES": "4",
    "BUNDLE_CHECKPOINT_BASE64": "1",
    "BUNDLE_PASTE_FALLBACK": "1",
}
CODE_FILES = [
    "run_qwen32b_300.py",
    "run_qwen7b_200.py",
    "qwen_job_runner.py",
    "experiment_extension_utils.py",
    "scripts/generate_predictions.py",
    "prompts/agentless_patch.txt",
    "prompts/direct.txt",
    "prompts/agentless_localize.txt",
    "outputs/hf_qwen25_coder7b_50_20260508_a10g/direct_predictions.jsonl",
    "outputs/hf_qwen25_coder7b_50_20260508_a10g/agentless_predictions.jsonl",
    "outputs/hf_qwen25_coder32b_awq_50_20260508_l40s/direct_predictions.jsonl",
    "outputs/hf_qwen25_coder32b_awq_50_20260508_l40s/agentless_predictions.jsonl",
    "outputs/qwen32b_checkpoint_b12_backup/new_predictions_qwen32b_direct.jsonl",
    "outputs/qwen32b_checkpoint_b12_backup/new_predictions_qwen32b_agentless.jsonl",
    "outputs/qwen32b_checkpoint_b12_backup/log_qwen32b_direct.jsonl",
    "outputs/qwen32b_checkpoint_b12_backup/log_qwen32b_agentless.jsonl",
]


def build_bundle(root: Path) -> Path:
    tmp = Path(tempfile.mkdtemp(prefix="swebench_hf_bundle_"))
    archive = tmp / "swebench_extension_code.tgz"
    with tarfile.open(archive, "w:gz") as tar:
        for rel in CODE_FILES:
            path = root / rel
            if not path.exists():
                raise FileNotFoundError(path)
            tar.add(path, arcname=rel)
    return archive


def upload_to_catbox(archive: Path) -> str:
    proc = subprocess.run(
        [
            "curl",
            "-fsS",
            "-F",
            "reqtype=fileupload",
            "-F",
            f"fileToUpload=@{archive}",
            "https://catbox.moe/user/api.php",
        ],
        text=True,
        capture_output=True,
        check=False,
        timeout=120,
    )
    if proc.returncode != 0 or not proc.stdout.strip().startswith("https://"):
        detail = (proc.stderr or proc.stdout or "").replace("\n", " ")[:500]
        raise RuntimeError(f"catbox upload failed: {detail}")
    return proc.stdout.strip()


def command_for(bundle_url: str, script_name: str) -> list[str]:
    return [
        "/bin/bash",
        "-lc",
        f"""set -euo pipefail
mkdir -p /workspace /outputs
cd /workspace
if ! command -v git >/dev/null 2>&1; then
  if command -v apt-get >/dev/null 2>&1; then
    apt-get update
    DEBIAN_FRONTEND=noninteractive apt-get install -y git curl ca-certificates
  else
    echo 'git missing and apt-get unavailable'
    exit 64
  fi
fi
git --version
curl -L --fail --retry 5 --retry-delay 5 -o code.tgz {bundle_url}
tar -xzf code.tgz
if [ -f outputs/qwen32b_checkpoint_b12_backup/new_predictions_qwen32b_direct.jsonl ]; then
  cp outputs/qwen32b_checkpoint_b12_backup/new_predictions_qwen32b_direct.jsonl /outputs/new_predictions_qwen32b_direct.jsonl
  cp outputs/qwen32b_checkpoint_b12_backup/new_predictions_qwen32b_agentless.jsonl /outputs/new_predictions_qwen32b_agentless.jsonl
  cp outputs/qwen32b_checkpoint_b12_backup/log_qwen32b_direct.jsonl /outputs/log_qwen32b_direct.jsonl
  cp outputs/qwen32b_checkpoint_b12_backup/log_qwen32b_agentless.jsonl /outputs/log_qwen32b_agentless.jsonl
  echo 'Seeded Qwen32B /outputs from checkpoint b12.'
fi
python3 -m pip install -q {' '.join(repr(dep) for dep in DEPENDENCIES)}
python3 {script_name}
""",
    ]


def submit(api: HfApi, *, bundle_url: str, script_name: str, flavor: str, timeout: str, token: str):
    return api.run_job(
        image=IMAGE,
        command=command_for(bundle_url, script_name),
        env=COMMON_ENV,
        secrets={"HF_TOKEN": token},
        flavor=flavor,
        timeout=timeout,
        token=token,
    )


def main() -> None:
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN is not set in the environment.")

    root = Path(__file__).resolve().parent
    bundle = build_bundle(root)
    bundle_url = upload_to_catbox(bundle)
    print(f"Uploaded job code bundle: {bundle_url}")

    api = HfApi(token=token)
    print("Submitting Qwen 32B AWQ job first.")
    print("Budget guard: l40sx1 timeout is 3.25h, max exposure about $5.85 at $1.80/hr.")
    job32 = submit(
        api,
        bundle_url=bundle_url,
        script_name="run_qwen32b_300.py",
        flavor="l40sx1",
        timeout="3.25h",
        token=token,
    )
    print(f"Qwen 32B job id: {job32.id}")
    print(f"Qwen 32B job url: {job32.url}")

    answer = input(
        "Submit Qwen 7B 200-instance a10g-large job now? "
        "This adds a 2.1h timeout, max exposure about $3.15. Type 'submit 7b' to continue: "
    ).strip()
    if answer != "submit 7b":
        print("Skipped Qwen 7B submission.")
        return

    job7 = submit(
        api,
        bundle_url=bundle_url,
        script_name="run_qwen7b_200.py",
        flavor="a10g-large",
        timeout="2.1h",
        token=token,
    )
    print(f"Qwen 7B job id: {job7.id}")
    print(f"Qwen 7B job url: {job7.url}")


if __name__ == "__main__":
    main()
