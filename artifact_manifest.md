# Artifact Manifest

This repository is a browsable replication package for the manuscript
"Failure Analysis of Lightweight LLM Code Agents for Software Repair."

## Paper

- `paper/jss_failure_analysis_code_agents_v4.pdf`: latest manuscript copy.
- `paper/jss_failure_analysis_code_agents_v4.tex`: latest manuscript source.
- `paper/jss_failure_analysis_code_agents_v3.*`: previous evaluated version retained for traceability.

## Prompts

- `prompts/direct_prompt.txt`: direct repair prompt.
- `prompts/localization_prompt.txt`: agentless localization prompt.
- `prompts/agentless_patch_prompt.txt`: agentless patch-generation prompt.
- `prompts/direct.txt`, `prompts/agentless_localize.txt`, and `prompts/agentless_patch.txt`: compatibility copies using the filenames expected by the generation scripts.

## Scripts

- `scripts/compute_metrics.py`: preliminary summarization metric script.
- `scripts/repair_patches.py`: patch normalization and repair script.
- `scripts/compare_runs.py`: paired comparison and McNemar exact-test summary.
- `scripts/run_chunked_evaluation.py`: local SWE-bench harness wrapper used for chunked evaluation.
- `scripts/run_openai_300.py`: resumable OpenAI generation script.
- `scripts/run_qwen32b_300.py`: Hugging Face Jobs entrypoint for Qwen 32B AWQ.
- `scripts/run_qwen7b_200.py`: Hugging Face Jobs entrypoint for Qwen 7B.
- `scripts/submit_hf_jobs.py`: Hugging Face Jobs submission helper.
- `scripts/merge_predictions.py`: merge and deduplicate prediction JSONL files.
- `scripts/cost_report.py`: generation cost summary.
- `scripts/validate_predictions.py`: final prediction file validator.

## Data

- `data/swebench_50_instance_ids.txt`: initial 50-instance diagnostic slice.
- `data/swebench_200_qwen7b_instance_ids.txt`: deterministic first-200 Qwen 7B subset.
- `data/swebench_300_instance_ids.txt`: full SWE-bench Lite test split instance IDs used for the 300-instance runs.
- Tests4Py stress-test records are represented as aggregate outputs in `results/preliminary_calibration/tests4py_bug_detection_results.csv`; this package does not redistribute additional source-level Tests4Py material.

## Results

- `results/preliminary_calibration/`: summarization and Tests4Py aggregate metric CSV files.
- `results/swebench_50/`: 50-instance prediction files, summaries, repair summaries, and comparisons.
- `results/swebench_200/`: Qwen 7B 200-instance prediction files, repaired files, repair summaries, reports, and comparison.
- `results/swebench_300/`: gpt-4o-mini and Qwen 32B AWQ 300-instance prediction files, repaired files, repair summaries, reports, and comparisons.
- `results/repair_all_summary.json`: repair validity summary across larger runs.
- `results/cost_report_summary.json`: generation cost summary.
- `results/validate_predictions_summary.json`: validator output for final JSONL files.

## Logs

- `logs/generation_sidecars/`: per-instance generation sidecar JSONL files.
- `logs/docker_harness_summaries/`: local SWE-bench Docker harness outputs retained for auditability.

## Figures

- `figures/figure1_workflow.pdf`: preliminary evaluation workflow.
- `figures/figure2_repository_protocol.pdf`: repository-level protocol.
- `figures/figure3_followup_timeline.pdf`: follow-up timeline.
- `figures/figure4_summarization.pdf`: summarization metric figure.
- `figures/figure5_confusion_matrix.pdf`: Tests4Py stress-test confusion matrix.
- `figures/figure6_swebench_outcomes.pdf`: SWE-bench outcome summary.

## Exclusions

The repository intentionally excludes local virtual environments, API keys,
tokens, `.env` files, private SSH keys, local Docker credential directories,
and other machine-specific secrets.
