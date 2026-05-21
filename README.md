# Failure Analysis of Lightweight LLM Code Agents for Software Repair

This repository contains replication artifacts for the manuscript:

"Failure Analysis of Lightweight LLM Code Agents for Software Repair"

The artifacts support the reported summarization metrics, Tests4Py stress test,
SWE-bench Lite prediction files, patch repair summaries, paired comparisons,
generation logs, figures, and Docker harness summaries.

## Scope

The reported SWE-bench Lite results are diagnostic baselines, not official
leaderboard submissions. Public benchmark data are referenced by task ID.
This repository does not redistribute private data, credentials, or proprietary
repositories.

## Contents

- `prompts/`: direct and agentless prompting templates
- `scripts/`: metric, repair, comparison, generation, merge, validation, and reporting scripts
- `data/`: SWE-bench Lite instance lists used for the evaluated runs
- `results/`: summarized experimental outputs, prediction JSONL files, repair summaries, and paired comparisons
- `logs/`: generation sidecars and Docker harness summaries
- `figures/`: figures used in the manuscript
- `paper/`: manuscript copies

## Licensing

Code in this repository is released under the MIT License. Non-code artifacts
including figures, tables, logs, prediction files, and result summaries are
released under CC BY 4.0 unless otherwise noted.

## Citation

If you use this artifact, please cite the archived Zenodo release:

Rishabh Singh. Replication Package for Failure Analysis of Lightweight LLM Code
Agents for Software Repair. Zenodo. https://doi.org/10.5281/zenodo.20321563
