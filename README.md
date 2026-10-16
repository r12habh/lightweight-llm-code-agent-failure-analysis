# Replication Package: Auditable Patch Validity for Lightweight Language-Model Software Repair

This repository contains the replication artifacts for the published article:

Rishabh Singh, "Auditable patch validity for lightweight language-model
software repair," *International Journal of Electrical and Computer Engineering
(IJECE)*, vol. 16, no. 5, pp. 2664-2671, 2026.

Published article: https://doi.org/10.11591/ijece.v16i5.pp2664-2671

Published article PDF: [Auditable patch validity for lightweight language-model
software repair](./auditable-patch-validity-lightweight-language-model-software-repair.pdf)

The repository was originally archived under the manuscript working title
"Failure Analysis of Lightweight LLM Code Agents for Software Repair."

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
- `auditable-patch-validity-lightweight-language-model-software-repair.pdf`:
  published IJECE article

## Licensing

Code in this repository is released under the MIT License. Non-code artifacts
including figures, tables, logs, prediction files, and result summaries are
released under CC BY 4.0 unless otherwise noted.

The published article PDF is distributed under the CC BY-SA 4.0 license stated
in the article.

## Citation

If you use this replication package, please cite the published article:

Rishabh Singh. "Auditable patch validity for lightweight language-model
software repair." *International Journal of Electrical and Computer Engineering
(IJECE)*, 16(5), 2664-2671, 2026.
https://doi.org/10.11591/ijece.v16i5.pp2664-2671

Please also cite the archived replication package when referring to the
software or artifacts directly:

Rishabh Singh. *Replication Package for Auditable Patch Validity for Lightweight
Language-Model Software Repair*. Zenodo.
https://doi.org/10.5281/zenodo.20321563
