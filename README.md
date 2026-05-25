# RetailBench Anonymous Artifact

This repository is an anonymized reproducibility package for double-blind
review. It contains the RetailBench simulator, agent wrappers, oracle runner,
paper-facing analysis scripts, and generated metric artifacts used to reproduce
the main tables and diagnostic figures.

## Contents

- `retail_environment.py`, `module/`, `agents/`, `model/`, `util/`: core
  simulator and agent code.
- `script/run_oracle.py`: privileged oracle-style policy runner.
- `script/run_batch_experiments.py`: batch launcher for LLM-agent experiments.
- `paper_submit_data/`: paper-facing metric extraction, diagnostic analysis,
  generated CSV/JSON summaries, and figure inputs.
- `configs/paper_main_hard_v2.json`: fixed evaluation configuration summary
  used for the main paper setting.
- `docs/data_access.md`: third-party data access and reconstruction notes.
- `docs/artifact_reproducibility.md`: reviewer-oriented reproduction guide.

## Quick Checks

Install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Regenerate stage-organized paper-facing figures from the included metrics:

```bash
python3 paper_submit_data/render_four_stage_report.py
```

The full LLM rollout logs and third-party raw data are not redistributed in this
anonymous review artifact. The included outputs are sufficient to inspect the
paper-facing tables and diagnostic figures. Full regeneration with
`paper_submit_data/analyze_metrics.py` requires raw run directories; see
`docs/data_access.md` for data reconstruction notes.

## Double-Blind Review Note

This package intentionally removes repository ownership information, local
filesystem paths, cache files, API credentials, raw private logs, and non-review
workspace artifacts. After acceptance, the authors plan to release a public
repository with a stable archival tag.
