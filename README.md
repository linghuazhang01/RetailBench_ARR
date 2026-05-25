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
- `artifacts/environment_data.tar.zst.part-*`: processed environment data
  required by the simulator, stored as split archive chunks.
- `artifacts/raw_trajectories_no_checkpoints.tar.zst.part-*`: raw rollout
  trajectories used by the paper analyses, stored as split archive chunks and
  excluding periodic checkpoint snapshots.
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

Restore the environment data and raw trajectories:

```bash
./script/prepare_reproducibility_data.sh
```

Regenerate stage-organized paper-facing figures from the included metrics:

```bash
python3 paper_submit_data/render_four_stage_report.py
```

Regenerate the paper-facing metric tables from the restored raw run directories:

```bash
python3 paper_submit_data/analyze_metrics.py \
  --manifest paper_submit_data/manifest.json \
  --output-dir paper_submit_data/outputs
```

The large files are stored as compressed split archives to avoid depending on
Git LFS. Periodic checkpoint snapshots are omitted; final run databases,
configs, logs, token usage files, `tool_calls.jsonl`, and `run_*.json`
trajectories are retained.

## Double-Blind Review Note

This package intentionally removes repository ownership information, local
filesystem paths, cache files, API credentials, raw private logs, and non-review
workspace artifacts. After acceptance, the authors plan to release a public
repository with a stable archival tag.
