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

## Running the Environment

Restore the archives before running simulator examples:

```bash
./script/prepare_reproducibility_data.sh
```

Run a lightweight tool-interface smoke test:

```bash
python3 retail_environment.py --mode tools --config-type hard_v2
```

Run the standalone simulator with the quality-aware non-LLM policy. The short
example below is intended as a functionality check; the paper setting uses a
180-day horizon.

```bash
python3 agents/run_non_llm_simulation.py \
  --days 10 \
  --config-type hard_v2 \
  --db-path model_run_time/non_llm_hard_v2_demo
```

You can also run the lower-level environment modes directly:

```bash
python3 retail_environment.py \
  --mode quality \
  --days 10 \
  --sample-size 2 \
  --config-type hard_v2 \
  --db-path model_run_time/quality_hard_v2_demo
```

Use `python3 retail_environment.py --mode help` to list available simulator
modes.

## Running LLM Agents

Set model credentials through environment variables or command-line arguments.
Do not edit source files to store keys.

```bash
export OPENAI_API_KEY=<your_api_key>
export OPENAI_BASE_URL=<openai_compatible_base_url>
```

Run a single ReAct agent:

```bash
python3 agents/run_react.py \
  --model <model_name> \
  --config_type hard_v2 \
  --max_days 3 \
  --max_turns 10 \
  --log_dir logs/demo_react
```

Other available agent wrappers are:

```text
agents/run_reflection.py
agents/run_step_reflection.py
agents/run_plan_and_act.py
```

Run one or more agents through the batch launcher:

```bash
python3 script/run_batch_experiments.py \
  --agent run_react.py \
  --config_type hard_v2 \
  --models <model_name> \
  --max_days 3 \
  --workers 1 \
  --output-dir logs/batches \
  --batch-name demo_react
```

For paper-scale experiments, increase `--max_days` to `180` and use the model
and framework list described in `paper_submit_data/manifest.json`.

## Double-Blind Review Note

This package intentionally removes repository ownership information, local
filesystem paths, cache files, API credentials, raw private logs, and non-review
workspace artifacts. After acceptance, the authors plan to release a public
repository with a stable archival tag.
