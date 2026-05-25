# Artifact Reproducibility Guide

This artifact supports three levels of review.

## Level 1: Paper-Facing Metrics and Figures

This level uses included CSV/JSON outputs and does not require API keys or raw
LLM logs.

```bash
python3 paper_submit_data/render_four_stage_report.py
```

Expected outputs include:

- `paper_submit_data/outputs/report.md`
- `paper_submit_data/outputs/best_framework_by_model.csv`
- `paper_submit_data/outputs/four_stage_metrics.csv`
- `paper_submit_data/outputs/four_stage_analysis_report.md`
- `paper_submit_data/outputs/figures/four_stage/*.png`

## Level 2: Raw-Trajectory Metric Regeneration

Restore processed environment data and raw run trajectories:

```bash
./script/prepare_reproducibility_data.sh
```

The script reconstructs the compressed archives from `artifacts/*.part-*`,
verifies `artifacts/SHA256SUMS`, and extracts them into the expected relative
paths.

To verify the archives without extraction, run:

```bash
./script/prepare_reproducibility_data.sh --check-only
```

Then regenerate paper-facing metrics from the raw run directories:

```bash
python3 paper_submit_data/analyze_metrics.py \
  --manifest paper_submit_data/manifest.json \
  --output-dir paper_submit_data/outputs
```

The restored `paper_submit_data/raw_runs/` tree contains each released run's
config, args, logs, token usage files, `tool_calls.jsonl`, `run_*.json`
trajectory, and final `db/records.db`. Periodic checkpoint snapshots are
excluded because they are redundant for metric regeneration and substantially
increase artifact size.

## Level 3: Full Simulator or Agent Runs

Full reruns require the restored processed data and model API credentials.
Set credentials through environment variables rather than editing source files.
For example:

```bash
export OPENAI_API_KEY=...
export OPENAI_BASE_URL=...
python3 script/run_batch_experiments.py --help
```

The oracle-style policy can be inspected through:

```bash
python3 script/run_oracle.py
```

The oracle policy is a privileged reference policy and is not a directly
comparable language-agent baseline.
