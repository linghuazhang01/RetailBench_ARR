# Artifact Reproducibility Guide

This artifact supports two levels of review.

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

## Level 2: Full Simulator or Agent Runs

Full reruns require reconstructed processed data and model API credentials.
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

Full metric regeneration with `paper_submit_data/analyze_metrics.py` requires
raw run directories corresponding to `paper_submit_data/manifest.json`. The
anonymous artifact includes generated metrics and paper-facing summaries, but
does not redistribute the large raw LLM rollout logs.
