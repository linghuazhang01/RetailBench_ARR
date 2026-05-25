#!/usr/bin/env bash
set -euo pipefail

python3 - <<'PY'
try:
    import matplotlib  # noqa: F401
    import pandas  # noqa: F401
except ModuleNotFoundError as exc:
    raise SystemExit(
        f"Missing dependency: {exc.name}. "
        "Install the artifact requirements first: pip install -r requirements.txt"
    )
PY

python3 paper_submit_data/render_four_stage_report.py
