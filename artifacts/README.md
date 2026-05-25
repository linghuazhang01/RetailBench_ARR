# Reproducibility Archives

This directory contains compressed archives for full artifact restoration.

- `environment_data.tar.zst.part-*`: split chunks for the processed simulator
  data archive under `data/`.
- `raw_trajectories_no_checkpoints.tar.zst.part-*`: split chunks for the raw
  rollout archive under `paper_submit_data/raw_runs/`, with symlink targets
  materialized and periodic checkpoint snapshots excluded.
- `SHA256SUMS`: checksums for the reconstructed archives.

After cloning, run:

```bash
./script/prepare_reproducibility_data.sh
```

To verify archives without extracting them:

```bash
./script/prepare_reproducibility_data.sh --check-only
```

The script concatenates the chunks, verifies checksums, extracts the archives,
and restores the relative paths expected by `util/default_config.py` and
`paper_submit_data/manifest.json`.
