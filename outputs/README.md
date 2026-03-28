# outputs/

All generated artifacts go here (ignored by git).

Subfolders:
- export/   exported datasets (or symlinks/metadata)
- viz/      visualizations, sample manifests, HTML indexes
- training/ checkpoints, logs, stats
- sim/      closed-loop simulation results, metrics, nuboard artifacts

## Permissions
When writing from Docker, always run with the host UID/GID (avoid root-owned outputs).
