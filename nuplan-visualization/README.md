# nuPlan Visualization (nuBoard)

Minimal nuplan-devkit for running nuBoard visualization.

## Requirements

- nuplan-devkit installed via pip: `pip install nuplan-devkit`
- bokeh==2.4.2
- Other dependencies from nuplan-devkit

## Usage

```bash
# Inside Docker container with nuplan-devkit installed
cd /path/to/nuplan-visualization

# Start nuBoard
python -m nuplan.planning.script.run_nuboard \
    scenario_builder=nuplan \
    'scenario_builder.data_root=/path/to/nuplan/data/cache/mini'
```

Then open http://localhost:5006 in your browser.

## What This Contains

- `nuplan/planning/nuboard/` - nuBoard visualization code
- `nuplan/planning/script/config/nuboard/` - nuBoard Hydra config
- `nuplan/planning/simulation/` - simulation data types
- `nuplan/planning/metrics/` - metrics computation
- `nuplan/database/` - database utilities (required for data loading)
- `nuplan/common/` - common utilities
- `nuplan/planning/scenario_builder/` - scenario builder

## What Was Removed

- Training code
- Submission code  
- CLI tools
- Documentation
- Tests

## Size

~10MB (vs ~500MB+ for full nuplan-devkit)
