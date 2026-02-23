# nuPlan Simulation Environment

Docker-based nuPlan closed-loop simulation environment with nuBoard visualization.

## Features

- ✅ nuPlan closed-loop simulation (IDMPlanner)
- ✅ nuBoard visualization
- 🔄 Diffusion Planner (TODO)

## Quick Start

### 1. Pull Docker Image

```bash
cd scripts
./docker_setup.sh pull
```

### 2. Run Container

```bash
./docker_setup.sh run
```

### 3. Enter Container

```bash
./docker_setup.sh enter
```

---

## Data Preparation

### Download nuPlan Dataset

1. Register at [nuplan.org](https://nuplan.org)
2. Download the following files:

| Dataset | Size | Purpose |
|---------|------|---------|
| `nuplan-v1.1/mini` | ~14GB | Quick testing, development |
| `nuplan-v1.1/train` | ~1TB | Training models |
| `nuplan-v1.1/test` | ~300GB | Benchmark evaluation |
| `nuplan-challenge` | - | Challenge-specific scenarios |
| `nuplan-maps-v1.0` | ~500MB | Map files (required) |

3. Extract and organize:
```
data/nuplan/
├── data/
│   └── cache/
│       └── mini/           # Extract mini dataset here
└── maps/
    └── nuplan-maps-v1.0/  # Extract maps here
```

**Note:** For simulation testing, `mini` dataset is sufficient.

---

## Running Simulation

### Basic Usage

```bash
# Inside Docker container
cd /workspace/nuplan-devkit

# Run simulation with default test split
../scripts/run_simulation.sh

# Run specific number of scenarios
../scripts/run_simulation.sh --num=5

# Run single scenario (by log name)
../scripts/run_simulation.sh --scenario=2021.06.03.12.02.06_veh-35_00233_00609
```

### Simulation Output

Results are saved to:
```
data/nuplan/exp/exp/simulation/closed_loop_nonreactive_agents/<timestamp>/
├── metrics/                    # Evaluation metrics (parquet)
├── simulation_log/             # Raw simulation data
│   └── IDMPlanner/
│       └── <scenario_name>/
│           └── <token>/
│               └── *.msgpack.xz
├── nuboard_*.nuboard           # nuBoard config file
└── log.txt                    # Simulation log
```

---

## Visualization with nuBoard

### Start nuBoard

```bash
# Inside Docker container
../scripts/run_nuboard.sh
```

nuBoard will be available at: **http://localhost:5006**

### Viewing Results

1. Open browser to http://localhost:5006
2. Click **"Open Experiment"**
3. Browse to your simulation results:
   ```
   /workspace/data/nuplan/exp/exp/simulation/closed_loop_nonreactive_agents/<timestamp>/
   ```
4. Select the `.nuboard` file

### Tabs Overview

- **OVERVIEW**: See evaluation scores for all scenarios
- **SCENARIO**: View detailed trajectory playback

---

## Scripts

| Script | Description |
|--------|-------------|
| `docker_setup.sh` | Docker container management |
| `run_simulation.sh` | Run closed-loop simulation |
| `run_nuboard.sh` | Start nuBoard visualization |

### Docker Setup Commands

```bash
./docker_setup.sh pull    # Pull image from Docker Hub
./docker_setup.sh run     # Create and start container
./docker_setup.sh start   # Start existing container
./docker_setup.sh stop    # Stop container
./docker_setup.sh enter   # Enter container shell
./docker_setup.sh rebuild # Rebuild image from container
./docker_setup.sh push    # Push to Docker Hub
```

---

## Known Issues

### msgpack Format Bug

If nuBoard shows "No simulation data" in SCENARIO tab, use pickle format:

```bash
# Add this to run_simulation.sh:
callback.simulation_log_callback.serialization_type=pickle
```

---

## Project Structure

```
diffusion-planner-project/
├── data/
│   └── nuplan/              # nuPlan dataset
│       ├── data/cache/mini/
│       └── maps/
├── nuplan-devkit/          # nuPlan devkit (GitHub version)
└── scripts/
    ├── docker_setup.sh     # Docker management
    ├── run_simulation.sh   # Run simulation
    └── run_nuboard.sh     # Start nuBoard
```

---

## References

- [nuPlan Dataset](https://nuplan.org)
- [nuPlan Devkit](https://github.com/motional/nuplan-devkit)

---

## License

MIT License
