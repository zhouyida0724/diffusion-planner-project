# Single Frame Feature Extraction

Extract features from a single frame of nuplan scenario using map_api with valid marking.

## Features

- **Polyline features** (dim 0-7):
  - dim 0-1: polyline (x, y) - center line coordinates
  - dim 2-3: polyline_vector (dx, dy) - adjacent point difference  
  - dim 4-5: polyline_to_left - left boundary relative position
  - dim 6-7: polyline_to_right - right boundary relative position

- **Traffic light** (dim 8-11): 4-dim one-hot (green/yellow/red/unknown)

- **Availability array**: boolean array marking valid data points

## Usage

```bash
python scripts/extract_single_frame/extract_single_frame.py
```

## Output

- NPZ file with feature arrays
- CSV file with extracted data

## Configuration

Edit the constants at the top of the script:
- `DB_PATH`: Path to nuplan database
- `MAP_ROOT`: Path to map files
- `SCENARIO_TOKEN`: Target scenario token
- `CENTER_FRAME_INDEX`: Target frame index
