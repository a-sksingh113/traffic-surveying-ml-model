# AI-Based Traffic Detection and Vehicle Timestamp Logging

This project detects and tracks road vehicles in a recorded video, logs when each tracked vehicle crosses an entry line and an exit line, and exports the result to Excel.

## Features

- YOLOv8 vehicle detection (`bicycle`, `car`, `motorcycle`, `bus`, `truck`)
- ByteTrack-based multi-object tracking with persistent IDs
- Entry/Exit line crossing event detection
- Timestamp generation from `frame_index / FPS`
- Excel export to `traffic_log.xlsx`
- Optional live visualization and annotated video output

## Setup

```bash
pip install -r requirements.txt
```

## Run

```bash
python traffic_logger.py --video VID_20260211_152445.mp4 --output traffic_log.xlsx --show
```

When `--show` is enabled, the preview window is opened in resizable mode and fit to your screen while preserving the full frame aspect ratio (no crop zoom).

Interactive controls during preview:

- `W` / `S` → move Entry line up/down
- `I` / `K` → move Exit line up/down
- `D` / `A` → change Entry line length from center
- `L` / `J` → change Exit line length from center
- `E` / `R` → rotate Entry line
- `U` / `O` → rotate Exit line
- `T` / `G` → change Entry line length from `p1` side
- `Y` / `H` → change Entry line length from `p2` side
- `V` / `B` → change Exit line length from `p1` side
- `N` / `M` → change Exit line length from `p2` side
- `Q` or `Esc` → quit processing

Line style is road-marker style (white segment with green endpoints), similar to perspective road guides.

Playback speed tips:

- `--playback-speed 1.0` keeps normal video timing when processing is fast enough
- `--skip-frames 1` (or `2`) reduces inference load and improves playback smoothness
- `--imgsz 512` or `--imgsz 416` improves speed on CPU

## Useful options

- `--entry-line 420` and `--exit-line 520` to manually set line positions
- `--entry-p1 690,460 --entry-p2 960,460` to set entry segment endpoints
- `--exit-p1 760,575 --exit-p2 1360,575` to set exit segment endpoints
- `--model yolov8s.pt` to use a larger YOLOv8 model
- `--tracker bytetrack.yaml` to keep ByteTrack (default)
- `--save-annotated output/annotated.mp4` to save processed video
- `--conf 0.35 --iou 0.5` to tune detection filtering
- `--playback-speed 1.0 --skip-frames 1 --imgsz 512` for faster normal-like playback on CPU

## Output format

The generated Excel file includes:

- Sheet `Trip Log`
	- `Vehicle`
	- `Vehicle Type`
	- `Direction` (`ENTRY->EXIT` or `EXIT->ENTRY`)
	- `LineA(t1)` first crossed-line timestamp (`HH:MM:SS.mmm`)
	- `LineB(t2)` second crossed-line timestamp (`HH:MM:SS.mmm`)
	- `Duration` time between `t1` and `t2` (`SS.mmm`)

- Sheet `Traffic Analysis` (3-minute intervals)
	- `Time Segment (minutes)`
	- `Number of Vehicles`
	- `Length of Segment (m)` (constant = 10)
	- `Time Mean Speed (km/h)`
	- `Space Mean Speed (km/h)`
	- `Flow (vehicles/hour)`
	- `Flow (Q) in PCU/hour`
	- `Density (PCU/km)`

`Traffic Analysis` uses PCU factors:

- `car = 1.0`
- `truck = 3.0`
- `bus = 3.0`
- `motorcycle = 0.5`
- `bicycle = 0.5`

Example:

| Vehicle | Vehicle type | Direction | LineA(t1) | LineB(t2) | Duration |
|-----------:|:------------:|:-----------:|:----------:|:---------:|:---------:|
| 1 | car | ENTRY->EXIT | 00:01:12.145 | 00:01:25.908 | 13.763 |
| 2 | bus | EXIT->ENTRY | 00:02:03.021 | 00:02:18.774 | 15.753 |
