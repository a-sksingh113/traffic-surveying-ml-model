from __future__ import annotations

import argparse
import ctypes
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import pandas as pd
from ultralytics import YOLO


VEHICLE_CLASSES = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}


@dataclass
class VehicleState:
    vehicle_id: int
    vehicle_type: str = "vehicle"
    last_center: Optional[Tuple[float, float]] = None
    pending_line: Optional[str] = None
    pending_frame: Optional[int] = None


def seconds_to_hhmmss_mmm(seconds: float) -> str:
    total_ms = int(round(max(seconds, 0.0) * 1000.0))
    hours = total_ms // 3_600_000
    minutes = (total_ms % 3_600_000) // 60_000
    secs = (total_ms % 60_000) // 1000
    millis = total_ms % 1000
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


def frame_to_timestamp(frame_index: int, fps: float) -> str:
    if fps <= 0:
        return "00:00:00.000"
    return seconds_to_hhmmss_mmm(frame_index / fps)


def point_side(p1: Tuple[int, int], p2: Tuple[int, int], point: Tuple[float, float]) -> float:
    return (p2[0] - p1[0]) * (point[1] - p1[1]) - (p2[1] - p1[1]) * (point[0] - p1[0])


def crossed_segment(
    prev_point: Optional[Tuple[float, float]],
    curr_point: Tuple[float, float],
    seg_p1: Tuple[int, int],
    seg_p2: Tuple[int, int],
) -> bool:
    if prev_point is None:
        return False

    prev_side = point_side(seg_p1, seg_p2, prev_point)
    curr_side = point_side(seg_p1, seg_p2, curr_point)

    if prev_side == 0.0 or curr_side == 0.0:
        return True

    return (prev_side > 0) != (curr_side > 0)


def clamp_line(value: int, frame_height: int) -> int:
    return max(0, min(frame_height - 1, value))


def parse_point(value: Optional[str], default: Tuple[int, int], width: int, height: int) -> Tuple[int, int]:
    if not value:
        return default
    try:
        x_str, y_str = value.split(",")
        x_val = max(0, min(width - 1, int(x_str.strip())))
        y_val = max(0, min(height - 1, int(y_str.strip())))
        return (x_val, y_val)
    except Exception:
        return default


def shift_segment_vertical(p1: Tuple[int, int], p2: Tuple[int, int], delta: int, height: int) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    y1 = clamp_line(p1[1] + delta, height)
    y2 = clamp_line(p2[1] + delta, height)
    return (p1[0], y1), (p2[0], y2)


def resize_segment(
    p1: Tuple[int, int],
    p2: Tuple[int, int],
    delta_len: float,
    width: int,
    height: int,
) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    x1, y1 = p1
    x2, y2 = p2

    vx = float(x2 - x1)
    vy = float(y2 - y1)
    length = math.hypot(vx, vy)
    if length < 1e-6:
        return p1, p2

    new_length = max(30.0, length + delta_len)
    ux = vx / length
    uy = vy / length
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    half = new_length / 2.0

    nx1 = int(round(cx - ux * half))
    ny1 = int(round(cy - uy * half))
    nx2 = int(round(cx + ux * half))
    ny2 = int(round(cy + uy * half))

    nx1 = max(0, min(width - 1, nx1))
    nx2 = max(0, min(width - 1, nx2))
    ny1 = max(0, min(height - 1, ny1))
    ny2 = max(0, min(height - 1, ny2))
    return (nx1, ny1), (nx2, ny2)


def draw_road_marker_line(image, p1: Tuple[int, int], p2: Tuple[int, int], label: str) -> None:
    cv2.line(image, p1, p2, (255, 255, 255), 4)
    cv2.circle(image, p1, 8, (0, 255, 128), -1)
    cv2.circle(image, p2, 8, (0, 255, 128), -1)
    label_x = max(10, min(p1[0], p2[0]) - 90)
    label_y = max(30, min(p1[1], p2[1]) - 6)
    cv2.putText(image, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)


def get_fit_size(width: int, height: int) -> tuple[int, int]:
    try:
        user32 = ctypes.windll.user32
        screen_w = int(user32.GetSystemMetrics(0) * 0.9)
        screen_h = int(user32.GetSystemMetrics(1) * 0.85)
    except Exception:
        return width, height

    scale = min(screen_w / max(width, 1), screen_h / max(height, 1), 1.0)
    return int(width * scale), int(height * scale)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AI traffic detection and vehicle timestamp logging")
    parser.add_argument("--video", type=str, required=True, help="Path to input video")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="YOLOv8 model path")
    parser.add_argument("--output", type=str, default="traffic_log.xlsx", help="Output Excel path")
    parser.add_argument("--entry-line", type=int, default=None, help="Y coordinate of entry line")
    parser.add_argument("--exit-line", type=int, default=None, help="Y coordinate of exit line")
    parser.add_argument("--entry-p1", type=str, default=None, help="Entry line start point as x,y")
    parser.add_argument("--entry-p2", type=str, default=None, help="Entry line end point as x,y")
    parser.add_argument("--exit-p1", type=str, default=None, help="Exit line start point as x,y")
    parser.add_argument("--exit-p2", type=str, default=None, help="Exit line end point as x,y")
    parser.add_argument("--conf", type=float, default=0.35, help="Detection confidence threshold")
    parser.add_argument("--iou", type=float, default=0.5, help="NMS IoU threshold")
    parser.add_argument("--tracker", type=str, default="bytetrack.yaml", help="Tracker config (e.g. bytetrack.yaml)")

    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size (smaller = faster)")
    parser.add_argument("--skip-frames", type=int, default=0, help="Skip N frames between detections for faster playback")
    parser.add_argument("--playback-speed", type=float, default=1.0, help="Display speed factor (1.0 = normal)")
    parser.add_argument("--show", action="store_true", help="Show live visualization")
    parser.add_argument("--save-annotated", type=str, default=None, help="Path to save annotated output video")
    return parser.parse_args()


def run() -> None:
    args = parse_args()
    video_path = Path(args.video)
    if not video_path.exists():
        raise FileNotFoundError(f"Input video not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    default_entry_p1 = (int(width * 0.48), int(height * 0.64))
    default_entry_p2 = (int(width * 0.66), int(height * 0.64))
    default_exit_p1 = (int(width * 0.52), int(height * 0.80))
    default_exit_p2 = (int(width * 0.95), int(height * 0.80))

    entry_p1 = parse_point(args.entry_p1, default_entry_p1, width, height)
    entry_p2 = parse_point(args.entry_p2, default_entry_p2, width, height)
    exit_p1 = parse_point(args.exit_p1, default_exit_p1, width, height)
    exit_p2 = parse_point(args.exit_p2, default_exit_p2, width, height)

    if args.entry_line is not None:
        y = clamp_line(args.entry_line, height)
        entry_p1, entry_p2 = (int(width * 0.35), y), (int(width * 0.7), y)
    if args.exit_line is not None:
        y = clamp_line(args.exit_line, height)
        exit_p1, exit_p2 = (int(width * 0.45), y), (int(width * 0.95), y)

    model = YOLO(args.model)
    skip_frames = max(0, args.skip_frames)
    playback_speed = args.playback_speed if args.playback_speed > 0 else 1.0

    writer = None
    if args.save_annotated:
        output_video = Path(args.save_annotated)
        output_video.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_video), fourcc, fps if fps > 0 else 25.0, (width, height))

    vehicle_states: Dict[int, VehicleState] = {}
    trip_rows = []
    frame_index = 0
    frame_interval_ms = (1000.0 / fps) / playback_speed if fps > 0 else 1.0

    if args.show:
        cv2.namedWindow("Traffic Detection", cv2.WINDOW_NORMAL)
        disp_w, disp_h = get_fit_size(width, height)
        cv2.resizeWindow("Traffic Detection", disp_w, disp_h)

    while True:
        frame_start = time.perf_counter()
        ok, frame = cap.read()
        if not ok:
            break

        should_infer = skip_frames == 0 or frame_index % (skip_frames + 1) == 0
        results = None
        if should_infer:
            results = model.track(
                source=frame,
                persist=True,
                tracker=args.tracker,
                classes=list(VEHICLE_CLASSES.keys()),
                conf=args.conf,
                iou=args.iou,
                imgsz=args.imgsz,
                verbose=False,
            )

        annotated = frame.copy()
        draw_road_marker_line(annotated, entry_p1, entry_p2, "ENTRY")
        draw_road_marker_line(annotated, exit_p1, exit_p2, "EXIT")
        cv2.putText(
            annotated,
            "W/S: Entry Y  I/K: Exit Y  D/A: Entry len  L/J: Exit len  Q/ESC: Quit",
            (20, max(30, height - 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            annotated,
            f"Speed x{playback_speed:.2f} | Skip {skip_frames} | Img {args.imgsz}",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

        if results and len(results) > 0:
            result = results[0]
            boxes = result.boxes

            if boxes is not None and boxes.id is not None:
                ids = boxes.id.int().cpu().tolist()
                xyxy = boxes.xyxy.cpu().tolist()
                cls_ids = boxes.cls.int().cpu().tolist()

                for track_id, box, class_id in zip(ids, xyxy, cls_ids):
                    x1, y1, x2, y2 = map(int, box)
                    center = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

                    state = vehicle_states.get(track_id)
                    if state is None:
                        state = VehicleState(vehicle_id=track_id)
                        vehicle_states[track_id] = state

                    state.vehicle_type = VEHICLE_CLASSES.get(class_id, "vehicle")

                    crossed_entry = crossed_segment(state.last_center, center, entry_p1, entry_p2)
                    crossed_exit = crossed_segment(state.last_center, center, exit_p1, exit_p2)

                    crossed_line = None
                    if crossed_entry and not crossed_exit:
                        crossed_line = "ENTRY"
                    elif crossed_exit and not crossed_entry:
                        crossed_line = "EXIT"
                    elif crossed_entry and crossed_exit:
                        dist_entry = abs(point_side(entry_p1, entry_p2, center))
                        dist_exit = abs(point_side(exit_p1, exit_p2, center))
                        crossed_line = "ENTRY" if dist_entry <= dist_exit else "EXIT"

                    if crossed_line is not None:
                        if state.pending_line is None:
                            state.pending_line = crossed_line
                            state.pending_frame = frame_index
                        elif state.pending_line == crossed_line:
                            state.pending_frame = frame_index
                        else:
                            first_line = state.pending_line
                            first_frame = state.pending_frame if state.pending_frame is not None else frame_index
                            second_line = crossed_line
                            second_frame = frame_index

                            entry_frame = first_frame if first_line == "ENTRY" else second_frame
                            exit_frame = first_frame if first_line == "EXIT" else second_frame
                            direction = f"{first_line}->{second_line}"

                            trip_rows.append(
                                {
                                    "Vehicle": state.vehicle_id,
                                    "Vehicle type": state.vehicle_type,
                                    "Direction": direction,
                                    "LineA(t1)": frame_to_timestamp(entry_frame, fps),
                                    "LineB(t2)": frame_to_timestamp(exit_frame, fps),
                                }
                            )

                            state.pending_line = None
                            state.pending_frame = None

                    state.last_center = center

                    label = f"ID {track_id} {VEHICLE_CLASSES.get(class_id, 'vehicle')}"
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 200, 0), 2)
                    cv2.putText(annotated, label, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 2)

        if writer is not None:
            writer.write(annotated)

        if args.show:
            cv2.imshow("Traffic Detection", annotated)
            elapsed_ms = (time.perf_counter() - frame_start) * 1000.0
            wait_ms = max(1, int(frame_interval_ms - elapsed_ms))
            key = cv2.waitKey(wait_ms) & 0xFF
            if key == 27 or key == ord("q"):
                break
            if key == ord("w"):
                entry_p1, entry_p2 = shift_segment_vertical(entry_p1, entry_p2, -5, height)
            elif key == ord("s"):
                entry_p1, entry_p2 = shift_segment_vertical(entry_p1, entry_p2, 5, height)
            elif key == ord("i"):
                exit_p1, exit_p2 = shift_segment_vertical(exit_p1, exit_p2, -5, height)
            elif key == ord("k"):
                exit_p1, exit_p2 = shift_segment_vertical(exit_p1, exit_p2, 5, height)
            elif key == ord("d"):
                entry_p1, entry_p2 = resize_segment(entry_p1, entry_p2, 24.0, width, height)
            elif key == ord("a"):
                entry_p1, entry_p2 = resize_segment(entry_p1, entry_p2, -24.0, width, height)
            elif key == ord("l"):
                exit_p1, exit_p2 = resize_segment(exit_p1, exit_p2, 24.0, width, height)
            elif key == ord("j"):
                exit_p1, exit_p2 = resize_segment(exit_p1, exit_p2, -24.0, width, height)

        frame_index += 1

    cap.release()
    if writer is not None:
        writer.release()
    if args.show:
        cv2.destroyAllWindows()

    rows = trip_rows

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=["Vehicle", "Vehicle type", "Direction", "LineA(t1)", "LineB(t2)"]).to_excel(out_path, index=False)
    print(f"Saved {len(rows)} records to {out_path}")


if __name__ == "__main__":
    run()
