#!/usr/bin/env python3
"""
Real-time small-object counter for Jetson cameras.

The counter is aimed at soybean / peanut / pill-like objects on a tray or a
conveyor. It uses OpenCV segmentation instead of a trained detector so it can
run immediately on Jetson without collecting a dataset first.
"""

import argparse
import json
import math
import os
import time
from dataclasses import asdict, dataclass
from datetime import datetime

import cv2
import numpy as np


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CAPTURE_DIR = os.path.join(BASE_DIR, "captures")


def get_csi_pipeline(
    capture_width=1280,
    capture_height=720,
    display_width=1280,
    display_height=720,
    framerate=30,
    flip_method=0,
):
    return (
        "nvarguscamerasrc ! "
        "video/x-raw(memory:NVMM), "
        f"width=(int){capture_width}, height=(int){capture_height}, "
        f"format=(string)NV12, framerate=(fraction){framerate}/1 ! "
        f"nvvidconv flip-method={flip_method} ! "
        f"video/x-raw, width=(int){display_width}, height=(int){display_height}, "
        "format=(string)BGRx ! "
        "videoconvert ! video/x-raw, format=(string)BGR ! appsink"
    )


@dataclass
class CounterConfig:
    mode: str = "color"
    h_min: int = 10
    h_max: int = 45
    s_min: int = 35
    s_max: int = 255
    v_min: int = 40
    v_max: int = 255
    bg_threshold: int = 28
    min_area: int = 80
    max_area: int = 8000
    min_circularity: float = 0.15
    blur: int = 5
    morph: int = 3
    dilate_iter: int = 1
    split_ratio: float = 0.48
    split_touching: bool = True


PRESETS = {
    "soybean": CounterConfig(
        mode="color",
        h_min=12,
        h_max=46,
        s_min=35,
        s_max=255,
        v_min=45,
        v_max=255,
        min_area=70,
        max_area=6000,
        min_circularity=0.2,
        split_ratio=0.45,
    ),
    "peanut": CounterConfig(
        mode="color",
        h_min=4,
        h_max=35,
        s_min=25,
        s_max=255,
        v_min=35,
        v_max=255,
        min_area=100,
        max_area=12000,
        min_circularity=0.08,
        split_ratio=0.42,
    ),
    "pill": CounterConfig(
        mode="background",
        h_min=0,
        h_max=179,
        s_min=0,
        s_max=120,
        v_min=80,
        v_max=255,
        bg_threshold=24,
        min_area=80,
        max_area=10000,
        min_circularity=0.18,
        split_ratio=0.5,
    ),
    "mixed": CounterConfig(
        mode="background",
        bg_threshold=26,
        min_area=70,
        max_area=12000,
        min_circularity=0.08,
        split_ratio=0.46,
    ),
}


class SmallObjectCounter:
    def __init__(self, config):
        self.config = config
        self.background_gray = None
        self.debug_mask = None
        self.last_detections = []

    def set_background(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.background_gray = cv2.GaussianBlur(gray, (21, 21), 0)

    def process(self, frame):
        mask = self._build_mask(frame)
        cleaned = self._clean_mask(mask)
        detections = self._split_and_measure(cleaned, frame)
        self.debug_mask = cleaned
        self.last_detections = detections
        return detections, cleaned

    def _build_mask(self, frame):
        if self.config.mode == "background" and self.background_gray is not None:
            return self._background_mask(frame)

        if self.config.mode == "background" and self.background_gray is None:
            return np.zeros(frame.shape[:2], dtype=np.uint8)

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower = np.array(
            [self.config.h_min, self.config.s_min, self.config.v_min],
            dtype=np.uint8,
        )
        upper = np.array(
            [self.config.h_max, self.config.s_max, self.config.v_max],
            dtype=np.uint8,
        )
        if self.config.h_min <= self.config.h_max:
            return cv2.inRange(hsv, lower, upper)

        lower_a = np.array([0, self.config.s_min, self.config.v_min], dtype=np.uint8)
        upper_a = np.array(
            [self.config.h_max, self.config.s_max, self.config.v_max],
            dtype=np.uint8,
        )
        lower_b = np.array(
            [self.config.h_min, self.config.s_min, self.config.v_min],
            dtype=np.uint8,
        )
        upper_b = np.array([179, self.config.s_max, self.config.v_max], dtype=np.uint8)
        return cv2.bitwise_or(cv2.inRange(hsv, lower_a, upper_a), cv2.inRange(hsv, lower_b, upper_b))

    def _background_mask(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        diff = cv2.absdiff(self.background_gray, gray)
        _, mask = cv2.threshold(diff, self.config.bg_threshold, 255, cv2.THRESH_BINARY)
        return mask

    def _clean_mask(self, mask):
        if self.config.blur >= 3:
            blur = _ensure_odd(self.config.blur)
            mask = cv2.medianBlur(mask, blur)

        morph = max(1, _ensure_odd(self.config.morph))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph, morph))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        if self.config.dilate_iter > 0:
            mask = cv2.dilate(mask, kernel, iterations=self.config.dilate_iter)

        return mask

    def _split_and_measure(self, mask, frame):
        if self.config.split_touching:
            return self._watershed_measure(mask, frame)
        return self._contour_measure(mask)

    def _watershed_measure(self, mask, frame):
        if cv2.countNonZero(mask) == 0:
            return []

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        sure_bg = cv2.dilate(mask, kernel, iterations=2)
        dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
        if float(dist.max()) <= 0:
            return []

        sure_fg = self._distance_peak_seeds(mask, dist)
        unknown = cv2.subtract(sure_bg, sure_fg)

        num_labels, markers = cv2.connectedComponents(sure_fg)
        if num_labels <= 1:
            return self._contour_measure(mask)

        markers = markers + 1
        markers[unknown == 255] = 0
        markers = cv2.watershed(frame, markers)

        detections = []
        for label in range(2, markers.max() + 1):
            object_mask = np.zeros(mask.shape, dtype=np.uint8)
            object_mask[markers == label] = 255
            detection = self._measure_single_mask(object_mask)
            if detection is not None:
                detections.append(detection)

        return detections

    def _distance_peak_seeds(self, mask, dist):
        seed_mask = np.zeros(mask.shape, dtype=np.uint8)
        peak_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        dilated_dist = cv2.dilate(dist, peak_kernel)
        _, components = cv2.connectedComponents(mask)

        for label in range(1, int(components.max()) + 1):
            component = components == label
            component_max = float(dist[component].max())
            if component_max <= 0:
                continue

            threshold = self.config.split_ratio * component_max
            peaks = component & (dist >= threshold) & (dist >= dilated_dist - 1e-5)
            if not np.any(peaks):
                peaks = component & (dist >= threshold)
            seed_mask[peaks] = 255

        seed_mask = cv2.dilate(seed_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)
        return seed_mask

    def _contour_measure(self, mask):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detections = []
        for contour in contours:
            detection = self._measure_contour(contour)
            if detection is not None:
                detections.append(detection)
        return detections

    def _measure_single_mask(self, object_mask):
        contours, _ = cv2.findContours(object_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        contour = max(contours, key=cv2.contourArea)
        return self._measure_contour(contour)

    def _measure_contour(self, contour):
        area = float(cv2.contourArea(contour))
        if area < self.config.min_area or area > self.config.max_area:
            return None

        perimeter = float(cv2.arcLength(contour, True))
        if perimeter <= 0:
            return None

        circularity = 4.0 * math.pi * area / (perimeter * perimeter)
        if circularity < self.config.min_circularity:
            return None

        moments = cv2.moments(contour)
        if moments["m00"] == 0:
            return None

        x, y, w, h = cv2.boundingRect(contour)
        cx = int(moments["m10"] / moments["m00"])
        cy = int(moments["m01"] / moments["m00"])
        return {
            "contour": contour,
            "bbox": (x, y, w, h),
            "centroid": (cx, cy),
            "area": area,
            "circularity": circularity,
        }


class CentroidLineCounter:
    def __init__(self, axis="y", pos_ratio=0.5, direction="both", max_distance=70, max_lost=8):
        self.axis = axis
        self.pos_ratio = pos_ratio
        self.direction = direction
        self.max_distance = max_distance
        self.max_lost = max_lost
        self.next_id = 1
        self.tracks = {}
        self.total = 0

    def reset(self):
        self.next_id = 1
        self.tracks = {}
        self.total = 0

    def update(self, detections, frame_shape, roi_offset=(0, 0)):
        line_pos = self._line_position(frame_shape)
        points = []
        for det in detections:
            cx, cy = det["centroid"]
            points.append((cx + roi_offset[0], cy + roi_offset[1]))

        unmatched_track_ids = set(self.tracks.keys())
        unmatched_points = set(range(len(points)))
        assignments = []

        distances = []
        for track_id, track in self.tracks.items():
            for idx, point in enumerate(points):
                distance = _distance(track["centroid"], point)
                if distance <= self.max_distance:
                    distances.append((distance, track_id, idx))

        for _, track_id, idx in sorted(distances, key=lambda item: item[0]):
            if track_id not in unmatched_track_ids or idx not in unmatched_points:
                continue
            assignments.append((track_id, idx))
            unmatched_track_ids.remove(track_id)
            unmatched_points.remove(idx)

        for track_id, idx in assignments:
            point = points[idx]
            track = self.tracks[track_id]
            previous = track["centroid"]
            track["centroid"] = point
            track["lost"] = 0
            if not track["counted"] and self._crossed(previous, point, line_pos):
                self.total += 1
                track["counted"] = True

        for idx in unmatched_points:
            self.tracks[self.next_id] = {
                "centroid": points[idx],
                "lost": 0,
                "counted": False,
            }
            self.next_id += 1

        for track_id in list(unmatched_track_ids):
            self.tracks[track_id]["lost"] += 1
            if self.tracks[track_id]["lost"] > self.max_lost:
                del self.tracks[track_id]

        return line_pos

    def _line_position(self, frame_shape):
        h, w = frame_shape[:2]
        span = h if self.axis == "y" else w
        return int(span * self.pos_ratio)

    def _crossed(self, previous, current, line_pos):
        prev_value = previous[1] if self.axis == "y" else previous[0]
        curr_value = current[1] if self.axis == "y" else current[0]

        if self.direction == "down" or self.direction == "right":
            return prev_value < line_pos <= curr_value
        if self.direction == "up" or self.direction == "left":
            return prev_value > line_pos >= curr_value
        return (prev_value < line_pos <= curr_value) or (prev_value > line_pos >= curr_value)


class CameraCounterApp:
    def __init__(self, args, config):
        self.args = args
        self.config = config
        self.counter = SmallObjectCounter(config)
        self.line_counter = CentroidLineCounter(
            axis=args.line_axis,
            pos_ratio=args.line_pos,
            direction=args.line_direction,
            max_distance=args.track_distance,
            max_lost=args.track_lost,
        )
        self.frame_count = 0
        self.fps = 0.0
        self.start_time = time.time()
        self.capture_count = 0
        self.roi = args.roi
        self.show_mask = False
        self.tune = args.tune
        self.fullscreen = False
        os.makedirs(CAPTURE_DIR, exist_ok=True)

    def run(self):
        cap = self._open_camera()
        window_name = "Jetson Small Object Counter"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        if self.tune:
            self._create_trackbars(window_name)

        print("[Run] q/ESC quit, s save, b background, r ROI, c reset, t tune, p config")
        if self.config.mode == "background":
            print("[Tip] Background mode needs an empty tray. Press b after the scene is clear.")

        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    print("[Warn] Failed to read camera frame")
                    time.sleep(0.02)
                    continue

                self._update_runtime()
                if self.tune:
                    self._read_trackbars()

                view = frame.copy()
                roi_frame, roi_offset = self._extract_roi(frame)
                detections, mask = self.counter.process(roi_frame)

                line_pos = None
                if self.args.count_mode == "line":
                    line_pos = self.line_counter.update(detections, frame.shape, roi_offset)

                self._draw(view, detections, roi_offset, line_pos)
                if self.show_mask:
                    self._show_mask(mask)

                cv2.imshow(window_name, view)
                key = cv2.waitKey(1) & 0xFF
                if not self._handle_key(key, frame, view, window_name):
                    break
        except KeyboardInterrupt:
            print("\n[Stop] Ctrl+C")
        finally:
            cap.release()
            cv2.destroyAllWindows()
            print(f"[Stats] Frames: {self.frame_count}, snapshots: {self.capture_count}")

    def _open_camera(self):
        if self.args.csi:
            pipeline = get_csi_pipeline(
                capture_width=self.args.width,
                capture_height=self.args.height,
                display_width=self.args.width,
                display_height=self.args.height,
                framerate=self.args.fps,
                flip_method=self.args.flip_method,
            )
            print("[Camera] CSI camera")
            print(f"[GStreamer] {pipeline}")
            cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        else:
            print(f"[Camera] USB camera /dev/video{self.args.device}")
            cap = cv2.VideoCapture(self.args.device)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.args.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.args.height)
            cap.set(cv2.CAP_PROP_FPS, self.args.fps)

        if not cap.isOpened():
            raise RuntimeError("Unable to open camera")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        print(f"[Camera] {width}x{height} @ {fps:.1f} FPS")
        print(f"[Config] mode={self.config.mode}, count_mode={self.args.count_mode}")
        return cap

    def _update_runtime(self):
        self.frame_count += 1
        if self.frame_count % 30 == 0:
            elapsed = max(time.time() - self.start_time, 1e-6)
            self.fps = self.frame_count / elapsed

    def _extract_roi(self, frame):
        if self.roi is None:
            return frame, (0, 0)
        x, y, w, h = clamp_roi(self.roi, frame.shape)
        return frame[y : y + h, x : x + w], (x, y)

    def _draw(self, frame, detections, roi_offset, line_pos):
        ox, oy = roi_offset
        for det in detections:
            x, y, w, h = det["bbox"]
            cx, cy = det["centroid"]
            x += ox
            y += oy
            cx += ox
            cy += oy
            cv2.rectangle(frame, (x, y), (x + w, y + h), (40, 220, 40), 2)
            cv2.circle(frame, (cx, cy), 3, (0, 80, 255), -1)

        if self.roi is not None:
            x, y, w, h = clamp_roi(self.roi, frame.shape)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 180, 20), 2)

        if self.args.count_mode == "line":
            self._draw_line(frame, line_pos)

        self._draw_panel(frame, len(detections))

    def _draw_line(self, frame, line_pos):
        if line_pos is None:
            return
        h, w = frame.shape[:2]
        if self.args.line_axis == "y":
            cv2.line(frame, (0, line_pos), (w, line_pos), (40, 160, 255), 2)
        else:
            cv2.line(frame, (line_pos, 0), (line_pos, h), (40, 160, 255), 2)

    def _draw_panel(self, frame, visible_count):
        h, w = frame.shape[:2]
        panel_w = 360
        panel_h = 152 if self.args.count_mode == "line" else 126
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (panel_w, panel_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.48, frame, 0.52, 0, frame)

        if self.args.count_mode == "line":
            first_line = f"Total: {self.line_counter.total}"
            second_line = f"Visible: {visible_count}"
        else:
            first_line = f"Count: {visible_count}"
            second_line = f"Mode: {self.config.mode}"

        cv2.putText(frame, first_line, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.82, (0, 255, 80), 2)
        cv2.putText(frame, second_line, (12, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 210, 255), 2)
        cv2.putText(frame, f"FPS: {self.fps:.1f}", (12, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)
        cv2.putText(frame, f"Area: {self.config.min_area}-{self.config.max_area}", (12, 114), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)

        if self.config.mode == "background" and self.counter.background_gray is None:
            cv2.putText(frame, "Press b with empty tray", (12, panel_h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (40, 160, 255), 1)

        tips = ["q quit", "s save", "b bg", "r roi", "t tune", "m mask"]
        cv2.putText(frame, "  ".join(tips), (10, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (210, 210, 210), 1)

    def _show_mask(self, mask):
        cv2.imshow("Counter Mask", mask)

    def _handle_key(self, key, frame, view, window_name):
        if key == 255:
            return True
        if key == ord("q") or key == 27:
            return False
        if key == ord("s") or key == 32:
            self._save_snapshot(view)
        elif key == ord("b"):
            roi_frame, _ = self._extract_roi(frame)
            self.counter.set_background(roi_frame)
            print("[Background] Captured")
        elif key == ord("c"):
            self.line_counter.reset()
            print("[Counter] Total reset")
        elif key == ord("r"):
            selected = cv2.selectROI(window_name, frame, fromCenter=False, showCrosshair=True)
            if selected[2] > 0 and selected[3] > 0:
                self.roi = tuple(int(v) for v in selected)
                print(f"[ROI] {self.roi}")
        elif key == ord("0"):
            self.roi = None
            print("[ROI] Cleared")
        elif key == ord("m"):
            self.show_mask = not self.show_mask
            if not self.show_mask:
                cv2.destroyWindow("Counter Mask")
        elif key == ord("t"):
            self.tune = not self.tune
            if self.tune:
                self._create_trackbars(window_name)
            print(f"[Tune] {'on' if self.tune else 'off'}")
        elif key == ord("p"):
            path = self.args.save_config or os.path.join(BASE_DIR, "count_config.json")
            self.save_config(path)
            print(f"[Config] Saved: {path}")
        elif key == ord("f"):
            self.fullscreen = not self.fullscreen
            cv2.setWindowProperty(
                window_name,
                cv2.WND_PROP_FULLSCREEN,
                cv2.WINDOW_FULLSCREEN if self.fullscreen else cv2.WINDOW_NORMAL,
            )
        return True

    def _save_snapshot(self, frame):
        self.capture_count += 1
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(CAPTURE_DIR, f"count_{ts}_{self.capture_count:03d}.jpg")
        cv2.imwrite(filename, frame)
        print(f"[Snapshot] {filename}")

    def _create_trackbars(self, window_name):
        cv2.createTrackbar("H min", window_name, self.config.h_min, 179, lambda value: None)
        cv2.createTrackbar("H max", window_name, self.config.h_max, 179, lambda value: None)
        cv2.createTrackbar("S min", window_name, self.config.s_min, 255, lambda value: None)
        cv2.createTrackbar("S max", window_name, self.config.s_max, 255, lambda value: None)
        cv2.createTrackbar("V min", window_name, self.config.v_min, 255, lambda value: None)
        cv2.createTrackbar("V max", window_name, self.config.v_max, 255, lambda value: None)
        cv2.createTrackbar("BG th", window_name, self.config.bg_threshold, 120, lambda value: None)
        cv2.createTrackbar("Min area", window_name, self.config.min_area, 5000, lambda value: None)
        cv2.createTrackbar("Max area x100", window_name, max(1, self.config.max_area // 100), 600, lambda value: None)
        cv2.createTrackbar("Circular x100", window_name, int(self.config.min_circularity * 100), 100, lambda value: None)
        cv2.createTrackbar("Split x100", window_name, int(self.config.split_ratio * 100), 90, lambda value: None)

    def _read_trackbars(self):
        window_name = "Jetson Small Object Counter"
        self.config.h_min = cv2.getTrackbarPos("H min", window_name)
        self.config.h_max = cv2.getTrackbarPos("H max", window_name)
        self.config.s_min = cv2.getTrackbarPos("S min", window_name)
        self.config.s_max = cv2.getTrackbarPos("S max", window_name)
        self.config.v_min = cv2.getTrackbarPos("V min", window_name)
        self.config.v_max = cv2.getTrackbarPos("V max", window_name)
        self.config.bg_threshold = max(1, cv2.getTrackbarPos("BG th", window_name))
        self.config.min_area = max(1, cv2.getTrackbarPos("Min area", window_name))
        self.config.max_area = max(self.config.min_area + 1, cv2.getTrackbarPos("Max area x100", window_name) * 100)
        self.config.min_circularity = cv2.getTrackbarPos("Circular x100", window_name) / 100.0
        self.config.split_ratio = max(0.05, cv2.getTrackbarPos("Split x100", window_name) / 100.0)

    def save_config(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self.config), f, ensure_ascii=False, indent=2)


def _ensure_odd(value):
    value = int(value)
    if value % 2 == 0:
        value += 1
    return max(1, value)


def _distance(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def parse_roi(value):
    if not value:
        return None
    parts = [int(item.strip()) for item in value.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("--roi must be x,y,w,h")
    x, y, w, h = parts
    if w <= 0 or h <= 0:
        raise argparse.ArgumentTypeError("--roi width and height must be positive")
    return x, y, w, h


def clamp_roi(roi, frame_shape):
    h, w = frame_shape[:2]
    x, y, rw, rh = roi
    x = max(0, min(w - 1, x))
    y = max(0, min(h - 1, y))
    rw = max(1, min(w - x, rw))
    rh = max(1, min(h - y, rh))
    return x, y, rw, rh


def load_config(args):
    if args.config:
        with open(args.config, "r", encoding="utf-8") as f:
            data = json.load(f)
        config = CounterConfig(**data)
    else:
        config = CounterConfig(**asdict(PRESETS[args.preset]))

    config.mode = args.mode or config.mode
    if args.min_area is not None:
        config.min_area = args.min_area
    if args.max_area is not None:
        config.max_area = args.max_area
    if args.bg_threshold is not None:
        config.bg_threshold = args.bg_threshold
    if args.hsv:
        h_min, h_max, s_min, s_max, v_min, v_max = parse_hsv(args.hsv)
        config.h_min = h_min
        config.h_max = h_max
        config.s_min = s_min
        config.s_max = s_max
        config.v_min = v_min
        config.v_max = v_max
    if args.no_split:
        config.split_touching = False
    return config


def parse_hsv(value):
    parts = [int(item.strip()) for item in value.split(",")]
    if len(parts) != 6:
        raise argparse.ArgumentTypeError("--hsv must be h_min,h_max,s_min,s_max,v_min,v_max")
    h_min, h_max, s_min, s_max, v_min, v_max = parts
    if not (0 <= h_min <= 179 and 0 <= h_max <= 179):
        raise argparse.ArgumentTypeError("Hue range must be 0-179")
    if not (0 <= s_min <= 255 and 0 <= s_max <= 255 and 0 <= v_min <= 255 and 0 <= v_max <= 255):
        raise argparse.ArgumentTypeError("S/V ranges must be 0-255")
    if s_min > s_max or v_min > v_max:
        raise argparse.ArgumentTypeError("S/V min values must be <= max values")
    return parts


def parse_args():
    parser = argparse.ArgumentParser(description="Real-time soybean / pill / peanut counter")
    parser.add_argument("--device", type=int, default=0, help="USB camera index")
    parser.add_argument("--csi", action="store_true", help="Use CSI camera through GStreamer")
    parser.add_argument("--width", type=int, default=1280, help="Camera width")
    parser.add_argument("--height", type=int, default=720, help="Camera height")
    parser.add_argument("--fps", type=int, default=30, help="Camera FPS")
    parser.add_argument("--flip-method", type=int, default=0, help="CSI flip method")
    parser.add_argument("--preset", choices=sorted(PRESETS.keys()), default="soybean", help="Object preset")
    parser.add_argument("--config", type=str, default=None, help="Load JSON config")
    parser.add_argument("--mode", choices=["color", "background"], default=None, help="Segmentation mode")
    parser.add_argument("--hsv", type=str, default=None, help="HSV range: h_min,h_max,s_min,s_max,v_min,v_max")
    parser.add_argument("--min-area", type=int, default=None, help="Minimum contour area")
    parser.add_argument("--max-area", type=int, default=None, help="Maximum contour area")
    parser.add_argument("--bg-threshold", type=int, default=None, help="Background difference threshold")
    parser.add_argument("--no-split", action="store_true", help="Disable watershed split for touching objects")
    parser.add_argument("--roi", type=parse_roi, default=None, help="Counting ROI: x,y,w,h")
    parser.add_argument("--tune", action="store_true", help="Show OpenCV tuning trackbars")
    parser.add_argument("--save-config", type=str, default=None, help="Config path used by p hotkey")
    parser.add_argument("--count-mode", choices=["present", "line"], default="present", help="present=count current frame, line=cumulative crossing count")
    parser.add_argument("--line-axis", choices=["x", "y"], default="y", help="Line axis for cumulative mode")
    parser.add_argument("--line-pos", type=float, default=0.5, help="Line position ratio 0.0-1.0")
    parser.add_argument("--line-direction", choices=["both", "up", "down", "left", "right"], default="both", help="Crossing direction")
    parser.add_argument("--track-distance", type=int, default=70, help="Max centroid matching distance")
    parser.add_argument("--track-lost", type=int, default=8, help="Frames to keep lost tracks")
    args = parser.parse_args()

    if not 0.0 < args.line_pos < 1.0:
        parser.error("--line-pos must be between 0 and 1")
    if args.line_axis == "y" and args.line_direction in ("left", "right"):
        parser.error("--line-direction left/right requires --line-axis x")
    if args.line_axis == "x" and args.line_direction in ("up", "down"):
        parser.error("--line-direction up/down requires --line-axis y")
    if args.min_area is not None and args.min_area <= 0:
        parser.error("--min-area must be positive")
    if args.max_area is not None and args.max_area <= 0:
        parser.error("--max-area must be positive")
    if args.min_area is not None and args.max_area is not None and args.min_area >= args.max_area:
        parser.error("--min-area must be smaller than --max-area")

    return args


def main():
    args = parse_args()

    print("=" * 58)
    print("  Jetson Real-time Small Object Counter")
    print("=" * 58)

    config = load_config(args)
    app = CameraCounterApp(args, config)
    app.run()


if __name__ == "__main__":
    main()
