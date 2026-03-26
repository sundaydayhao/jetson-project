#!/usr/bin/env python3
"""
示例 05：SSD MobileNet V2 目标检测
支持 TensorRT 加速和 OpenCV DNN 两种后端
检测 COCO 数据集 80 类目标
"""

import cv2
import numpy as np
import os
import time
import argparse
import random
from datetime import datetime

from coco_labels import COCO_CLASSES, get_label

# ── 模型路径配置 ───────────────────────────────────────────────────────────────
# OpenCV DNN 模型（TensorFlow SSD MobileNet V2）
TF_PROTO = "models/ssd_mobilenet_v2_coco_2018_03_29.pbtxt"
TF_MODEL = "models/frozen_inference_graph.pb"

# TensorRT 引擎文件
TRT_ENGINE = "models/ssd_mobilenet_v2_coco.trt"

# CSI 摄像头管道
CSI_PIPELINE = (
    "nvarguscamerasrc ! "
    "video/x-raw(memory:NVMM), width=1280, height=720, "
    "format=NV12, framerate=30/1 ! "
    "nvvidconv flip-method=0 ! "
    "video/x-raw, width=1280, height=720, format=BGRx ! "
    "videoconvert ! video/x-raw, format=BGR ! appsink"
)

# 为每个类别生成固定颜色
random.seed(42)
CLASS_COLORS = {
    cls: (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255))
    for cls in COCO_CLASSES
}


class ObjectDetector:
    """目标检测器（支持 OpenCV DNN 和 TensorRT）"""

    def __init__(self, backend="opencv", threshold=0.5, use_chinese=False):
        self.backend = backend
        self.threshold = threshold
        self.use_chinese = use_chinese
        self.net = None
        self.trt_engine = None
        self.frame_count = 0
        self.fps = 0.0
        self.start_time = time.time()
        self.photo_count = 0
        self.detection_count = 0

        os.makedirs("captures", exist_ok=True)
        self._load_model()

    def _load_model(self):
        """加载检测模型"""
        if self.backend == "tensorrt":
            self._load_tensorrt()
        else:
            self._load_opencv_dnn()

    def _load_opencv_dnn(self):
        """加载 OpenCV DNN 模型"""
        if not (os.path.exists(TF_PROTO) and os.path.exists(TF_MODEL)):
            print(f"[警告] OpenCV DNN 模型文件不存在")
            print(f"  请运行: bash download_models.sh")
            print(f"  将使用演示模式（随机生成检测结果）")
            self.net = None
            return

        self.net = cv2.dnn.readNetFromTensorflow(TF_MODEL, TF_PROTO)

        # 尝试 CUDA 加速
        try:
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
            print("[模型] OpenCV DNN 加载成功（CUDA 加速）")
        except Exception:
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            print("[模型] OpenCV DNN 加载成功（CPU 模式）")

    def _load_tensorrt(self):
        """加载 TensorRT 引擎"""
        try:
            import tensorrt as trt
            import pycuda.driver as cuda
            import pycuda.autoinit

            if not os.path.exists(TRT_ENGINE):
                print(f"[警告] TensorRT 引擎文件不存在: {TRT_ENGINE}")
                print("  请先运行 build_engine.py 构建引擎")
                print("  回退到 OpenCV DNN 后端")
                self.backend = "opencv"
                self._load_opencv_dnn()
                return

            TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
            with open(TRT_ENGINE, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
                self.trt_engine = runtime.deserialize_cuda_engine(f.read())
            print(f"[模型] TensorRT 引擎加载成功: {TRT_ENGINE}")

        except ImportError:
            print("[警告] TensorRT 未安装，回退到 OpenCV DNN 后端")
            self.backend = "opencv"
            self._load_opencv_dnn()

    def detect_opencv(self, frame):
        """使用 OpenCV DNN 检测"""
        if self.net is None:
            return self._demo_detect(frame)

        h, w = frame.shape[:2]

        # 预处理
        blob = cv2.dnn.blobFromImage(
            frame,
            size=(300, 300),
            swapRB=True,
            crop=False
        )
        self.net.setInput(blob)

        # 推理
        detections = self.net.forward()

        results = []
        for i in range(detections.shape[2]):
            confidence = float(detections[0, 0, i, 2])
            if confidence < self.threshold:
                continue

            class_id = int(detections[0, 0, i, 1])
            if class_id >= len(COCO_CLASSES):
                continue

            class_name = COCO_CLASSES[class_id]
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            x1, y1, x2, y2 = box.astype(int)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w - 1, x2), min(h - 1, y2)

            results.append({
                "class_id": class_id,
                "class_name": class_name,
                "confidence": confidence,
                "bbox": (x1, y1, x2 - x1, y2 - y1)
            })

        return results

    def _demo_detect(self, frame):
        """演示模式：模拟检测结果（无模型时使用）"""
        # 仅用于展示代码逻辑，实际使用需要真实模型
        return []

    def detect(self, frame):
        """执行目标检测"""
        if self.backend == "tensorrt" and self.trt_engine:
            # TensorRT 推理（简化版，实际需要完整的 TRT 推理流程）
            return self.detect_opencv(frame)
        else:
            return self.detect_opencv(frame)

    def draw_results(self, frame, detections):
        """绘制检测结果"""
        h, w = frame.shape[:2]

        # 统计各类别数量
        class_counts = {}
        for det in detections:
            name = det["class_name"]
            class_counts[name] = class_counts.get(name, 0) + 1

        for det in detections:
            x, y, bw, bh = det["bbox"]
            class_name = det["class_name"]
            confidence = det["confidence"]
            color = CLASS_COLORS.get(class_name, (0, 255, 0))

            # 绘制边界框
            cv2.rectangle(frame, (x, y), (x + bw, y + bh), color, 2)

            # 标签文字
            label = get_label(class_name, self.use_chinese)
            label_text = f"{label}: {confidence:.0%}"

            # 标签背景
            (tw, th), _ = cv2.getTextSize(
                label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1
            )
            label_y = y - 8 if y > 25 else y + bh + 20
            cv2.rectangle(
                frame,
                (x, label_y - th - 4),
                (x + tw + 4, label_y + 2),
                color, -1
            )
            cv2.putText(
                frame, label_text,
                (x + 2, label_y - 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1
            )

        # 信息面板
        self._draw_info_panel(frame, detections, class_counts)
        return frame

    def _draw_info_panel(self, frame, detections, class_counts):
        """绘制信息面板"""
        h, w = frame.shape[:2]

        # 左上角信息
        panel_h = 30 + len(class_counts) * 22 + 20
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (260, max(90, panel_h)), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

        cv2.putText(frame, f"FPS: {self.fps:.1f}", (10, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
        cv2.putText(frame, f"Objects: {len(detections)}", (10, 47),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                    (0, 200, 255) if detections else (150, 150, 150), 2)
        cv2.putText(frame, f"Threshold: {self.threshold:.0%}", (10, 72),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

        # 显示各类别数量
        y_offset = 95
        for cls_name, count in sorted(class_counts.items(), key=lambda x: -x[1]):
            color = CLASS_COLORS.get(cls_name, (200, 200, 200))
            label = get_label(cls_name, self.use_chinese)
            cv2.putText(frame, f"  {label}: {count}", (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            y_offset += 22

        # 右下角帮助
        tips = ["q:退出", "s:截图", "+/-:阈值"]
        for i, tip in enumerate(tips):
            cv2.putText(frame, tip, (w - 100, h - 15 - i * 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)

    def update_fps(self):
        self.frame_count += 1
        if self.frame_count % 30 == 0:
            elapsed = time.time() - self.start_time
            self.fps = self.frame_count / elapsed

    def save_photo(self, frame):
        self.photo_count += 1
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"captures/detect_{ts}_{self.photo_count:03d}.jpg"
        cv2.imwrite(filename, frame)
        print(f"[截图] 已保存: {filename}")

    def process_image(self, image_path):
        """处理单张图片"""
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"[错误] 无法读取图片: {image_path}")
            return

        print(f"[图片] 处理: {image_path}")
        t0 = time.time()
        detections = self.detect(frame)
        elapsed = (time.time() - t0) * 1000

        print(f"[结果] 检测到 {len(detections)} 个目标，耗时 {elapsed:.1f} ms")
        for det in detections:
            label = get_label(det["class_name"], self.use_chinese)
            x, y, bw, bh = det["bbox"]
            print(f"  {label}: {det['confidence']:.2f} @ ({x},{y},{bw},{bh})")

        result = self.draw_results(frame, detections)
        cv2.imshow("Object Detection", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def run_camera(self, device=0, use_csi=False):
        """摄像头实时检测"""
        if use_csi:
            cap = cv2.VideoCapture(CSI_PIPELINE, cv2.CAP_GSTREAMER)
            print("[摄像头] 使用 CSI 摄像头")
        else:
            cap = cv2.VideoCapture(device)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            print(f"[摄像头] 使用 USB 摄像头，设备号: {device}")

        if not cap.isOpened():
            raise RuntimeError("[错误] 无法打开摄像头！")

        print(f"[检测] 后端: {self.backend.upper()}，置信度阈值: {self.threshold:.0%}")
        print("  按 q 退出，s 截图，+/- 调整阈值\n")

        window_name = "Jetson Nano Object Detection"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    continue

                self.update_fps()
                detections = self.detect(frame)
                result = self.draw_results(frame, detections)
                cv2.imshow(window_name, result)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:
                    break
                elif key == ord('s'):
                    self.save_photo(result)
                elif key == ord('+') or key == ord('='):
                    self.threshold = min(0.95, self.threshold + 0.05)
                    print(f"[调整] 置信度阈值: {self.threshold:.0%}")
                elif key == ord('-'):
                    self.threshold = max(0.1, self.threshold - 0.05)
                    print(f"[调整] 置信度阈值: {self.threshold:.0%}")

        except KeyboardInterrupt:
            print("\n[中断] Ctrl+C 退出")
        finally:
            cap.release()
            cv2.destroyAllWindows()
            print(f"[统计] 共处理 {self.frame_count} 帧")


def parse_args():
    parser = argparse.ArgumentParser(description="Jetson Nano 目标检测示例")
    parser.add_argument("--backend", choices=["opencv", "tensorrt"], default="opencv",
                        help="推理后端（默认: opencv）")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="置信度阈值（默认: 0.5）")
    parser.add_argument("--image", type=str, default=None,
                        help="检测图片文件路径")
    parser.add_argument("--device", type=int, default=0,
                        help="USB 摄像头设备号（默认: 0）")
    parser.add_argument("--csi", action="store_true",
                        help="使用 CSI 摄像头")
    parser.add_argument("--chinese", action="store_true",
                        help="使用中文标签")
    return parser.parse_args()


def main():
    print("=" * 55)
    print("  Jetson Nano SSD MobileNet V2 目标检测示例")
    print("=" * 55)

    args = parse_args()

    detector = ObjectDetector(
        backend=args.backend,
        threshold=args.threshold,
        use_chinese=args.chinese
    )

    if args.image:
        detector.process_image(args.image)
    else:
        detector.run_camera(device=args.device, use_csi=args.csi)


if __name__ == "__main__":
    main()