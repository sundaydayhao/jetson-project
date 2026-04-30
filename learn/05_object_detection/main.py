#!/usr/bin/env python3
"""
示例 05：SSD MobileNet 目标检测

- OpenCV 后端：TensorFlow SSD MobileNet V2
- TensorRT 后端：ONNX SSD MobileNet V1
"""

import argparse
import os
import random
import time
from datetime import datetime

import cv2
import numpy as np

from coco_labels import VALID_COCO_CLASS_NAMES, get_class_name, get_label


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
CAPTURE_DIR = os.path.join(BASE_DIR, "captures")

# OpenCV DNN 模型（TensorFlow SSD MobileNet V2）
TF_PROTO = os.path.join(MODEL_DIR, "ssd_mobilenet_v2_coco_2018_03_29.pbtxt")
TF_MODEL = os.path.join(MODEL_DIR, "frozen_inference_graph.pb")

# TensorRT 路径（ONNX Model Zoo SSD MobileNet V1）
TRT_ONNX = os.path.join(MODEL_DIR, "ssd_mobilenet_v1_12.onnx")
TRT_ENGINE = os.path.join(MODEL_DIR, "ssd_mobilenet_v1_12_fp16.trt")

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
    cls: (
        random.randint(50, 255),
        random.randint(50, 255),
        random.randint(50, 255),
    )
    for cls in VALID_COCO_CLASS_NAMES
}


class TRTObjectDetector:
    """TensorRT SSD 检测器"""

    def __init__(self, engine_path):
        self.engine_path = engine_path
        self.engine = None
        self.context = None
        self.stream = None
        self.trt = None
        self.cuda = None

        self.input_name = None
        self.output_names = []
        self.output_map = {}
        self.input_dtype = None
        self.input_layout = "NHWC"
        self.profile_min_shape = None
        self.profile_opt_shape = None
        self.profile_max_shape = None

        self.buffers = {}
        self.current_input_shape = None

        self._load_engine()

    def _load_engine(self):
        try:
            import tensorrt as trt
            import pycuda.driver as cuda
            import pycuda.autoinit  # noqa: F401
        except ImportError as e:
            raise ImportError(
                f"缺少 TensorRT 依赖: {e}\n"
                "请确认已安装 TensorRT 和 pycuda"
            ) from e

        if not os.path.exists(self.engine_path):
            raise FileNotFoundError(
                f"TensorRT 引擎不存在: {self.engine_path}\n"
                "请先运行: python3 build_engine.py"
            )

        self.trt = trt
        self.cuda = cuda

        logger = trt.Logger(trt.Logger.WARNING)
        with open(self.engine_path, "rb") as f, trt.Runtime(logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())

        if self.engine is None:
            raise RuntimeError(f"无法反序列化 TensorRT 引擎: {self.engine_path}")

        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("无法创建 TensorRT 执行上下文")

        self.stream = cuda.Stream()

        input_names = []
        output_names = []
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            mode = self.engine.get_tensor_mode(name)
            if mode == trt.TensorIOMode.INPUT:
                input_names.append(name)
            else:
                output_names.append(name)

        if len(input_names) != 1:
            raise RuntimeError("当前示例仅支持单输入 TensorRT 检测模型")

        self.input_name = input_names[0]
        self.output_names = output_names
        self.input_dtype = trt.nptype(self.engine.get_tensor_dtype(self.input_name))

        input_shape = tuple(int(dim) for dim in self.engine.get_tensor_shape(self.input_name))
        if any(dim < 0 for dim in input_shape):
            min_shape, opt_shape, max_shape = self.engine.get_tensor_profile_shape(
                self.input_name, 0
            )
            self.profile_min_shape = tuple(int(dim) for dim in min_shape)
            self.profile_opt_shape = tuple(int(dim) for dim in opt_shape)
            self.profile_max_shape = tuple(int(dim) for dim in max_shape)
        else:
            self.profile_min_shape = input_shape
            self.profile_opt_shape = input_shape
            self.profile_max_shape = input_shape

        self.input_layout = self._infer_input_layout()
        self.output_map = self._resolve_output_names()

        print(f"[TRT] 引擎加载成功: {self.engine_path}")
        print(
            "[TRT] 输入: "
            f"{self.input_name} "
            f"{self.profile_min_shape} -> {self.profile_opt_shape} -> {self.profile_max_shape} "
            f"{self.input_dtype.__name__}"
        )
        print(
            "[TRT] 输出: "
            + ", ".join(
                f"{key}={value}"
                for key, value in self.output_map.items()
                if value
            )
        )

    def _infer_input_layout(self):
        shape = self.profile_opt_shape
        if len(shape) != 4:
            raise RuntimeError(f"不支持的输入维度: {shape}")

        if shape[-1] == 3:
            return "NHWC"
        if shape[1] == 3:
            return "NCHW"
        return "NHWC"

    def _resolve_output_names(self):
        mapping = {
            "num_detections": None,
            "detection_boxes": None,
            "detection_scores": None,
            "detection_classes": None,
        }

        for name in self.output_names:
            lowered = name.lower()
            if "num_detections" in lowered:
                mapping["num_detections"] = name
            elif "detection_boxes" in lowered:
                mapping["detection_boxes"] = name
            elif "detection_scores" in lowered:
                mapping["detection_scores"] = name
            elif "detection_classes" in lowered:
                mapping["detection_classes"] = name

        if all(mapping.values()):
            return mapping

        fallback = list(self.output_names)
        expected = [
            "num_detections",
            "detection_boxes",
            "detection_scores",
            "detection_classes",
        ]
        for key, name in zip(expected, fallback):
            if mapping[key] is None:
                mapping[key] = name

        return mapping

    def _shape_in_profile(self, input_shape):
        for value, min_value, max_value in zip(
            input_shape, self.profile_min_shape, self.profile_max_shape
        ):
            if value < min_value or value > max_value:
                return False
        return True

    def _prepare_input(self, frame):
        """准备 TensorRT 输入，并返回缩放信息"""
        original_h, original_w = frame.shape[:2]
        input_frame = frame
        scale_x = 1.0
        scale_y = 1.0

        if self.input_layout == "NHWC":
            input_shape = (1, original_h, original_w, 3)
            if not self._shape_in_profile(input_shape):
                target_h = self.profile_opt_shape[1]
                target_w = self.profile_opt_shape[2]
                input_frame = cv2.resize(frame, (target_w, target_h))
                scale_x = original_w / target_w
                scale_y = original_h / target_h

            rgb = cv2.cvtColor(input_frame, cv2.COLOR_BGR2RGB)
            input_data = np.expand_dims(rgb.astype(self.input_dtype), axis=0)
        else:
            target_h = self.profile_opt_shape[2]
            target_w = self.profile_opt_shape[3]
            if frame.shape[0] != target_h or frame.shape[1] != target_w:
                input_frame = cv2.resize(frame, (target_w, target_h))
                scale_x = original_w / target_w
                scale_y = original_h / target_h

            rgb = cv2.cvtColor(input_frame, cv2.COLOR_BGR2RGB)
            chw = rgb.transpose(2, 0, 1)
            input_data = np.expand_dims(chw.astype(self.input_dtype), axis=0)

        return np.ascontiguousarray(input_data), {
            "orig_h": original_h,
            "orig_w": original_w,
            "input_h": input_frame.shape[0],
            "input_w": input_frame.shape[1],
            "scale_x": scale_x,
            "scale_y": scale_y,
        }

    def _ensure_buffers(self, input_shape):
        input_shape = tuple(int(dim) for dim in input_shape)
        if self.current_input_shape == input_shape:
            return

        if not self.context.set_input_shape(self.input_name, input_shape):
            raise RuntimeError(f"无法设置 TensorRT 输入形状: {input_shape}")

        self.buffers = {}
        tensor_names = [self.input_name] + self.output_names
        for tensor_name in tensor_names:
            tensor_shape = tuple(
                int(dim) for dim in self.context.get_tensor_shape(tensor_name)
            )
            if any(dim < 0 for dim in tensor_shape):
                raise RuntimeError(f"TensorRT 输出形状未解析完成: {tensor_name} {tensor_shape}")

            dtype = self.trt.nptype(self.engine.get_tensor_dtype(tensor_name))
            size = int(self.trt.volume(tensor_shape))
            host_mem = self.cuda.pagelocked_empty(size, dtype)
            device_mem = self.cuda.mem_alloc(host_mem.nbytes)
            self.context.set_tensor_address(tensor_name, int(device_mem))

            self.buffers[tensor_name] = {
                "host": host_mem,
                "device": device_mem,
                "shape": tensor_shape,
                "dtype": dtype,
            }

        self.current_input_shape = input_shape

    def infer(self, frame, threshold=0.5):
        input_data, meta = self._prepare_input(frame)
        self._ensure_buffers(input_data.shape)

        np.copyto(self.buffers[self.input_name]["host"], input_data.ravel())
        self.cuda.memcpy_htod_async(
            self.buffers[self.input_name]["device"],
            self.buffers[self.input_name]["host"],
            self.stream,
        )

        if not self.context.execute_async_v3(stream_handle=self.stream.handle):
            raise RuntimeError("TensorRT 执行失败")

        outputs = {}
        for name in self.output_names:
            buffer = self.buffers[name]
            self.cuda.memcpy_dtoh_async(buffer["host"], buffer["device"], self.stream)

        self.stream.synchronize()

        for name in self.output_names:
            buffer = self.buffers[name]
            outputs[name] = buffer["host"].reshape(buffer["shape"]).copy()

        return self._decode_outputs(outputs, meta, threshold)

    def _decode_outputs(self, outputs, meta, threshold):
        num_name = self.output_map["num_detections"]
        boxes_name = self.output_map["detection_boxes"]
        scores_name = self.output_map["detection_scores"]
        classes_name = self.output_map["detection_classes"]

        if not all([num_name, boxes_name, scores_name, classes_name]):
            raise RuntimeError("TensorRT 输出名称解析失败，无法完成后处理")

        num_detections = int(np.ravel(outputs[num_name])[0])
        boxes = np.array(outputs[boxes_name]).reshape(-1, 4)
        scores = np.array(outputs[scores_name]).reshape(-1)
        classes = np.array(outputs[classes_name]).reshape(-1)

        input_h = meta["input_h"]
        input_w = meta["input_w"]
        scale_x = meta["scale_x"]
        scale_y = meta["scale_y"]
        orig_h = meta["orig_h"]
        orig_w = meta["orig_w"]

        results = []
        for i in range(min(num_detections, len(scores), len(classes), len(boxes))):
            confidence = float(scores[i])
            if confidence < threshold:
                continue

            class_name = get_class_name(classes[i])
            if not class_name or class_name == "background":
                continue

            y1, x1, y2, x2 = boxes[i]
            x1 = int(max(0, min(orig_w - 1, x1 * input_w * scale_x)))
            y1 = int(max(0, min(orig_h - 1, y1 * input_h * scale_y)))
            x2 = int(max(0, min(orig_w - 1, x2 * input_w * scale_x)))
            y2 = int(max(0, min(orig_h - 1, y2 * input_h * scale_y)))

            if x2 <= x1 or y2 <= y1:
                continue

            results.append({
                "class_name": class_name,
                "confidence": confidence,
                "bbox": (x1, y1, x2 - x1, y2 - y1),
            })

        return results


class ObjectDetector:
    """目标检测器（支持 OpenCV DNN 和 TensorRT）"""

    def __init__(
        self,
        backend="opencv",
        threshold=0.5,
        use_chinese=False,
        engine_path=TRT_ENGINE,
        onnx_path=TRT_ONNX,
        trt_precision="fp16",
    ):
        self.backend = backend
        self.threshold = threshold
        self.use_chinese = use_chinese
        self.engine_path = engine_path
        self.onnx_path = onnx_path
        self.trt_precision = trt_precision

        self.net = None
        self.trt_detector = None
        self.frame_count = 0
        self.fps = 0.0
        self.start_time = time.time()
        self.photo_count = 0
        self.opencv_target = "CPU"
        self._warned_cuda_fallback = False

        os.makedirs(CAPTURE_DIR, exist_ok=True)
        self._load_model()

    def _load_model(self):
        if self.backend == "tensorrt":
            self._load_tensorrt()
        else:
            self._load_opencv_dnn()

    def _load_opencv_dnn(self):
        if not (os.path.exists(TF_PROTO) and os.path.exists(TF_MODEL)):
            print("[警告] OpenCV DNN 模型文件不存在")
            print("  请运行: bash download_models.sh")
            print("  将使用演示模式（无真实检测结果）")
            self.net = None
            return

        self.net = cv2.dnn.readNetFromTensorflow(TF_MODEL, TF_PROTO)

        if self._opencv_cuda_available():
            try:
                self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
                self.opencv_target = "CUDA"
                print("[模型] OpenCV DNN 加载成功（CUDA 加速）")
                return
            except Exception as e:
                print(f"[警告] OpenCV CUDA 初始化失败，回退到 CPU: {e}")

        self._set_opencv_cpu_mode()
        print("[模型] OpenCV DNN 加载成功（CPU 模式）")

    def _opencv_cuda_available(self):
        if not hasattr(cv2.dnn, "DNN_BACKEND_CUDA") or not hasattr(cv2.dnn, "DNN_TARGET_CUDA"):
            return False
        try:
            return cv2.cuda.getCudaEnabledDeviceCount() > 0
        except Exception:
            return False

    def _set_opencv_cpu_mode(self):
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        self.opencv_target = "CPU"

    def _ensure_tensorrt_engine(self):
        if os.path.exists(self.engine_path):
            return True

        if not os.path.exists(self.onnx_path):
            print(f"[警告] TensorRT ONNX 模型不存在: {self.onnx_path}")
            print("  请运行: bash download_models.sh")
            return False

        print("[TRT] 未找到引擎文件，开始自动构建...")
        try:
            from build_engine import build_engine
        except ImportError as e:
            print(f"[警告] 无法导入 build_engine.py: {e}")
            return False

        return build_engine(
            onnx_path=self.onnx_path,
            engine_path=self.engine_path,
            precision=self.trt_precision,
        )

    def _load_tensorrt(self):
        if not self._ensure_tensorrt_engine():
            print("[警告] TensorRT 引擎不可用，回退到 OpenCV DNN 后端")
            self.backend = "opencv"
            self._load_opencv_dnn()
            return

        try:
            self.trt_detector = TRTObjectDetector(self.engine_path)
            print("[模型] TensorRT 检测后端已就绪")
        except (ImportError, FileNotFoundError, RuntimeError) as e:
            print(f"[警告] TensorRT 初始化失败，回退到 OpenCV DNN 后端: {e}")
            self.backend = "opencv"
            self._load_opencv_dnn()

    def detect_opencv(self, frame):
        if self.net is None:
            return self._demo_detect(frame)

        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, size=(300, 300), swapRB=True, crop=False)
        self.net.setInput(blob)

        try:
            detections = self.net.forward()
        except cv2.error as e:
            if self.opencv_target == "CUDA":
                if not self._warned_cuda_fallback:
                    print(f"[警告] OpenCV DNN CUDA 推理失败，自动回退到 CPU: {e}")
                    self._warned_cuda_fallback = True
                self._set_opencv_cpu_mode()
                self.net.setInput(blob)
                detections = self.net.forward()
            else:
                raise

        results = []
        for i in range(detections.shape[2]):
            confidence = float(detections[0, 0, i, 2])
            if confidence < self.threshold:
                continue

            class_id = int(detections[0, 0, i, 1])
            class_name = get_class_name(class_id)
            if not class_name or class_name == "background":
                continue

            x1, y1, x2, y2 = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            x1, y1 = max(0, int(x1)), max(0, int(y1))
            x2, y2 = min(w - 1, int(x2)), min(h - 1, int(y2))
            if x2 <= x1 or y2 <= y1:
                continue

            results.append({
                "class_name": class_name,
                "confidence": confidence,
                "bbox": (x1, y1, x2 - x1, y2 - y1),
            })

        return results

    def detect_tensorrt(self, frame):
        if self.trt_detector is None:
            return self.detect_opencv(frame)

        try:
            return self.trt_detector.infer(frame, threshold=self.threshold)
        except Exception as e:
            print(f"[警告] TensorRT 推理失败，回退到 OpenCV DNN: {e}")
            self.backend = "opencv"
            self._load_opencv_dnn()
            return self.detect_opencv(frame)

    def _demo_detect(self, frame):
        return []

    def detect(self, frame):
        if self.backend == "tensorrt":
            return self.detect_tensorrt(frame)
        return self.detect_opencv(frame)

    def draw_results(self, frame, detections):
        class_counts = {}
        for det in detections:
            name = det["class_name"]
            class_counts[name] = class_counts.get(name, 0) + 1

        for det in detections:
            x, y, bw, bh = det["bbox"]
            class_name = det["class_name"]
            confidence = det["confidence"]
            color = CLASS_COLORS.get(class_name, (0, 255, 0))

            cv2.rectangle(frame, (x, y), (x + bw, y + bh), color, 2)

            label = get_label(class_name, self.use_chinese)
            label_text = f"{label}: {confidence:.0%}"
            (tw, th), _ = cv2.getTextSize(
                label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1
            )
            label_y = y - 8 if y > 25 else y + bh + 20
            cv2.rectangle(
                frame,
                (x, label_y - th - 4),
                (x + tw + 4, label_y + 2),
                color,
                -1,
            )
            cv2.putText(
                frame,
                label_text,
                (x + 2, label_y - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 0),
                1,
            )

        self._draw_info_panel(frame, detections, class_counts)
        return frame

    def _draw_info_panel(self, frame, detections, class_counts):
        h, w = frame.shape[:2]

        panel_h = 30 + len(class_counts) * 22 + 20
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (280, max(90, panel_h)), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

        cv2.putText(
            frame, f"FPS: {self.fps:.1f}", (10, 22),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2
        )
        cv2.putText(
            frame, f"Objects: {len(detections)}", (10, 47),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65,
            (0, 200, 255) if detections else (150, 150, 150), 2
        )
        cv2.putText(
            frame, f"Threshold: {self.threshold:.0%}", (10, 72),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1
        )

        y_offset = 95
        for cls_name, count in sorted(class_counts.items(), key=lambda x: -x[1]):
            color = CLASS_COLORS.get(cls_name, (200, 200, 200))
            label = get_label(cls_name, self.use_chinese)
            cv2.putText(
                frame, f"  {label}: {count}", (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1
            )
            y_offset += 22

        tips = ["q:退出", "s:截图", "+/-:阈值"]
        for i, tip in enumerate(tips):
            cv2.putText(
                frame, tip, (w - 100, h - 15 - i * 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1
            )

    def update_fps(self):
        self.frame_count += 1
        if self.frame_count % 30 == 0:
            elapsed = time.time() - self.start_time
            self.fps = self.frame_count / elapsed

    def save_photo(self, frame):
        self.photo_count += 1
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(CAPTURE_DIR, f"detect_{ts}_{self.photo_count:03d}.jpg")
        cv2.imwrite(filename, frame)
        print(f"[截图] 已保存: {filename}")

    def process_image(self, image_path):
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

        backend_desc = self.backend.upper()
        if self.backend == "opencv":
            backend_desc = f"OPENCV-{self.opencv_target}"

        print(f"[检测] 后端: {backend_desc}，置信度阈值: {self.threshold:.0%}")
        print("  按 q 退出，s 截图，+/- 调整阈值\n")

        window_name = "Jetson Object Detection"
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
                if key == ord("q") or key == 27:
                    break
                if key == ord("s"):
                    self.save_photo(result)
                elif key == ord("+") or key == ord("="):
                    self.threshold = min(0.95, self.threshold + 0.05)
                    print(f"[调整] 置信度阈值: {self.threshold:.0%}")
                elif key == ord("-"):
                    self.threshold = max(0.1, self.threshold - 0.05)
                    print(f"[调整] 置信度阈值: {self.threshold:.0%}")

        except KeyboardInterrupt:
            print("\n[中断] Ctrl+C 退出")
        finally:
            cap.release()
            cv2.destroyAllWindows()
            print(f"[统计] 共处理 {self.frame_count} 帧")


def parse_args():
    parser = argparse.ArgumentParser(description="目标检测示例（OpenCV / TensorRT）")
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
    parser.add_argument("--engine", type=str, default=TRT_ENGINE,
                        help="TensorRT 引擎路径")
    parser.add_argument("--onnx", type=str, default=TRT_ONNX,
                        help="TensorRT 后端使用的 ONNX 路径（用于自动构建）")
    parser.add_argument("--trt-precision", choices=["fp32", "fp16", "int8"],
                        default="fp16", help="自动构建 TensorRT 引擎时使用的精度")
    return parser.parse_args()


def main():
    print("=" * 60)
    print("  Jetson SSD MobileNet 目标检测示例")
    print("=" * 60)

    args = parse_args()

    detector = ObjectDetector(
        backend=args.backend,
        threshold=args.threshold,
        use_chinese=args.chinese,
        engine_path=args.engine,
        onnx_path=args.onnx,
        trt_precision=args.trt_precision,
    )

    if args.image:
        detector.process_image(args.image)
    else:
        detector.run_camera(device=args.device, use_csi=args.csi)


if __name__ == "__main__":
    main()
