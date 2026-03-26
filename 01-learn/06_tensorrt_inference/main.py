#!/usr/bin/env python3
"""
步骤 3：使用 TensorRT 引擎进行图像分类推理
支持单张图片推理和摄像头实时分类
"""

import cv2
import numpy as np
import os
import time
import argparse

# ImageNet 均值和标准差（用于预处理）
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# CSI 摄像头管道
CSI_PIPELINE = (
    "nvarguscamerasrc ! "
    "video/x-raw(memory:NVMM), width=1280, height=720, "
    "format=NV12, framerate=30/1 ! "
    "nvvidconv flip-method=0 ! "
    "video/x-raw, width=640, height=480, format=BGRx ! "
    "videoconvert ! video/x-raw, format=BGR ! appsink"
)


def load_imagenet_labels(label_file="imagenet_labels.txt"):
    """加载 ImageNet 标签"""
    if os.path.exists(label_file):
        with open(label_file) as f:
            return [line.strip() for line in f.readlines()]
    # 返回简单的数字标签
    return [f"class_{i}" for i in range(1000)]


def preprocess(image, input_size=(224, 224)):
    """
    图像预处理（ImageNet 标准化）
    1. 缩放到 224x224
    2. BGR → RGB
    3. 归一化到 [0, 1]
    4. 减均值除标准差
    5. HWC → CHW（通道优先）
    6. 添加 batch 维度
    """
    # 缩放
    img = cv2.resize(image, input_size)
    # BGR → RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # 归一化
    img = img.astype(np.float32) / 255.0
    # 标准化
    img = (img - IMAGENET_MEAN) / IMAGENET_STD
    # HWC → CHW
    img = img.transpose(2, 0, 1)
    # 添加 batch 维度
    img = np.expand_dims(img, axis=0)
    return np.ascontiguousarray(img, dtype=np.float32)


class TRTInference:
    """TensorRT 推理引擎封装"""

    def __init__(self, engine_path):
        self.engine_path = engine_path
        self.engine = None
        self.context = None
        self.inputs = []
        self.outputs = []
        self.bindings = []
        self.stream = None

        self._load_engine()

    def _load_engine(self):
        """加载 TensorRT 引擎"""
        try:
            import tensorrt as trt
            import pycuda.driver as cuda
            import pycuda.autoinit
        except ImportError as e:
            raise ImportError(
                f"缺少依赖: {e}\n"
                "请安装: pip3 install pycuda\n"
                "TensorRT 应随 JetPack 安装"
            )

        if not os.path.exists(self.engine_path):
            raise FileNotFoundError(
                f"引擎文件不存在: {self.engine_path}\n"
                "请先运行: python3 build_engine.py"
            )

        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        print(f"[TRT] 加载引擎: {self.engine_path}")

        with open(self.engine_path, "rb") as f, \
             trt.Runtime(TRT_LOGGER) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())

        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()

        # 分配 CUDA 内存
        for binding in self.engine:
            size = trt.volume(self.engine.get_binding_shape(binding))
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            # 分配主机内存（页锁定）
            host_mem = cuda.pagelocked_empty(size, dtype)
            # 分配设备内存
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            self.bindings.append(int(device_mem))

            if self.engine.binding_is_input(binding):
                self.inputs.append({"host": host_mem, "device": device_mem})
                print(f"[TRT] 输入: {binding} {self.engine.get_binding_shape(binding)}")
            else:
                self.outputs.append({"host": host_mem, "device": device_mem})
                print(f"[TRT] 输出: {binding} {self.engine.get_binding_shape(binding)}")

        print(f"[TRT] 引擎加载成功")

    def infer(self, input_data):
        """
        执行推理
        :param input_data: 预处理后的输入数据（numpy array）
        :return: 输出数据列表
        """
        import pycuda.driver as cuda

        # 复制输入数据到主机内存
        np.copyto(self.inputs[0]["host"], input_data.ravel())

        # 主机 → 设备（异步）
        cuda.memcpy_htod_async(
            self.inputs[0]["device"],
            self.inputs[0]["host"],
            self.stream
        )

        # 执行推理
        self.context.execute_async_v2(
            bindings=self.bindings,
            stream_handle=self.stream.handle
        )

        # 设备 → 主机（异步）
        for out in self.outputs:
            cuda.memcpy_dtoh_async(out["host"], out["device"], self.stream)

        # 同步等待完成
        self.stream.synchronize()

        return [out["host"] for out in self.outputs]

    def classify(self, image, top_k=5):
        """
        图像分类
        :param image: BGR 图像（numpy array）
        :param top_k: 返回 Top-K 结果
        :return: [(class_id, score), ...]
        """
        # 预处理
        input_data = preprocess(image)

        # 推理
        t0 = time.time()
        outputs = self.infer(input_data)
        elapsed_ms = (time.time() - t0) * 1000

        # 后处理：Softmax
        logits = outputs[0]
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / exp_logits.sum()

        # Top-K
        top_indices = np.argsort(probs)[::-1][:top_k]
        results = [(int(idx), float(probs[idx])) for idx in top_indices]

        return results, elapsed_ms


def draw_classification_result(frame, results, labels, elapsed_ms, fps=0):
    """在图像上绘制分类结果"""
    h, w = frame.shape[:2]

    # 半透明背景
    panel_h = 30 + len(results) * 28 + 20
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (400, panel_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    # FPS 和推理时间
    cv2.putText(frame, f"FPS: {fps:.1f}  Infer: {elapsed_ms:.1f}ms",
                (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Top-K 结果
    for i, (class_id, score) in enumerate(results):
        label = labels[class_id] if class_id < len(labels) else f"class_{class_id}"
        # 截断过长的标签
        if len(label) > 35:
            label = label[:32] + "..."

        y = 50 + i * 28
        # 置信度条
        bar_width = int(score * 200)
        bar_color = (0, 255, 0) if i == 0 else (0, 200, 100)
        cv2.rectangle(frame, (10, y - 14), (10 + bar_width, y + 2), bar_color, -1)

        # 标签和分数
        text = f"#{i+1} {label}: {score:.1%}"
        color = (255, 255, 0) if i == 0 else (200, 200, 200)
        cv2.putText(frame, text, (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    return frame


def run_image(engine_path, image_path, top_k=5):
    """单张图片推理"""
    labels = load_imagenet_labels()

    try:
        trt_infer = TRTInference(engine_path)
    except (ImportError, FileNotFoundError) as e:
        print(f"[错误] {e}")
        return

    frame = cv2.imread(image_path)
    if frame is None:
        print(f"[错误] 无法读取图片: {image_path}")
        return

    print(f"\n[推理] 图片: {image_path}")
    results, elapsed_ms = trt_infer.classify(frame, top_k=top_k)

    print(f"[结果] 推理耗时: {elapsed_ms:.2f} ms")
    print(f"[结果] Top-{top_k} 分类结果:")
    for i, (class_id, score) in enumerate(results):
        label = labels[class_id] if class_id < len(labels) else f"class_{class_id}"
        print(f"  #{i+1} [{class_id:4d}] {label}: {score:.4f} ({score:.1%})")

    # 显示结果
    result_frame = draw_classification_result(frame.copy(), results, labels, elapsed_ms)
    cv2.imshow("TensorRT Classification", result_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def run_camera(engine_path, device=0, use_csi=False, top_k=3):
    """摄像头实时分类"""
    labels = load_imagenet_labels()

    try:
        trt_infer = TRTInference(engine_path)
    except (ImportError, FileNotFoundError) as e:
        print(f"[错误] {e}")
        return

    if use_csi:
        cap = cv2.VideoCapture(CSI_PIPELINE, cv2.CAP_GSTREAMER)
        print("[摄像头] 使用 CSI 摄像头")
    else:
        cap = cv2.VideoCapture(device)
        print(f"[摄像头] 使用 USB 摄像头，设备号: {device}")

    if not cap.isOpened():
        print("[错误] 无法打开摄像头！")
        return

    print("[运行] 实时分类（按 q 退出）\n")

    window_name = "Jetson Nano TensorRT Classification"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    frame_count = 0
    fps = 0.0
    start_time = time.time()
    last_results = []
    last_elapsed = 0.0
    infer_interval = 5  # 每 5 帧推理一次（降低延迟）

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            frame_count += 1
            if frame_count % 30 == 0:
                fps = frame_count / (time.time() - start_time)

            # 每隔几帧推理一次
            if frame_count % infer_interval == 0:
                last_results, last_elapsed = trt_infer.classify(frame, top_k=top_k)

            # 绘制结果
            display = draw_classification_result(
                frame.copy(), last_results, labels, last_elapsed, fps
            )
            cv2.imshow(window_name, display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break

    except KeyboardInterrupt:
        print("\n[中断] Ctrl+C 退出")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print(f"[统计] 共处理 {frame_count} 帧，平均 FPS: {fps:.1f}")


def benchmark(engine_path, iterations=100):
    """性能基准测试"""
    try:
        trt_infer = TRTInference(engine_path)
    except (ImportError, FileNotFoundError) as e:
        print(f"[错误] {e}")
        return

    print(f"\n[基准测试] 运行 {iterations} 次推理...")
    dummy_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    # 预热
    for _ in range(10):
        trt_infer.classify(dummy_image)

    # 正式测试
    times = []
    for i in range(iterations):
        _, elapsed = trt_infer.classify(dummy_image)
        times.append(elapsed)
        if (i + 1) % 20 == 0:
            print(f"  进度: {i+1}/{iterations}")

    times = np.array(times)
    print(f"\n[结果] 推理性能统计（{iterations} 次）:")
    print(f"  平均延迟: {times.mean():.2f} ms")
    print(f"  最小延迟: {times.min():.2f} ms")
    print(f"  最大延迟: {times.max():.2f} ms")
    print(f"  标准差:   {times.std():.2f} ms")
    print(f"  吞吐量:   {1000 / times.mean():.1f} FPS")


def main():
    parser = argparse.ArgumentParser(description="TensorRT 图像分类推理")
    parser.add_argument("--engine", type=str, default="models/resnet50_fp16.trt",
                        help="TensorRT 引擎路径")
    parser.add_argument("--image", type=str, default=None,
                        help="输入图片路径")
    parser.add_argument("--camera", action="store_true",
                        help="使用摄像头实时分类")
    parser.add_argument("--csi", action="store_true",
                        help="使用 CSI 摄像头")
    parser.add_argument("--device", type=int, default=0,
                        help="USB 摄像头设备号")
    parser.add_argument("--topk", type=int, default=5,
                        help="显示 Top-K 结果（默认: 5）")
    parser.add_argument("--benchmark", action="store_true",
                        help="运行性能基准测试")
    parser.add_argument("--iterations", type=int, default=100,
                        help="基准测试迭代次数")
    args = parser.parse_args()

    print("=" * 55)
    print("  Jetson Nano TensorRT 推理示例")
    print("=" * 55)
    print(f"  引擎: {args.engine}")

    if args.benchmark:
        benchmark(args.engine, args.iterations)
    elif args.image:
        run_image(args.engine, args.image, top_k=args.topk)
    elif args.camera:
        run_camera(args.engine, device=args.device, use_csi=args.csi, top_k=args.topk)
    else:
        print("\n[提示] 请指定运行模式:")
        print("  图片推理: python3 main.py --engine <engine> --image <image>")
        print("  摄像头:   python3 main.py --engine <engine> --camera")
        print("  基准测试: python3 main.py --engine <engine> --benchmark")


if __name__ == "__main__":
    main()