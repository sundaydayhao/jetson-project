#!/usr/bin/env python3
"""
示例 04：OpenCV 人脸检测
支持 Haar 级联分类器和 DNN 两种检测方法
支持摄像头实时检测和图片文件检测
"""

import cv2
import numpy as np
import os
import time
import argparse
from datetime import datetime


# ── 模型路径配置 ───────────────────────────────────────────────────────────────
# Haar 级联分类器（OpenCV 自带）
HAAR_FACE_CASCADE = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
HAAR_EYE_CASCADE = cv2.data.haarcascades + "haarcascade_eye.xml"
HAAR_SMILE_CASCADE = cv2.data.haarcascades + "haarcascade_smile.xml"

# DNN 模型文件（需要下载，见 download_models.sh）
DNN_PROTO = "models/deploy.prototxt"
DNN_MODEL = "models/res10_300x300_ssd_iter_140000.caffemodel"

# CSI 摄像头 GStreamer 管道
CSI_PIPELINE = (
    "nvarguscamerasrc ! "
    "video/x-raw(memory:NVMM), width=1280, height=720, "
    "format=NV12, framerate=30/1 ! "
    "nvvidconv flip-method=0 ! "
    "video/x-raw, width=1280, height=720, format=BGRx ! "
    "videoconvert ! video/x-raw, format=BGR ! appsink"
)


class FaceDetector:
    """人脸检测器"""

    def __init__(self, method="haar", use_eyes=False, use_smile=False):
        self.method = method
        self.use_eyes = use_eyes
        self.use_smile = use_smile
        self.net = None
        self.face_cascade = None
        self.eye_cascade = None
        self.smile_cascade = None
        self.frame_count = 0
        self.fps = 0.0
        self.start_time = time.time()
        self.photo_count = 0

        os.makedirs("captures", exist_ok=True)
        self._load_models()

    def _load_models(self):
        """加载检测模型"""
        # 加载 Haar 级联分类器
        if not os.path.exists(HAAR_FACE_CASCADE):
            raise FileNotFoundError(f"Haar 模型文件不存在: {HAAR_FACE_CASCADE}")

        self.face_cascade = cv2.CascadeClassifier(HAAR_FACE_CASCADE)
        self.eye_cascade = cv2.CascadeClassifier(HAAR_EYE_CASCADE)
        self.smile_cascade = cv2.CascadeClassifier(HAAR_SMILE_CASCADE)
        print("[模型] Haar 级联分类器加载成功")

        # 加载 DNN 模型
        if self.method == "dnn":
            if os.path.exists(DNN_PROTO) and os.path.exists(DNN_MODEL):
                self.net = cv2.dnn.readNetFromCaffe(DNN_PROTO, DNN_MODEL)
                # 尝试使用 CUDA 加速
                try:
                    self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                    self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
                    print("[模型] DNN 模型加载成功（CUDA 加速）")
                except Exception:
                    self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                    self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
                    print("[模型] DNN 模型加载成功（CPU 模式）")
            else:
                print(f"[警告] DNN 模型文件不存在，回退到 Haar 方法")
                print(f"  请运行: bash download_models.sh")
                self.method = "haar"

    def detect_haar(self, frame):
        """
        使用 Haar 级联分类器检测人脸
        返回：[(x, y, w, h, confidence), ...]
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # 直方图均衡化，提升低光照下的检测效果
        gray = cv2.equalizeHist(gray)

        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,    # 每次缩放比例（越小越慢但越准）
            minNeighbors=5,     # 最小邻居数（越大误检越少）
            minSize=(30, 30),   # 最小人脸尺寸
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        results = []
        for (x, y, w, h) in faces if len(faces) > 0 else []:
            results.append((x, y, w, h, 1.0))  # Haar 无置信度，设为 1.0

            # 在人脸区域内检测眼睛
            if self.use_eyes:
                roi_gray = gray[y:y+h, x:x+w]
                eyes = self.eye_cascade.detectMultiScale(
                    roi_gray, scaleFactor=1.1, minNeighbors=10, minSize=(15, 15)
                )
                for (ex, ey, ew, eh) in eyes:
                    cv2.rectangle(
                        frame,
                        (x + ex, y + ey),
                        (x + ex + ew, y + ey + eh),
                        (0, 255, 255), 2
                    )

            # 检测微笑
            if self.use_smile:
                roi_gray = gray[y + h//2:y+h, x:x+w]
                smiles = self.smile_cascade.detectMultiScale(
                    roi_gray, scaleFactor=1.7, minNeighbors=22, minSize=(25, 25)
                )
                for (sx, sy, sw, sh) in smiles:
                    cv2.rectangle(
                        frame,
                        (x + sx, y + h//2 + sy),
                        (x + sx + sw, y + h//2 + sy + sh),
                        (255, 0, 255), 2
                    )

        return results

    def detect_dnn(self, frame):
        """
        使用 DNN 模型检测人脸
        返回：[(x, y, w, h, confidence), ...]
        """
        h, w = frame.shape[:2]

        # 预处理：缩放到 300x300，减均值
        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)),
            scalefactor=1.0,
            size=(300, 300),
            mean=(104.0, 177.0, 123.0)
        )
        self.net.setInput(blob)
        detections = self.net.forward()

        results = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:  # 置信度阈值
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                x1, y1, x2, y2 = box.astype(int)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                results.append((x1, y1, x2 - x1, y2 - y1, float(confidence)))

        return results

    def detect(self, frame):
        """根据当前方法检测人脸"""
        if self.method == "dnn" and self.net is not None:
            return self.detect_dnn(frame)
        else:
            return self.detect_haar(frame)

    def draw_results(self, frame, faces):
        """在帧上绘制检测结果"""
        h, w = frame.shape[:2]

        for i, (x, y, fw, fh, conf) in enumerate(faces):
            # 绘制人脸矩形框
            color = (0, 255, 0)  # 绿色
            cv2.rectangle(frame, (x, y), (x + fw, y + fh), color, 2)

            # 绘制角标（更美观）
            corner_len = min(fw, fh) // 5
            thickness = 3
            # 左上角
            cv2.line(frame, (x, y), (x + corner_len, y), color, thickness)
            cv2.line(frame, (x, y), (x, y + corner_len), color, thickness)
            # 右上角
            cv2.line(frame, (x + fw, y), (x + fw - corner_len, y), color, thickness)
            cv2.line(frame, (x + fw, y), (x + fw, y + corner_len), color, thickness)
            # 左下角
            cv2.line(frame, (x, y + fh), (x + corner_len, y + fh), color, thickness)
            cv2.line(frame, (x, y + fh), (x, y + fh - corner_len), color, thickness)
            # 右下角
            cv2.line(frame, (x + fw, y + fh), (x + fw - corner_len, y + fh), color, thickness)
            cv2.line(frame, (x + fw, y + fh), (x + fw, y + fh - corner_len), color, thickness)

            # 显示置信度标签
            label = f"Face {i+1}"
            if conf < 1.0:
                label += f" {conf:.0%}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            label_y = y - 10 if y > 20 else y + fh + 20
            cv2.rectangle(
                frame,
                (x, label_y - label_size[1] - 4),
                (x + label_size[0] + 4, label_y + 2),
                color, -1
            )
            cv2.putText(
                frame, label,
                (x + 2, label_y - 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1
            )

        # 绘制信息面板
        self._draw_info_panel(frame, len(faces))
        return frame

    def _draw_info_panel(self, frame, face_count):
        """绘制信息面板"""
        h, w = frame.shape[:2]

        # 半透明背景
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (280, 90), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

        method_name = "DNN" if self.method == "dnn" else "Haar"
        cv2.putText(frame, f"FPS: {self.fps:.1f}", (10, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
        cv2.putText(frame, f"Method: {method_name}", (10, 47),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
        cv2.putText(frame, f"Faces: {face_count}", (10, 72),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                    (0, 0, 255) if face_count > 0 else (200, 200, 200), 2)

        # 帮助提示
        tips = ["q:退出", "s:截图", "m:切换方法", "e:眼睛检测"]
        for i, tip in enumerate(tips):
            cv2.putText(frame, tip, (w - 110, h - 15 - i * 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)

    def update_fps(self):
        """更新 FPS 计数"""
        self.frame_count += 1
        if self.frame_count % 30 == 0:
            elapsed = time.time() - self.start_time
            self.fps = self.frame_count / elapsed

    def save_photo(self, frame):
        """保存截图"""
        self.photo_count += 1
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"captures/face_{ts}_{self.photo_count:03d}.jpg"
        cv2.imwrite(filename, frame)
        print(f"[截图] 已保存: {filename}")

    def process_image(self, image_path):
        """处理单张图片"""
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"[错误] 无法读取图片: {image_path}")
            return

        print(f"[图片] 处理: {image_path}，尺寸: {frame.shape[1]}x{frame.shape[0]}")

        t0 = time.time()
        faces = self.detect(frame)
        elapsed = (time.time() - t0) * 1000

        print(f"[结果] 检测到 {len(faces)} 张人脸，耗时 {elapsed:.1f} ms")
        for i, (x, y, w, h, conf) in enumerate(faces):
            print(f"  人脸 {i+1}: 位置=({x},{y})，大小={w}x{h}，置信度={conf:.2f}")

        result = self.draw_results(frame, faces)
        cv2.imshow("Face Detection Result", result)
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

        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"[摄像头] 分辨率: {w}x{h}")
        print(f"[检测] 使用方法: {self.method.upper()}")
        print("  按 q 退出，s 截图，m 切换方法，e 切换眼睛检测\n")

        window_name = "Jetson Nano Face Detection"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    continue

                self.update_fps()

                # 检测人脸
                faces = self.detect(frame)

                # 绘制结果
                result = self.draw_results(frame, faces)
                cv2.imshow(window_name, result)

                # 键盘处理
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:
                    break
                elif key == ord('s'):
                    self.save_photo(result)
                elif key == ord('m'):
                    self.method = "dnn" if self.method == "haar" else "haar"
                    print(f"[切换] 检测方法: {self.method.upper()}")
                    if self.method == "dnn" and self.net is None:
                        print("[警告] DNN 模型未加载，请运行 download_models.sh")
                        self.method = "haar"
                elif key == ord('e'):
                    self.use_eyes = not self.use_eyes
                    print(f"[切换] 眼睛检测: {'开' if self.use_eyes else '关'}")

        except KeyboardInterrupt:
            print("\n[中断] Ctrl+C 退出")
        finally:
            cap.release()
            cv2.destroyAllWindows()
            print(f"[统计] 共处理 {self.frame_count} 帧")


def parse_args():
    parser = argparse.ArgumentParser(description="Jetson Nano 人脸检测示例")
    parser.add_argument("--method", choices=["haar", "dnn"], default="haar",
                        help="检测方法（默认: haar）")
    parser.add_argument("--image", type=str, default=None,
                        help="检测图片文件路径")
    parser.add_argument("--device", type=int, default=0,
                        help="USB 摄像头设备号（默认: 0）")
    parser.add_argument("--csi", action="store_true",
                        help="使用 CSI 摄像头")
    parser.add_argument("--eyes", action="store_true",
                        help="同时检测眼睛")
    parser.add_argument("--smile", action="store_true",
                        help="同时检测微笑")
    return parser.parse_args()


def main():
    print("=" * 50)
    print("  Jetson Nano 人脸检测示例")
    print("=" * 50)

    args = parse_args()

    detector = FaceDetector(
        method=args.method,
        use_eyes=args.eyes,
        use_smile=args.smile
    )

    if args.image:
        detector.process_image(args.image)
    else:
        detector.run_camera(device=args.device, use_csi=args.csi)


if __name__ == "__main__":
    main()