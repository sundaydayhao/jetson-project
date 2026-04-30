#!/usr/bin/env python3
"""
示例 03：摄像头图像采集与显示
支持 USB 摄像头和 CSI 摄像头（通过 GStreamer）
功能：实时预览、截图、录像、图像处理
"""

import cv2
import numpy as np
import time
import os
import argparse
from datetime import datetime


# ── GStreamer 管道配置（CSI 摄像头）────────────────────────────────────────────
def get_csi_pipeline(
    capture_width=1280,
    capture_height=720,
    display_width=1280,
    display_height=720,
    framerate=30,
    flip_method=0
):
    """
    生成 CSI 摄像头的 GStreamer 管道字符串
    flip_method: 0=不翻转, 1=逆时针90°, 2=旋转180°, 3=顺时针90°
                 4=水平翻转, 5=上下翻转+逆时针90°, 6=水平翻转+顺时针90°, 7=垂直翻转
    """
    return (
        f"nvarguscamerasrc ! "
        f"video/x-raw(memory:NVMM), "
        f"width=(int){capture_width}, height=(int){capture_height}, "
        f"format=(string)NV12, framerate=(fraction){framerate}/1 ! "
        f"nvvidconv flip-method={flip_method} ! "
        f"video/x-raw, width=(int){display_width}, height=(int){display_height}, "
        f"format=(string)BGRx ! "
        f"videoconvert ! "
        f"video/x-raw, format=(string)BGR ! appsink"
    )


class CameraCapture:
    """摄像头采集类"""

    def __init__(self, device=0, use_csi=False, save_video=None):
        self.device = device
        self.use_csi = use_csi
        self.save_video = save_video
        self.cap = None
        self.writer = None
        self.frame_count = 0
        self.fps = 0
        self.start_time = time.time()
        self.photo_count = 0

        # 显示选项
        self.show_gray = False
        self.flip_horizontal = False
        self.fullscreen = False

        # 创建保存目录
        os.makedirs("captures", exist_ok=True)

    def open(self):
        """打开摄像头"""
        if self.use_csi:
            pipeline = get_csi_pipeline()
            print(f"[摄像头] 使用 CSI 摄像头")
            print(f"[GStreamer] {pipeline}")
            self.cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        else:
            print(f"[摄像头] 使用 USB 摄像头，设备号: {self.device}")
            self.cap = cv2.VideoCapture(self.device)
            # 设置分辨率
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            self.cap.set(cv2.CAP_PROP_FPS, 30)

        if not self.cap.isOpened():
            raise RuntimeError(f"[错误] 无法打开摄像头！")

        # 获取实际参数
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        print(f"[摄像头] 分辨率: {width}x{height}，帧率: {fps:.1f} FPS")

        # 初始化视频录制
        if self.save_video:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            self.writer = cv2.VideoWriter(
                self.save_video, fourcc, fps, (width, height)
            )
            print(f"[录制] 视频将保存到: {self.save_video}")

        return width, height

    def read_frame(self):
        """读取一帧"""
        ret, frame = self.cap.read()
        if not ret:
            return None

        self.frame_count += 1

        # 计算 FPS（每 30 帧更新一次）
        if self.frame_count % 30 == 0:
            elapsed = time.time() - self.start_time
            self.fps = self.frame_count / elapsed

        return frame

    def process_frame(self, frame):
        """处理帧（应用显示选项）"""
        # 水平翻转
        if self.flip_horizontal:
            frame = cv2.flip(frame, 1)

        # 灰度显示
        if self.show_gray:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        return frame

    def draw_overlay(self, frame):
        """在帧上绘制信息叠加层"""
        h, w = frame.shape[:2]

        # 绘制半透明背景
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (300, 80), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)

        # 显示 FPS
        cv2.putText(
            frame, f"FPS: {self.fps:.1f}",
            (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
        )

        # 显示帧数
        cv2.putText(
            frame, f"Frame: {self.frame_count}",
            (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
        )

        # 显示时间戳
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(
            frame, timestamp,
            (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1
        )

        # 显示模式标志
        flags = []
        if self.show_gray:
            flags.append("GRAY")
        if self.flip_horizontal:
            flags.append("FLIP")
        if self.save_video:
            flags.append("REC")
        if flags:
            flag_text = " | ".join(flags)
            cv2.putText(
                frame, flag_text,
                (w - 200, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2
            )

        # 操作提示
        help_text = [
            "q: 退出",
            "s/空格: 截图",
            "g: 灰度",
            "h: 翻转",
            "f: 全屏",
        ]
        for i, text in enumerate(help_text):
            cv2.putText(
                frame, text,
                (w - 120, h - 20 - i * 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1
            )

        return frame

    def save_photo(self, frame):
        """保存截图"""
        self.photo_count += 1
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"captures/photo_{timestamp}_{self.photo_count:03d}.jpg"
        cv2.imwrite(filename, frame)
        print(f"[截图] 已保存: {filename}")
        return filename

    def run(self):
        """主循环"""
        width, height = self.open()

        window_name = "Jetson Nano Camera"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, min(width, 1280), min(height, 720))

        print("\n[运行] 摄像头预览已启动")
        print("  按 q 退出，s/空格 截图，g 灰度，h 翻转，f 全屏\n")

        try:
            while True:
                frame = self.read_frame()
                if frame is None:
                    print("[警告] 读取帧失败，重试...")
                    continue

                # 处理帧
                processed = self.process_frame(frame.copy())

                # 录制原始帧
                if self.writer:
                    self.writer.write(processed)

                # 绘制叠加信息
                display = self.draw_overlay(processed)

                # 显示
                cv2.imshow(window_name, display)

                # 键盘处理
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:  # q 或 ESC
                    print("[退出] 用户按下退出键")
                    break
                elif key == ord('s') or key == 32:  # s 或 空格
                    self.save_photo(processed)
                elif key == ord('g'):
                    self.show_gray = not self.show_gray
                    print(f"[切换] 灰度模式: {'开' if self.show_gray else '关'}")
                elif key == ord('h'):
                    self.flip_horizontal = not self.flip_horizontal
                    print(f"[切换] 水平翻转: {'开' if self.flip_horizontal else '关'}")
                elif key == ord('f'):
                    self.fullscreen = not self.fullscreen
                    if self.fullscreen:
                        cv2.setWindowProperty(
                            window_name,
                            cv2.WND_PROP_FULLSCREEN,
                            cv2.WINDOW_FULLSCREEN
                        )
                    else:
                        cv2.setWindowProperty(
                            window_name,
                            cv2.WND_PROP_FULLSCREEN,
                            cv2.WINDOW_NORMAL
                        )

        except KeyboardInterrupt:
            print("\n[中断] Ctrl+C 退出")
        finally:
            self.close()

    def close(self):
        """释放资源"""
        if self.cap:
            self.cap.release()
        if self.writer:
            self.writer.release()
        cv2.destroyAllWindows()
        print(f"[统计] 共采集 {self.frame_count} 帧，截图 {self.photo_count} 张")
        print("[清理] 资源已释放")


def parse_args():
    parser = argparse.ArgumentParser(description="Jetson Nano 摄像头采集示例")
    parser.add_argument(
        "--device", type=int, default=0,
        help="USB 摄像头设备号（默认: 0）"
    )
    parser.add_argument(
        "--csi", action="store_true",
        help="使用 CSI 摄像头（Raspberry Pi Camera）"
    )
    parser.add_argument(
        "--save", type=str, default=None,
        help="保存视频文件路径（例如: output.avi）"
    )
    return parser.parse_args()


def main():
    print("=" * 50)
    print("  Jetson Nano 摄像头采集示例")
    print("=" * 50)

    args = parse_args()

    camera = CameraCapture(
        device=args.device,
        use_csi=args.csi,
        save_video=args.save
    )
    camera.run()


if __name__ == "__main__":
    main()