# 示例 03：摄像头图像采集与显示

## 功能说明

使用 OpenCV 采集摄像头图像，支持 USB 摄像头和 CSI 摄像头（Raspberry Pi Camera v2），演示图像采集、显示、保存和基本处理。

## 支持的摄像头

### USB 摄像头
- 任意 UVC 兼容的 USB 摄像头
- 连接到 Jetson Nano USB 口即可使用

### CSI 摄像头（推荐）
- Raspberry Pi Camera Module v2（Sony IMX219）
- 连接到 Jetson Nano CSI 接口（MIPI CSI-2）
- 需要使用 GStreamer 管道

## 依赖

Jetson Nano 的 JetPack 已预装 OpenCV，无需额外安装。

验证 OpenCV 安装：
```bash
python3 -c "import cv2; print(cv2.__version__)"
```

## 运行方式

```bash
# USB 摄像头（默认设备 /dev/video0）
python3 main.py

# 指定 USB 摄像头设备号
python3 main.py --device 1

# CSI 摄像头
python3 main.py --csi

# 保存视频
python3 main.py --save output.avi

# 拍照模式（按空格键拍照）
python3 main.py --photo
```

## 键盘控制

| 按键 | 功能 |
|------|------|
| `q` | 退出程序 |
| `s` / 空格 | 保存当前帧为图片 |
| `f` | 切换全屏 |
| `g` | 切换灰度/彩色显示 |
| `h` | 切换水平翻转 |

## 预期效果

- 打开摄像头预览窗口
- 实时显示帧率（FPS）
- 按键控制截图和退出

## 知识点

- OpenCV VideoCapture 使用
- GStreamer 管道（CSI 摄像头）
- 实时 FPS 计算
- 图像翻转、灰度转换
- 视频录制（VideoWriter）