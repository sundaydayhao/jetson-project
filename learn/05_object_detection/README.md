# 示例 05：目标检测（OpenCV / TensorRT 双后端）

## 功能说明

使用预训练的 SSD MobileNet 模型进行实时目标检测，支持摄像头实时检测和单张图片检测。

- `opencv` 后端：TensorFlow SSD MobileNet V2
- `tensorrt` 后端：ONNX SSD MobileNet V1 + TensorRT 引擎

## 依赖

```bash
# OpenCV 模式
pip3 install numpy opencv-python

# TensorRT 模式额外需要
pip3 install pycuda

# 下载模型
bash download_models.sh
```

说明：

- TensorRT Python 绑定通常随系统安装，不建议直接用 `pip install tensorrt`
- `main.py --backend tensorrt` 首次运行时，如果发现 `models/ssd_mobilenet_v1_12_fp16.trt` 不存在，会自动尝试从 ONNX 构建引擎
- 如需手动构建，可直接运行 `python3 build_engine.py`

## TensorRT 流程

```bash
# 1. 下载 OpenCV / TensorRT 两套模型
bash download_models.sh

# 2. 手动构建 TensorRT 引擎（可选）
python3 build_engine.py

# 3. 使用 TensorRT 后端跑摄像头
python3 main.py --backend tensorrt
```

如果你的输入分辨率较高，也可以自定义 TensorRT optimization profile：

```bash
python3 build_engine.py \
  --min-shape 1x300x300x3 \
  --opt-shape 1x720x1280x3 \
  --max-shape 1x1080x1920x3
```

## 运行方式

```bash
# 默认 OpenCV 后端
python3 main.py

# OpenCV DNN 后端
python3 main.py --backend opencv

# TensorRT 后端
python3 main.py --backend tensorrt

# 检测图片
python3 main.py --image test.jpg

# 指定置信度阈值
python3 main.py --threshold 0.4

# CSI 摄像头
python3 main.py --csi
```

## 键盘控制

| 按键 | 功能 |
|------|------|
| `q` | 退出 |
| `s` | 截图 |
| `+/-` | 调整置信度阈值 |

## 说明

- 如果当前 OpenCV 没有 CUDA DNN 支持，`opencv` 后端会自动回退到 CPU
- 如果 TensorRT 初始化失败，程序会打印原因并自动回退到 OpenCV 后端
- COCO 标签按官方原始类别 ID 映射，避免出现类别名称错位

## 参考来源

- ONNX Model Zoo SSD MobileNet V1: https://huggingface.co/onnxmodelzoo/ssd_mobilenet_v1_12
- TensorFlow SSD MobileNet V2: http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz
