# 示例 06：TensorRT 模型推理

## 功能说明

演示如何将 ONNX 模型转换为 TensorRT 引擎，并在 Jetson Nano 上进行高效推理。以 ResNet-50 图像分类为例，展示完整的 TensorRT 工作流程。

## TensorRT 工作流程

```
PyTorch/TensorFlow 模型
        ↓
    导出 ONNX 格式
        ↓
  TensorRT 解析 ONNX
        ↓
  构建优化引擎（.trt）
        ↓
    加载引擎推理
```

## 文件说明

| 文件 | 说明 |
|------|------|
| `export_onnx.py` | 将 PyTorch ResNet-50 导出为 ONNX |
| `build_engine.py` | 将 ONNX 转换为 TensorRT 引擎 |
| `main.py` | 使用 TensorRT 引擎进行推理 |
| `imagenet_labels.py` | ImageNet 1000 类标签 |

## 依赖安装

```bash
# TensorRT（JetPack 已包含）
# PyCUDA
pip3 install pycuda

# PyTorch（用于导出 ONNX，可选）
# 从 NVIDIA 官方下载适合 JetPack 版本的 wheel
pip3 install torch torchvision

# ONNX
pip3 install onnx onnxruntime
```

## 使用步骤

```bash
# 步骤 1：导出 ONNX 模型（需要 PyTorch）
python3 export_onnx.py

# 步骤 2：构建 TensorRT 引擎（首次运行较慢，约 5-10 分钟）
python3 build_engine.py --onnx models/resnet50.onnx --engine models/resnet50.trt

# 步骤 3：使用引擎推理
python3 main.py --engine models/resnet50.trt --image test.jpg

# 步骤 4：摄像头实时分类
python3 main.py --engine models/resnet50.trt --camera
```

## 预期效果

- 显示图片的 Top-5 分类结果
- 显示每次推理耗时
- 对比 CPU 和 TensorRT 推理速度

## 性能参考（Jetson Nano）

| 模型 | 精度 | CPU 推理 | TensorRT FP16 |
|------|------|----------|---------------|
| ResNet-50 | FP32 | ~200ms | ~15ms |
| MobileNet V2 | FP32 | ~50ms | ~5ms |

## 知识点

- ONNX 模型格式
- TensorRT 引擎构建（FP32/FP16/INT8）
- CUDA 内存管理
- 批量推理（Batch Inference）
- 动态形状（Dynamic Shape）