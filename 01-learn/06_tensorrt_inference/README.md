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
# PyTorch（用于导出 ONNX，可选）
# 从 NVIDIA 官方下载适合 JetPack 版本的 wheel
pip3 install torch torchvision

# ONNX
pip3 install onnx onnxruntime
```

### PyCUDA 安装说明

`pycuda` 依赖 CUDA Toolkit 的开发头文件。若安装时报错：

```text
fatal error: cuda.h: No such file or directory
```

说明当前系统缺少 CUDA Toolkit，或者 Python 构建过程没有找到 `/usr/local/cuda/include`。

Jetson 设备上可先确认：

```bash
ls /usr/local/cuda/include/cuda.h
```

若存在，再执行：

```bash
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export CPATH=$CUDA_HOME/include:$CPATH
export LIBRARY_PATH=$CUDA_HOME/lib64:$LIBRARY_PATH

pip3 install pycuda
```

如果是在没有安装 CUDA Toolkit 的 `x86_64` 主机上，请先安装 CUDA，再继续运行本示例。

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
