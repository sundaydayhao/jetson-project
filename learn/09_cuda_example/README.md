# 示例 09：CUDA 并行计算加速

## 功能说明

演示如何在 Jetson Nano 上使用 CUDA 进行 GPU 并行计算，包括向量运算、矩阵乘法和图像处理加速，并与 CPU 版本进行性能对比。

## Jetson Nano GPU 规格

- GPU：128 核 Maxwell 架构
- 显存：与系统内存共享（4GB LPDDR4）
- CUDA 计算能力：5.3
- 最大时钟频率：921 MHz

## 示例内容

| 示例 | 说明 |
|------|------|
| 向量加法 | 最简单的 CUDA 并行计算 |
| 矩阵乘法 | 展示 GPU 并行优势 |
| 图像高斯模糊 | CUDA 图像处理 |
| 图像边缘检测 | CUDA Sobel 算子 |

## 依赖安装

```bash
# NumPy
pip3 install numpy

# OpenCV（图像处理示例）
# JetPack 已预装
```

### PyCUDA 安装说明

`pycuda` 依赖 CUDA Toolkit 提供的开发头文件，例如 `cuda.h`。如果安装时报错：

```text
fatal error: cuda.h: No such file or directory
```

说明当前系统没有安装 CUDA Toolkit，或者编译器找不到 CUDA 的 `include` 目录。

在 Jetson 设备上，通常 CUDA 会随 JetPack 安装到 `/usr/local/cuda`，可先检查：

```bash
ls /usr/local/cuda/include/cuda.h
```

如果文件存在，再安装 `pycuda`：

```bash
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export CPATH=$CUDA_HOME/include:$CPATH
export LIBRARY_PATH=$CUDA_HOME/lib64:$LIBRARY_PATH

pip3 install pycuda
```

如果你当前是在普通 `x86_64` Linux 主机上，而不是已安装 JetPack 的 Jetson 设备，那么需要先安装 NVIDIA CUDA Toolkit，否则 `pycuda` 无法编译。

## 运行方式

```bash
# 运行所有示例
python3 main.py

# 只运行向量加法
python3 main.py --demo vector

# 只运行矩阵乘法
python3 main.py --demo matrix

# 图像处理示例
python3 main.py --demo image --input test.jpg

# 性能基准测试
python3 main.py --benchmark
```

## 预期效果

- 显示 GPU 设备信息
- 对比 CPU 和 GPU 计算时间
- GPU 加速比（Speedup）

## 知识点

- CUDA 编程模型（线程、块、网格）
- PyCUDA 基本使用
- 内存管理（主机内存 ↔ 设备内存）
- CUDA 核函数（Kernel）编写
- 性能优化技巧
