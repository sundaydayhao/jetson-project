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
# PyCUDA（CUDA Python 绑定）
pip3 install pycuda

# NumPy
pip3 install numpy

# OpenCV（图像处理示例）
# JetPack 已预装
```

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