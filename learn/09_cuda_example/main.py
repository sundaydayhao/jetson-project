#!/usr/bin/env python3
"""
示例 09：CUDA 并行计算加速
演示 PyCUDA 基本用法：向量加法、矩阵乘法、图像处理
并与 CPU 版本进行性能对比
"""

import numpy as np
import time
import argparse
import os


# ── GPU 设备信息 ───────────────────────────────────────────────────────────────
def print_gpu_info():
    """打印 GPU 设备信息"""
    try:
        import pycuda.driver as cuda
        import pycuda.autoinit

        print("\n[GPU 信息]")
        device = cuda.Device(0)
        attrs = device.get_attributes()

        print(f"  设备名称:     {device.name()}")
        print(f"  计算能力:     {device.compute_capability()}")
        print(f"  显存总量:     {device.total_memory() // 1024 // 1024} MB")
        print(f"  多处理器数:   {attrs[cuda.device_attribute.MULTIPROCESSOR_COUNT]}")
        print(f"  最大线程/块:  {attrs[cuda.device_attribute.MAX_THREADS_PER_BLOCK]}")
        print(f"  最大块尺寸:   "
              f"{attrs[cuda.device_attribute.MAX_BLOCK_DIM_X]} x "
              f"{attrs[cuda.device_attribute.MAX_BLOCK_DIM_Y]} x "
              f"{attrs[cuda.device_attribute.MAX_BLOCK_DIM_Z]}")
        print(f"  最大网格尺寸: "
              f"{attrs[cuda.device_attribute.MAX_GRID_DIM_X]} x "
              f"{attrs[cuda.device_attribute.MAX_GRID_DIM_Y]}")
        print(f"  共享内存/块:  {attrs[cuda.device_attribute.MAX_SHARED_MEMORY_PER_BLOCK] // 1024} KB")
        print(f"  时钟频率:     {attrs[cuda.device_attribute.CLOCK_RATE] // 1000} MHz")
        return True
    except ImportError:
        print("[错误] PyCUDA 未安装，请运行: pip3 install pycuda")
        return False
    except Exception as e:
        print(f"[错误] 获取 GPU 信息失败: {e}")
        return False


# ── 示例 1：向量加法 ───────────────────────────────────────────────────────────
VECTOR_ADD_KERNEL = """
__global__ void vector_add(float *a, float *b, float *c, int n)
{
    // 计算全局线程索引
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // 边界检查
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}
"""


def demo_vector_add(n=1_000_000):
    """
    向量加法：c = a + b
    演示最基本的 CUDA 并行计算
    """
    print(f"\n{'='*55}")
    print(f"  示例 1：向量加法（N = {n:,}）")
    print(f"{'='*55}")

    # 生成随机数据
    a = np.random.randn(n).astype(np.float32)
    b = np.random.randn(n).astype(np.float32)

    # ── CPU 版本 ──
    t0 = time.time()
    c_cpu = a + b
    cpu_time = (time.time() - t0) * 1000
    print(f"\n[CPU] 耗时: {cpu_time:.3f} ms")

    # ── GPU 版本 ──
    try:
        import pycuda.driver as cuda
        import pycuda.autoinit
        from pycuda.compiler import SourceModule

        # 编译 CUDA 核函数
        mod = SourceModule(VECTOR_ADD_KERNEL)
        vector_add = mod.get_function("vector_add")

        # 分配 GPU 内存并传输数据
        t0 = time.time()
        a_gpu = cuda.mem_alloc(a.nbytes)
        b_gpu = cuda.mem_alloc(b.nbytes)
        c_gpu = cuda.mem_alloc(a.nbytes)

        cuda.memcpy_htod(a_gpu, a)
        cuda.memcpy_htod(b_gpu, b)

        # 配置线程块和网格
        block_size = 256
        grid_size = (n + block_size - 1) // block_size

        # 执行核函数
        vector_add(
            a_gpu, b_gpu, c_gpu, np.int32(n),
            block=(block_size, 1, 1),
            grid=(grid_size, 1)
        )

        # 取回结果
        c_gpu_result = np.empty_like(a)
        cuda.memcpy_dtoh(c_gpu_result, c_gpu)
        gpu_time = (time.time() - t0) * 1000

        # 验证结果
        max_error = np.max(np.abs(c_cpu - c_gpu_result))
        print(f"[GPU] 耗时: {gpu_time:.3f} ms（含内存传输）")
        print(f"[验证] 最大误差: {max_error:.2e}")
        print(f"[加速] GPU/CPU 比: {cpu_time/gpu_time:.2f}x")

        # 释放 GPU 内存
        a_gpu.free()
        b_gpu.free()
        c_gpu.free()

    except ImportError:
        print("[跳过] PyCUDA 未安装")
    except Exception as e:
        print(f"[错误] GPU 计算失败: {e}")


# ── 示例 2：矩阵乘法 ───────────────────────────────────────────────────────────
MATMUL_KERNEL = """
#define TILE_SIZE 16

__global__ void matmul(float *A, float *B, float *C, int N)
{
    // 使用共享内存优化的矩阵乘法（分块算法）
    __shared__ float tile_A[TILE_SIZE][TILE_SIZE];
    __shared__ float tile_B[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    float sum = 0.0f;

    for (int t = 0; t < (N + TILE_SIZE - 1) / TILE_SIZE; t++) {
        // 加载数据到共享内存
        if (row < N && t * TILE_SIZE + threadIdx.x < N)
            tile_A[threadIdx.y][threadIdx.x] = A[row * N + t * TILE_SIZE + threadIdx.x];
        else
            tile_A[threadIdx.y][threadIdx.x] = 0.0f;

        if (col < N && t * TILE_SIZE + threadIdx.y < N)
            tile_B[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * N + col];
        else
            tile_B[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        // 计算部分和
        for (int k = 0; k < TILE_SIZE; k++)
            sum += tile_A[threadIdx.y][k] * tile_B[k][threadIdx.x];

        __syncthreads();
    }

    if (row < N && col < N)
        C[row * N + col] = sum;
}
"""


def demo_matrix_multiply(n=512):
    """
    矩阵乘法：C = A × B（N×N 矩阵）
    使用共享内存分块优化
    """
    print(f"\n{'='*55}")
    print(f"  示例 2：矩阵乘法（{n}×{n}）")
    print(f"{'='*55}")

    A = np.random.randn(n, n).astype(np.float32)
    B = np.random.randn(n, n).astype(np.float32)

    # ── CPU 版本（NumPy）──
    t0 = time.time()
    C_cpu = np.dot(A, B)
    cpu_time = (time.time() - t0) * 1000
    print(f"\n[CPU/NumPy] 耗时: {cpu_time:.3f} ms")

    # ── GPU 版本 ──
    try:
        import pycuda.driver as cuda
        import pycuda.autoinit
        from pycuda.compiler import SourceModule

        mod = SourceModule(MATMUL_KERNEL)
        matmul = mod.get_function("matmul")

        t0 = time.time()
        A_gpu = cuda.mem_alloc(A.nbytes)
        B_gpu = cuda.mem_alloc(B.nbytes)
        C_gpu = cuda.mem_alloc(A.nbytes)

        cuda.memcpy_htod(A_gpu, A)
        cuda.memcpy_htod(B_gpu, B)

        tile_size = 16
        grid_dim = (n + tile_size - 1) // tile_size

        matmul(
            A_gpu, B_gpu, C_gpu, np.int32(n),
            block=(tile_size, tile_size, 1),
            grid=(grid_dim, grid_dim)
        )

        C_gpu_result = np.empty_like(A)
        cuda.memcpy_dtoh(C_gpu_result, C_gpu)
        gpu_time = (time.time() - t0) * 1000

        max_error = np.max(np.abs(C_cpu - C_gpu_result))
        print(f"[GPU] 耗时: {gpu_time:.3f} ms（含内存传输）")
        print(f"[验证] 最大误差: {max_error:.2e}")
        print(f"[加速] GPU/CPU 比: {cpu_time/gpu_time:.2f}x")

        A_gpu.free()
        B_gpu.free()
        C_gpu.free()

    except ImportError:
        print("[跳过] PyCUDA 未安装")
    except Exception as e:
        print(f"[错误] GPU 计算失败: {e}")


# ── 示例 3：图像处理（高斯模糊）──────────────────────────────────────────────
GAUSSIAN_BLUR_KERNEL = """
__global__ void gaussian_blur(
    unsigned char *input,
    unsigned char *output,
    int width, int height,
    float *kernel, int kernel_size)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int half = kernel_size / 2;
    float sum = 0.0f;
    float weight_sum = 0.0f;

    for (int ky = -half; ky <= half; ky++) {
        for (int kx = -half; kx <= half; kx++) {
            int nx = x + kx;
            int ny = y + ky;

            if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                float w = kernel[(ky + half) * kernel_size + (kx + half)];
                sum += input[ny * width + nx] * w;
                weight_sum += w;
            }
        }
    }

    output[y * width + x] = (unsigned char)(sum / weight_sum);
}
"""

SOBEL_KERNEL = """
__global__ void sobel_edge(
    unsigned char *input,
    unsigned char *output,
    int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < 1 || x >= width - 1 || y < 1 || y >= height - 1) {
        if (x < width && y < height) output[y * width + x] = 0;
        return;
    }

    // Sobel X 方向
    int gx = -input[(y-1)*width + (x-1)] + input[(y-1)*width + (x+1)]
             - 2*input[y*width + (x-1)] + 2*input[y*width + (x+1)]
             - input[(y+1)*width + (x-1)] + input[(y+1)*width + (x+1)];

    // Sobel Y 方向
    int gy = -input[(y-1)*width + (x-1)] - 2*input[(y-1)*width + x]
             - input[(y-1)*width + (x+1)] + input[(y+1)*width + (x-1)]
             + 2*input[(y+1)*width + x] + input[(y+1)*width + (x+1)];

    int magnitude = (int)sqrtf((float)(gx*gx + gy*gy));
    output[y * width + x] = (unsigned char)min(magnitude, 255);
}
"""


def create_gaussian_kernel(size=5, sigma=1.0):
    """创建高斯核"""
    import math
    kernel = np.zeros((size, size), dtype=np.float32)
    center = size // 2
    for i in range(size):
        for j in range(size):
            x = i - center
            y = j - center
            kernel[i, j] = math.exp(-(x**2 + y**2) / (2 * sigma**2))
    kernel /= kernel.sum()
    return kernel


def demo_image_processing(image_path=None):
    """
    图像处理示例：高斯模糊和 Sobel 边缘检测
    """
    print(f"\n{'='*55}")
    print(f"  示例 3：CUDA 图像处理")
    print(f"{'='*55}")

    try:
        import cv2
    except ImportError:
        print("[跳过] OpenCV 未安装")
        return

    # 加载或创建测试图像
    if image_path and os.path.exists(image_path):
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        print(f"[图像] 加载: {image_path}")
    else:
        # 创建测试图像（棋盘格）
        img = np.zeros((480, 640), dtype=np.uint8)
        for i in range(0, 480, 40):
            for j in range(0, 640, 40):
                if (i // 40 + j // 40) % 2 == 0:
                    img[i:i+40, j:j+40] = 200
        # 添加噪声
        noise = np.random.randint(0, 30, img.shape, dtype=np.uint8)
        img = cv2.add(img, noise)
        print("[图像] 使用生成的测试图像（棋盘格 + 噪声）")

    height, width = img.shape
    print(f"[图像] 尺寸: {width}x{height}")

    # ── CPU 高斯模糊 ──
    t0 = time.time()
    blurred_cpu = cv2.GaussianBlur(img, (5, 5), 1.0)
    cpu_blur_time = (time.time() - t0) * 1000
    print(f"\n[CPU] 高斯模糊耗时: {cpu_blur_time:.3f} ms")

    # ── CPU Sobel 边缘检测 ──
    t0 = time.time()
    sobel_cpu = cv2.Sobel(img, cv2.CV_8U, 1, 1, ksize=3)
    cpu_sobel_time = (time.time() - t0) * 1000
    print(f"[CPU] Sobel 边缘检测耗时: {cpu_sobel_time:.3f} ms")

    # ── GPU 版本 ──
    try:
        import pycuda.driver as cuda
        import pycuda.autoinit
        from pycuda.compiler import SourceModule

        mod = SourceModule(GAUSSIAN_BLUR_KERNEL + "\n" + SOBEL_KERNEL)
        gaussian_blur_gpu = mod.get_function("gaussian_blur")
        sobel_edge_gpu = mod.get_function("sobel_edge")

        # 高斯核
        gauss_kernel = create_gaussian_kernel(5, 1.0)

        # 分配 GPU 内存
        img_flat = img.flatten().astype(np.uint8)
        input_gpu = cuda.mem_alloc(img_flat.nbytes)
        output_gpu = cuda.mem_alloc(img_flat.nbytes)
        kernel_gpu = cuda.mem_alloc(gauss_kernel.nbytes)

        cuda.memcpy_htod(input_gpu, img_flat)
        cuda.memcpy_htod(kernel_gpu, gauss_kernel)

        block = (16, 16, 1)
        grid = ((width + 15) // 16, (height + 15) // 16)

        # GPU 高斯模糊
        t0 = time.time()
        gaussian_blur_gpu(
            input_gpu, output_gpu,
            np.int32(width), np.int32(height),
            kernel_gpu, np.int32(5),
            block=block, grid=grid
        )
        cuda.Context.synchronize()
        gpu_blur_time = (time.time() - t0) * 1000

        blurred_gpu = np.empty_like(img_flat)
        cuda.memcpy_dtoh(blurred_gpu, output_gpu)
        blurred_gpu = blurred_gpu.reshape(height, width)

        print(f"\n[GPU] 高斯模糊耗时: {gpu_blur_time:.3f} ms")
        print(f"[加速] 高斯模糊 GPU/CPU: {cpu_blur_time/gpu_blur_time:.2f}x")

        # GPU Sobel 边缘检测
        t0 = time.time()
        sobel_edge_gpu(
            input_gpu, output_gpu,
            np.int32(width), np.int32(height),
            block=block, grid=grid
        )
        cuda.Context.synchronize()
        gpu_sobel_time = (time.time() - t0) * 1000

        sobel_gpu = np.empty_like(img_flat)
        cuda.memcpy_dtoh(sobel_gpu, output_gpu)
        sobel_gpu = sobel_gpu.reshape(height, width)

        print(f"[GPU] Sobel 边缘检测耗时: {gpu_sobel_time:.3f} ms")
        print(f"[加速] Sobel 边缘检测 GPU/CPU: {cpu_sobel_time/gpu_sobel_time:.2f}x")

        # 保存结果
        os.makedirs("output", exist_ok=True)
        cv2.imwrite("output/original.jpg", img)
        cv2.imwrite("output/blurred_cpu.jpg", blurred_cpu)
        cv2.imwrite("output/blurred_gpu.jpg", blurred_gpu)
        cv2.imwrite("output/sobel_cpu.jpg", sobel_cpu)
        cv2.imwrite("output/sobel_gpu.jpg", sobel_gpu)
        print(f"\n[保存] 结果已保存到 output/ 目录")

        # 显示结果
        result = np.hstack([img, blurred_gpu, sobel_gpu])
        cv2.imshow("Original | GPU Blur | GPU Sobel", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        input_gpu.free()
        output_gpu.free()
        kernel_gpu.free()

    except ImportError:
        print("[跳过] PyCUDA 未安装")
    except Exception as e:
        print(f"[错误] GPU 图像处理失败: {e}")


# ── 示例 4：性能基准测试 ───────────────────────────────────────────────────────
def benchmark():
    """综合性能基准测试"""
    print(f"\n{'='*55}")
    print(f"  CUDA 性能基准测试")
    print(f"{'='*55}")

    sizes = [10_000, 100_000, 1_000_000, 10_000_000]

    print(f"\n{'向量加法性能对比':^55}")
    print(f"{'大小':>15} {'CPU(ms)':>12} {'GPU(ms)':>12} {'加速比':>10}")
    print("-" * 55)

    try:
        import pycuda.driver as cuda
        import pycuda.autoinit
        from pycuda.compiler import SourceModule

        mod = SourceModule(VECTOR_ADD_KERNEL)
        vector_add = mod.get_function("vector_add")

        for n in sizes:
            a = np.random.randn(n).astype(np.float32)
            b = np.random.randn(n).astype(np.float32)

            # CPU
            t0 = time.time()
            for _ in range(5):
                c = a + b
            cpu_time = (time.time() - t0) / 5 * 1000

            # GPU
            a_gpu = cuda.mem_alloc(a.nbytes)
            b_gpu = cuda.mem_alloc(b.nbytes)
            c_gpu = cuda.mem_alloc(a.nbytes)
            cuda.memcpy_htod(a_gpu, a)
            cuda.memcpy_htod(b_gpu, b)

            block_size = 256
            grid_size = (n + block_size - 1) // block_size

            # 预热
            vector_add(a_gpu, b_gpu, c_gpu, np.int32(n),
                       block=(block_size, 1, 1), grid=(grid_size, 1))

            t0 = time.time()
            for _ in range(5):
                vector_add(a_gpu, b_gpu, c_gpu, np.int32(n),
                           block=(block_size, 1, 1), grid=(grid_size, 1))
            cuda.Context.synchronize()
            gpu_time = (time.time() - t0) / 5 * 1000

            speedup = cpu_time / gpu_time
            print(f"{n:>15,} {cpu_time:>12.3f} {gpu_time:>12.3f} {speedup:>9.2f}x")

            a_gpu.free()
            b_gpu.free()
            c_gpu.free()

    except ImportError:
        print("[跳过] PyCUDA 未安装")
    except Exception as e:
        print(f"[错误] 基准测试失败: {e}")


def parse_args():
    parser = argparse.ArgumentParser(description="Jetson Nano CUDA 加速示例")
    parser.add_argument("--demo", choices=["all", "vector", "matrix", "image"],
                        default="all", help="运行的示例（默认: all）")
    parser.add_argument("--input", type=str, default=None,
                        help="图像处理示例的输入图片")
    parser.add_argument("--benchmark", action="store_true",
                        help="运行性能基准测试")
    parser.add_argument("--size", type=int, default=None,
                        help="向量/矩阵大小")
    return parser.parse_args()


def main():
    print("=" * 55)
    print("  Jetson Nano CUDA 并行计算示例")
    print("=" * 55)

    args = parse_args()

    # 显示 GPU 信息
    if not print_gpu_info():
        print("\n[提示] 无法访问 GPU，请确认:")
        print("  1. PyCUDA 已安装: pip3 install pycuda")
        print("  2. CUDA 已安装（JetPack 包含）")
        print("  3. 在 Jetson Nano 上运行")
        return

    if args.benchmark:
        benchmark()
        return

    n = args.size

    if args.demo in ("all", "vector"):
        demo_vector_add(n or 1_000_000)

    if args.demo in ("all", "matrix"):
        demo_matrix_multiply(n or 512)

    if args.demo in ("all", "image"):
        demo_image_processing(args.input)

    print(f"\n{'='*55}")
    print("  所有示例运行完成")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()