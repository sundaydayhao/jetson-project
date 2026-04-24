#!/usr/bin/env python3
"""
将 ONNX 目标检测模型构建为 TensorRT 引擎。

默认使用 ONNX Model Zoo 的 SSD MobileNet V1 模型：
https://huggingface.co/onnxmodelzoo/ssd_mobilenet_v1_12
"""

import argparse
import os
import time


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
DEFAULT_ONNX = os.path.join(MODEL_DIR, "ssd_mobilenet_v1_12.onnx")
DEFAULT_ENGINE = os.path.join(MODEL_DIR, "ssd_mobilenet_v1_12_fp16.trt")


def parse_shape(text):
    """解析形如 1x480x640x3 的形状字符串"""
    normalized = text.lower().replace(",", "x").replace(" ", "")
    parts = [part for part in normalized.split("x") if part]
    if len(parts) != 4:
        raise ValueError(f"无效的 shape: {text}")
    return tuple(int(part) for part in parts)


def shape_to_str(shape):
    return "x".join(str(int(dim)) for dim in shape)


def get_tensor_shape(tensor):
    return tuple(int(dim) for dim in tensor.shape)


def build_engine(
    onnx_path=DEFAULT_ONNX,
    engine_path=DEFAULT_ENGINE,
    precision="fp16",
    min_shape=(1, 300, 300, 3),
    opt_shape=(1, 480, 640, 3),
    max_shape=(1, 1080, 1920, 3),
    workspace_size=512 * 1024 * 1024,
):
    """构建 TensorRT 引擎"""
    try:
        import tensorrt as trt
    except ImportError:
        print("[错误] TensorRT 未安装")
        print("  请先安装 NVIDIA TensorRT Python 绑定")
        return False

    if not os.path.exists(onnx_path):
        print(f"[错误] ONNX 文件不存在: {onnx_path}")
        print("  请先运行: bash download_models.sh")
        return False

    logger = trt.Logger(trt.Logger.WARNING)
    start_time = time.time()

    print(f"[构建] ONNX: {onnx_path}")
    print(f"[构建] 引擎: {engine_path}")
    print(f"[构建] 精度: {precision.upper()}")
    print(f"[构建] 工作空间: {workspace_size // 1024 // 1024} MB")

    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)

    with trt.Builder(logger) as builder, \
         builder.create_network(network_flags) as network, \
         trt.OnnxParser(network, logger) as parser:

        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_size)

        if precision == "fp16":
            if builder.platform_has_fast_fp16:
                config.set_flag(trt.BuilderFlag.FP16)
                print("[精度] 启用 FP16")
            else:
                print("[警告] 当前平台不支持 FP16，回退到 FP32")
        elif precision == "int8":
            if builder.platform_has_fast_int8:
                print("[警告] INT8 需要额外校准数据，本示例未实现校准，回退到 FP32")
            else:
                print("[警告] 当前平台不支持 INT8，回退到 FP32")

        print("[解析] 解析 ONNX 模型...")
        with open(onnx_path, "rb") as f:
            if not parser.parse(f.read()):
                print("[错误] ONNX 解析失败:")
                for i in range(parser.num_errors):
                    print(f"  {parser.get_error(i)}")
                return False

        print("[解析] ONNX 解析成功")

        for i in range(network.num_inputs):
            tensor = network.get_input(i)
            print(f"[网络] 输入: {tensor.name} {get_tensor_shape(tensor)} {tensor.dtype}")

        for i in range(network.num_outputs):
            tensor = network.get_output(i)
            print(f"[网络] 输出: {tensor.name} {get_tensor_shape(tensor)} {tensor.dtype}")

        if network.num_inputs != 1:
            print("[错误] 当前示例仅支持单输入检测模型")
            return False

        input_tensor = network.get_input(0)
        input_shape = get_tensor_shape(input_tensor)

        if any(dim < 0 for dim in input_shape):
            profile = builder.create_optimization_profile()
            if len(min_shape) != len(input_shape):
                print("[错误] Optimization profile 维度与模型输入不一致")
                return False

            profile.set_shape(input_tensor.name, min_shape, opt_shape, max_shape)
            config.add_optimization_profile(profile)
            print(
                "[Profile] "
                f"min={shape_to_str(min_shape)} "
                f"opt={shape_to_str(opt_shape)} "
                f"max={shape_to_str(max_shape)}"
            )

        print("[构建] 开始构建 TensorRT 引擎...")
        serialized_engine = builder.build_serialized_network(network, config)
        if serialized_engine is None:
            print("[错误] TensorRT 引擎构建失败")
            return False

        os.makedirs(os.path.dirname(engine_path) or ".", exist_ok=True)
        with open(engine_path, "wb") as f:
            f.write(bytes(serialized_engine))

    elapsed = time.time() - start_time
    file_size = os.path.getsize(engine_path) / 1024 / 1024
    print(f"[完成] 引擎已保存: {engine_path}")
    print(f"  耗时: {elapsed:.1f} 秒")
    print(f"  大小: {file_size:.1f} MB")
    return True


def main():
    parser = argparse.ArgumentParser(description="ONNX -> TensorRT 引擎构建工具")
    parser.add_argument("--onnx", type=str, default=DEFAULT_ONNX,
                        help="ONNX 模型路径")
    parser.add_argument("--engine", type=str, default=DEFAULT_ENGINE,
                        help="输出引擎路径")
    parser.add_argument("--precision", choices=["fp32", "fp16", "int8"],
                        default="fp16", help="精度模式（默认: fp16）")
    parser.add_argument("--min-shape", type=str, default="1x300x300x3",
                        help="动态输入下的最小 shape（默认: 1x300x300x3）")
    parser.add_argument("--opt-shape", type=str, default="1x480x640x3",
                        help="动态输入下的优化 shape（默认: 1x480x640x3）")
    parser.add_argument("--max-shape", type=str, default="1x1080x1920x3",
                        help="动态输入下的最大 shape（默认: 1x1080x1920x3）")
    parser.add_argument("--workspace", type=int, default=512,
                        help="工作空间大小 MB（默认: 512）")
    args = parser.parse_args()

    print("=" * 60)
    print("  目标检测 TensorRT 引擎构建工具")
    print("=" * 60)

    success = build_engine(
        onnx_path=args.onnx,
        engine_path=args.engine,
        precision=args.precision,
        min_shape=parse_shape(args.min_shape),
        opt_shape=parse_shape(args.opt_shape),
        max_shape=parse_shape(args.max_shape),
        workspace_size=args.workspace * 1024 * 1024,
    )

    if not success:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
