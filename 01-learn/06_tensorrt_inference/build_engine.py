#!/usr/bin/env python3
"""
步骤 2：将 ONNX 模型转换为 TensorRT 引擎
支持 FP32、FP16 精度模式
注意：首次构建引擎较慢（5-15 分钟），构建后可重复使用
"""

import os
import argparse
import time


def build_engine(
    onnx_path,
    engine_path,
    precision="fp16",
    max_batch_size=1,
    workspace_size=1 << 28  # 256 MB
):
    """
    构建 TensorRT 引擎

    :param onnx_path: ONNX 模型路径
    :param engine_path: 输出引擎路径
    :param precision: 精度模式 fp32/fp16/int8
    :param max_batch_size: 最大批量大小
    :param workspace_size: 工作空间大小（字节）
    """
    try:
        import tensorrt as trt
    except ImportError:
        print("[错误] TensorRT 未安装")
        print("  JetPack 已包含 TensorRT，请确认安装正确")
        print("  或手动安装: sudo apt-get install python3-libnvinfer-dev")
        return False

    if not os.path.exists(onnx_path):
        print(f"[错误] ONNX 文件不存在: {onnx_path}")
        print(f"  请先运行: python3 export_onnx.py")
        return False

    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

    print(f"[构建] ONNX 路径: {onnx_path}")
    print(f"[构建] 引擎路径: {engine_path}")
    print(f"[构建] 精度模式: {precision.upper()}")
    print(f"[构建] 最大批量: {max_batch_size}")
    print(f"[构建] 工作空间: {workspace_size // 1024 // 1024} MB")
    print(f"\n[提示] 首次构建引擎需要 5-15 分钟，请耐心等待...\n")

    start_time = time.time()

    # 创建 Builder
    with trt.Builder(TRT_LOGGER) as builder, \
         builder.create_network(
             1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
         ) as network, \
         trt.OnnxParser(network, TRT_LOGGER) as parser:

        # 构建配置
        config = builder.create_builder_config()
        config.max_workspace_size = workspace_size

        # 设置精度
        if precision == "fp16":
            if builder.platform_has_fast_fp16:
                config.set_flag(trt.BuilderFlag.FP16)
                print("[精度] 启用 FP16 加速")
            else:
                print("[警告] 当前平台不支持 FP16，使用 FP32")
        elif precision == "int8":
            if builder.platform_has_fast_int8:
                config.set_flag(trt.BuilderFlag.INT8)
                print("[精度] 启用 INT8 量化（需要校准数据）")
            else:
                print("[警告] 当前平台不支持 INT8，使用 FP32")

        # 解析 ONNX
        print("[解析] 解析 ONNX 模型...")
        with open(onnx_path, "rb") as f:
            if not parser.parse(f.read()):
                print("[错误] ONNX 解析失败:")
                for error in range(parser.num_errors):
                    print(f"  {parser.get_error(error)}")
                return False
        print("[解析] ONNX 解析成功")

        # 显示网络信息
        print(f"\n[网络] 层数: {network.num_layers}")
        print(f"[网络] 输入: {network.get_input(0).name} "
              f"{network.get_input(0).shape}")
        print(f"[网络] 输出: {network.get_output(0).name} "
              f"{network.get_output(0).shape}")

        # 构建引擎
        print(f"\n[构建] 开始优化引擎（这需要几分钟）...")
        engine = builder.build_engine(network, config)

        if engine is None:
            print("[错误] 引擎构建失败！")
            return False

        # 保存引擎
        os.makedirs(os.path.dirname(engine_path) or ".", exist_ok=True)
        with open(engine_path, "wb") as f:
            f.write(engine.serialize())

        elapsed = time.time() - start_time
        file_size = os.path.getsize(engine_path) / 1024 / 1024
        print(f"\n[完成] 引擎构建成功！")
        print(f"  耗时: {elapsed:.1f} 秒")
        print(f"  文件: {engine_path} ({file_size:.1f} MB)")
        print(f"\n[下一步] 运行推理:")
        print(f"  python3 main.py --engine {engine_path} --image test.jpg")
        return True


def main():
    parser = argparse.ArgumentParser(description="ONNX → TensorRT 引擎构建工具")
    parser.add_argument("--onnx", type=str, default="models/resnet50.onnx",
                        help="ONNX 模型路径")
    parser.add_argument("--engine", type=str, default="models/resnet50_fp16.trt",
                        help="输出引擎路径")
    parser.add_argument("--precision", choices=["fp32", "fp16", "int8"],
                        default="fp16", help="精度模式（默认: fp16）")
    parser.add_argument("--batch", type=int, default=1,
                        help="最大批量大小（默认: 1）")
    parser.add_argument("--workspace", type=int, default=256,
                        help="工作空间大小 MB（默认: 256）")
    args = parser.parse_args()

    print("=" * 55)
    print("  ONNX → TensorRT 引擎构建工具")
    print("=" * 55)

    success = build_engine(
        onnx_path=args.onnx,
        engine_path=args.engine,
        precision=args.precision,
        max_batch_size=args.batch,
        workspace_size=args.workspace * 1024 * 1024
    )

    if not success:
        print("\n[失败] 引擎构建失败，请检查错误信息")
        exit(1)


if __name__ == "__main__":
    main()