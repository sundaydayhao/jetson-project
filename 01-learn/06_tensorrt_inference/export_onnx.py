#!/usr/bin/env python3
"""
步骤 1：将 PyTorch ResNet-50 模型导出为 ONNX 格式
需要安装 PyTorch 和 torchvision
"""

import os
import argparse

os.makedirs("models", exist_ok=True)


def export_resnet50(output_path="models/resnet50.onnx", opset=11):
    """导出 ResNet-50 为 ONNX"""
    try:
        import torch
        import torchvision.models as models
    except ImportError:
        print("[错误] 请先安装 PyTorch: pip3 install torch torchvision")
        return False

    print("[导出] 加载 ResNet-50 预训练模型...")
    model = models.resnet50(pretrained=True)
    model.eval()

    # 创建示例输入（batch=1, channels=3, height=224, width=224）
    dummy_input = torch.randn(1, 3, 224, 224)

    print(f"[导出] 导出 ONNX 模型到: {output_path}")
    print(f"[导出] ONNX opset 版本: {opset}")

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"}
        }
    )

    # 验证 ONNX 模型
    try:
        import onnx
        model_onnx = onnx.load(output_path)
        onnx.checker.check_model(model_onnx)
        print(f"[验证] ONNX 模型验证通过")

        # 显示模型信息
        print(f"\n[模型信息]")
        print(f"  输入: {[inp.name for inp in model_onnx.graph.input]}")
        print(f"  输出: {[out.name for out in model_onnx.graph.output]}")
        file_size = os.path.getsize(output_path) / 1024 / 1024
        print(f"  文件大小: {file_size:.1f} MB")
    except ImportError:
        print("[提示] 安装 onnx 可验证模型: pip3 install onnx")

    print(f"\n[完成] ONNX 模型已保存: {output_path}")
    print(f"[下一步] 运行: python3 build_engine.py --onnx {output_path}")
    return True


def export_mobilenet_v2(output_path="models/mobilenet_v2.onnx", opset=11):
    """导出 MobileNet V2 为 ONNX（更轻量，适合 Jetson Nano）"""
    try:
        import torch
        import torchvision.models as models
    except ImportError:
        print("[错误] 请先安装 PyTorch")
        return False

    print("[导出] 加载 MobileNet V2 预训练模型...")
    model = models.mobilenet_v2(pretrained=True)
    model.eval()

    dummy_input = torch.randn(1, 3, 224, 224)

    print(f"[导出] 导出 ONNX 模型到: {output_path}")
    torch.onnx.export(
        model, dummy_input, output_path,
        export_params=True, opset_version=opset,
        do_constant_folding=True,
        input_names=["input"], output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}
    )

    file_size = os.path.getsize(output_path) / 1024 / 1024
    print(f"[完成] MobileNet V2 ONNX 已保存: {output_path} ({file_size:.1f} MB)")
    return True


def main():
    parser = argparse.ArgumentParser(description="导出 PyTorch 模型为 ONNX")
    parser.add_argument("--model", choices=["resnet50", "mobilenet_v2", "all"],
                        default="resnet50", help="要导出的模型")
    parser.add_argument("--opset", type=int, default=11, help="ONNX opset 版本")
    args = parser.parse_args()

    print("=" * 55)
    print("  PyTorch → ONNX 模型导出工具")
    print("=" * 55)

    if args.model in ("resnet50", "all"):
        export_resnet50(opset=args.opset)

    if args.model in ("mobilenet_v2", "all"):
        export_mobilenet_v2(opset=args.opset)


if __name__ == "__main__":
    main()