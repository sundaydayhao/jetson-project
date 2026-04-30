#!/bin/bash
# 下载 05_object_detection 所需模型
#
# - OpenCV 后端：TensorFlow SSD MobileNet V2
# - TensorRT 后端：ONNX SSD MobileNet V1

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MODEL_DIR="$SCRIPT_DIR/models"
mkdir -p "$MODEL_DIR"

download() {
  local url="$1"
  local output="$2"

  if command -v curl >/dev/null 2>&1; then
    curl -L --fail "$url" -o "$output"
  else
    wget -O "$output" "$url"
  fi
}

echo "[目录] 模型目录: $MODEL_DIR"
cd "$MODEL_DIR"

echo "[下载] TensorFlow SSD MobileNet V2（OpenCV 后端）..."
download \
  "http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz" \
  "ssd_mobilenet_v2_coco.tar.gz"

echo "[解压] 解压 OpenCV 模型..."
tar -xzf ssd_mobilenet_v2_coco.tar.gz
cp ssd_mobilenet_v2_coco_2018_03_29/frozen_inference_graph.pb .

echo "[下载] OpenCV pbtxt 配置文件..."
download \
  "https://raw.githubusercontent.com/opencv/opencv_extra/master/testdata/dnn/ssd_mobilenet_v2_coco_2018_03_29.pbtxt" \
  "ssd_mobilenet_v2_coco_2018_03_29.pbtxt"

echo "[下载] ONNX SSD MobileNet V1（TensorRT 后端）..."
download \
  "https://huggingface.co/onnxmodelzoo/ssd_mobilenet_v1_12/resolve/main/model/ssd_mobilenet_v1_12.onnx" \
  "ssd_mobilenet_v1_12.onnx"

echo "[完成] 已准备的模型文件："
ls -lh \
  frozen_inference_graph.pb \
  ssd_mobilenet_v2_coco_2018_03_29.pbtxt \
  ssd_mobilenet_v1_12.onnx
