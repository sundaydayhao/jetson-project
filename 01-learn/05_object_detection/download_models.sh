#!/bin/bash
# 下载 SSD MobileNet V2 COCO 预训练模型

mkdir -p models
cd models

echo "[下载] SSD MobileNet V2 COCO 模型..."
wget -q --show-progress \
  "http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz" \
  -O ssd_mobilenet_v2_coco.tar.gz

echo "[解压] 解压模型文件..."
tar -xzf ssd_mobilenet_v2_coco.tar.gz
cp ssd_mobilenet_v2_coco_2018_03_29/frozen_inference_graph.pb .

echo "[下载] pbtxt 配置文件..."
wget -q --show-progress \
  "https://raw.githubusercontent.com/opencv/opencv_extra/master/testdata/dnn/ssd_mobilenet_v2_coco_2018_03_29.pbtxt" \
  -O ssd_mobilenet_v2_coco_2018_03_29.pbtxt

echo "[完成] 模型文件已准备好"
ls -lh *.pb *.pbtxt 2>/dev/null