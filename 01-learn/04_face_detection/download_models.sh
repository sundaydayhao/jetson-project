#!/bin/bash
# 下载 OpenCV DNN 人脸检测模型（Caffe 格式）

mkdir -p models
cd models

echo "[下载] deploy.prototxt ..."
wget -q --show-progress \
  "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt" \
  -O deploy.prototxt

echo "[下载] res10_300x300_ssd_iter_140000.caffemodel ..."
wget -q --show-progress \
  "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel" \
  -O res10_300x300_ssd_iter_140000.caffemodel

echo "[完成] 模型文件已下载到 models/ 目录"
ls -lh