# 示例 04：OpenCV 人脸检测

## 功能说明

使用 OpenCV 的 Haar 级联分类器和 DNN 模块进行实时人脸检测，支持摄像头实时检测和图片检测两种模式。

## 检测方法

### 方法 1：Haar 级联分类器（传统方法）
- 速度快，CPU 占用低
- 对正面人脸效果好
- 对侧脸、遮挡效果较差

### 方法 2：DNN 人脸检测（深度学习）
- 精度更高，对各角度人脸效果好
- 使用 OpenCV 内置的 Caffe 模型
- 在 Jetson Nano 上可利用 CUDA 加速

## 依赖

```bash
# OpenCV 已随 JetPack 预装
# 下载 DNN 模型文件
bash download_models.sh
```

## 运行方式

```bash
# 摄像头实时检测（Haar 方法）
python3 main.py

# 摄像头实时检测（DNN 方法）
python3 main.py --method dnn

# 检测图片文件
python3 main.py --image test.jpg

# CSI 摄像头
python3 main.py --csi

# 同时检测人脸和眼睛
python3 main.py --eyes
```

## 键盘控制

| 按键 | 功能 |
|------|------|
| `q` | 退出 |
| `s` | 截图 |
| `m` | 切换检测方法（Haar/DNN） |
| `e` | 切换眼睛检测 |

## 预期效果

- 实时在人脸周围绘制矩形框
- 显示检测到的人脸数量和置信度
- 显示实时 FPS

## 知识点

- Haar 级联分类器原理与使用
- OpenCV DNN 模块使用
- 人脸检测参数调优（scaleFactor、minNeighbors）
- 多尺度检测
- 非极大值抑制（NMS）