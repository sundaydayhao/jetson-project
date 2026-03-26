# 示例 05：SSD 目标检测（TensorRT 加速）

## 功能说明

使用预训练的 SSD MobileNet V2 模型进行实时目标检测，支持 80 类 COCO 数据集目标。通过 TensorRT 优化，在 Jetson Nano 上实现高效推理。

## 检测类别（COCO 80类）

人、自行车、汽车、摩托车、飞机、公共汽车、火车、卡车、船、交通灯、消防栓、停车标志、长椅、鸟、猫、狗、马、羊、牛、大象、熊、斑马、长颈鹿、背包、雨伞、手提包、领带、手提箱、飞盘、滑雪板、运动球、风筝、棒球棒、棒球手套、滑板、冲浪板、网球拍、瓶子、酒杯、杯子、叉子、刀、勺子、碗、香蕉、苹果、三明治、橙子、西兰花、胡萝卜、热狗、披萨、甜甜圈、蛋糕、椅子、沙发、盆栽植物、床、餐桌、马桶、电视、笔记本电脑、鼠标、遥控器、键盘、手机、微波炉、烤箱、烤面包机、水槽、冰箱、书、时钟、花瓶、剪刀、泰迪熊、吹风机、牙刷

## 依赖

```bash
# 安装 TensorRT Python 绑定（JetPack 已包含）
# 安装 PyCUDA
pip3 install pycuda

# 下载预训练模型
bash download_models.sh
```

## 运行方式

```bash
# 摄像头实时检测（使用 TensorRT 引擎）
python3 main.py

# 使用 OpenCV DNN 后端（无需 TensorRT）
python3 main.py --backend opencv

# 检测图片
python3 main.py --image test.jpg

# 指定置信度阈值
python3 main.py --threshold 0.4

# CSI 摄像头
python3 main.py --csi
```

## 键盘控制

| 按键 | 功能 |
|------|------|
| `q` | 退出 |
| `s` | 截图 |
| `+/-` | 调整置信度阈值 |

## 预期效果

- 实时检测画面中的物体
- 显示类别名称、置信度和边界框
- 不同类别使用不同颜色标注
- 显示实时 FPS

## 知识点

- SSD（Single Shot MultiBox Detector）原理
- TensorRT 引擎构建与推理
- OpenCV DNN 模块目标检测
- COCO 数据集类别
- 非极大值抑制（NMS）