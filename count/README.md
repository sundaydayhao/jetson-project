# 实时黄豆 / 药片 / 花生豆计数

这个目录提供一个基于 OpenCV 的小目标实时计数程序，支持 USB 摄像头和 Jetson CSI 摄像头。它不依赖训练好的模型，适合在固定相机、固定背景、托盘或传送带场景下统计黄豆、花生豆、药片等颗粒状物体。

## 运行

```bash
cd /home/hao/projects/jetson-project/count

# USB 摄像头，默认黄豆颜色分割
python3 main.py --preset soybean

# CSI 摄像头
python3 main.py --csi --preset soybean

# 花生豆
python3 main.py --preset peanut

# 药片或颜色不稳定的混合物，推荐背景差分模式
python3 main.py --preset pill
```

## 推荐现场流程

1. 固定摄像头和光源，让背景尽量干净、颜色单一。
2. 启动程序后按 `r` 框选计数区域，只保留托盘或传送带区域。
3. 如果使用 `pill` / `mixed` / `--mode background`，先清空托盘，再按 `b` 采集空背景。
4. 按 `m` 查看二值掩膜，按 `t` 打开滑条调 HSV、面积阈值和拆分阈值。
5. 调到每个物体只出现一个绿色框后读取左上角 `Count`。

## 快捷键

| 按键 | 功能 |
| --- | --- |
| `q` / `ESC` | 退出 |
| `s` / 空格 | 保存截图到 `captures/` |
| `b` | 采集空背景，用于背景差分 |
| `r` | 鼠标框选 ROI |
| `0` | 清除 ROI |
| `m` | 显示 / 隐藏掩膜窗口 |
| `t` | 打开 / 关闭调参滑条 |
| `p` | 保存当前参数到 JSON 配置 |
| `c` | 清零累计过线计数 |
| `f` | 全屏切换 |

## 当前画面计数

默认模式统计当前画面中可见物体数量：

```bash
python3 main.py --preset soybean --count-mode present
```

常用参数：

```bash
# 自定义 HSV 范围：h_min,h_max,s_min,s_max,v_min,v_max
python3 main.py --mode color --hsv 10,45,35,255,40,255

# 调整目标面积范围，过滤灰尘、小反光或过大粘连区域
python3 main.py --preset soybean --min-area 80 --max-area 6000

# 不拆分粘连物体
python3 main.py --preset soybean --no-split

# 加载保存过的参数
python3 main.py --config count_config.json
```

## 传送带累计计数

如果物体从画面中经过，需要累计过线数量：

```bash
# 横线位于画面高度 50%，上下方向穿过都会计数
python3 main.py --preset soybean --count-mode line

# 只统计向下经过横线的物体
python3 main.py --preset soybean --count-mode line --line-direction down

# 竖线位于画面宽度 60%，只统计向右经过的物体
python3 main.py --preset soybean --count-mode line --line-axis x --line-pos 0.6 --line-direction right
```

## 适配建议

- 黄豆/花生：优先用 `--preset soybean` 或 `--preset peanut`，打开 `m` 看掩膜是否只保留豆子。
- 白色或彩色药片：优先用 `--preset pill`，清空区域后按 `b` 采背景。
- 如果计数偏少：降低 `min-area`，或调低 `Split x100` 让粘连物体更容易被分开。
- 如果计数偏多：提高 `min-area`、提高 `Circular x100`，或减少反光和阴影。
- 物体重叠严重时，传统视觉会有上限；需要稳定处理堆叠/遮挡时建议采集样本训练 YOLO 分割模型。
