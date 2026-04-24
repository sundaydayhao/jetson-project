# Jetson 上移植 YOLO 与摄像头排障实战笔记

这份文档整理了前面在 Jetson 上部署 YOLO、调用摄像头和导出 TensorRT 时遇到的典型问题，以及对应的排查思路。

适用场景：

- 设备：Jetson Orin / Xavier / Nano 一类 NVIDIA Jetson 开发板
- 系统：JetPack 5.x 或 6.x
- 目标：在 Jetson 上用摄像头做 YOLO 实时识别，并进一步导出 TensorRT

## 1. 先说结论

Jetson 上跑 YOLO 的推荐顺序是：

1. 先确认摄像头本身能正常工作
2. 再用 `Ultralytics YOLO` 跑通 `source=0`
3. 确认 `torch.cuda.is_available()` 为 `True`
4. 最后再导出 `TensorRT engine`

如果一上来就直接折腾 TensorRT，通常很容易把“摄像头问题”“显示问题”“PyTorch/CUDA 兼容问题”“TensorRT 版本冲突”混在一起。

## 2. 摄像头实时识别的推荐路线

### 2.1 USB 摄像头

最省事的方式：

```bash
yolo predict model=yolo26n.pt source=0 save=True imgsz=640 device=0
```

说明：

- `source=0` 一般表示第一路 USB 摄像头
- 如果是在 SSH 里跑，建议先用 `save=True`
- `show=True` 需要桌面显示环境

先单独确认摄像头能否工作：

```bash
v4l2-ctl --list-devices
gst-launch-1.0 v4l2src device=/dev/video0 ! videoconvert ! autovideosink
```

### 2.2 CSI 摄像头

CSI 摄像头在 Jetson 上通常更适合用 GStreamer 管道读取。

先测相机：

```bash
gst-launch-1.0 nvarguscamerasrc ! \
  'video/x-raw(memory:NVMM), width=(int)1280, height=(int)720, format=(string)NV12, framerate=(fraction)30/1' ! \
  nv3dsink -e
```

如果后续要结合 OpenCV + YOLO，推荐用 `nvarguscamerasrc` 管道喂给 `cv2.VideoCapture(...)`。

## 3. `cheese` 报 `JPEG parameter struct mismatch` 怎么办

如果在 Jetson 上运行 `cheese` 报错：

```text
JPEG parameter struct mismatch: library thinks size is 584, caller expects 728
```

这通常不是摄像头坏了，而是 JPEG 相关库或 NVIDIA GStreamer 插件路径发生了 ABI 冲突。

前面排查中，一个常见修复路径是：

1. 先备份以下两个 GStreamer 插件
2. 暂时移走它们
3. 清理 GStreamer 缓存
4. 重启后再试 `cheese`

示例：

```bash
sudo mkdir -p /opt/nvidia-plugin-backup/gstreamer-1.0

sudo mv /usr/lib/aarch64-linux-gnu/gstreamer-1.0/libgstnvvideo4linux2.so \
  /opt/nvidia-plugin-backup/gstreamer-1.0/

sudo mv /usr/lib/aarch64-linux-gnu/gstreamer-1.0/libgstnvjpeg.so \
  /opt/nvidia-plugin-backup/gstreamer-1.0/

rm -f ~/.cache/gstreamer-1.0/registry.aarch64.bin
reboot
```

注意：

- 这类修复只适合在 `cheese` 有问题、但你又急着验证摄像头时使用
- 可能影响依赖 NVIDIA GStreamer 硬件插件的其他程序
- 如果只是想看摄像头画面，不一定非要用 `cheese`，`gst-launch-1.0` 或 `guvcview` 往往更稳

## 4. `yolo26n.pt` 无法加载：模型文件损坏

如果你看到这类报错：

```text
PytorchStreamReader failed reading zip archive: failed finding central directory
```

这通常说明：

- 模型没有下完整
- 传输时损坏
- 本地同名文件其实不是模型，而是一个错误页或残缺文件

建议先检查：

```bash
find ~ -name 'yolo26n.pt' -ls
ls -lh ~/yolo26n.pt
file ~/yolo26n.pt
head -c 128 ~/yolo26n.pt
```

常见修复方法：

```bash
rm -f ~/yolo26n.pt
wget -O ~/yolo26n.pt https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt
```

先用图片验证模型能否正常加载：

```bash
yolo predict model=~/yolo26n.pt source='https://ultralytics.com/images/bus.jpg' device=0
```

图片能跑通，再回到摄像头。

## 5. `show=True` 失败：没有图形显示环境

如果你看到：

```text
WARNING ⚠️ Environment does not support cv2.imshow() or PIL Image.show()
The DISPLAY environment variable isn't set.
```

说明你当前是在：

- SSH 终端
- 没有登录本地图形桌面
- 或者没有正确设置 `DISPLAY`

这种情况下不要先纠结窗口，先改成保存结果：

```bash
yolo predict model=yolo26n.pt source=0 save=True imgsz=640 device=0
```

如果你就在 Jetson 本机桌面上操作，可以试：

```bash
export DISPLAY=:0
yolo predict model=yolo26n.pt source=0 show=True imgsz=640 device=0
```

## 6. `device=0` 无法使用：PyTorch 和 Jetson CUDA 环境不匹配

前面实际遇到过这一类情况：

```text
The NVIDIA driver on your system is too old
torch.cuda.is_available(): False
torch.cuda.device_count(): 1
ValueError: Invalid CUDA 'device=0' requested
```

这通常不是 Jetson 真的“没 GPU”，而是：

- 虚拟环境里装了不适合 Jetson 的通用 `torch`
- `pip` 拉到了 PC 版/通用 CUDA 版 wheel
- PyTorch 编译时依赖的 CUDA 版本，高于当前 JetPack 自带驱动支持的版本

例如前面碰到的错误组合就是：

- 环境中装了 `torch-2.11.0+cu130`
- 但设备实际跑的是 JetPack 6.x 这条线

这会导致 CUDA 设备枚举看起来“有卡”，但真正初始化失败。

### 6.1 正确思路

Jetson 上不要按 PC 的方式装 PyTorch。  
优先使用：

- NVIDIA 官方 Jetson PyTorch 安装方案
- 或 Ultralytics Jetson 指南里给出的 ARM64 wheel

### 6.2 最小检查命令

```bash
cat /etc/nv_tegra_release

python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda:", torch.version.cuda)
print("available:", torch.cuda.is_available())
print("count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("name:", torch.cuda.get_device_name(0))
PY
```

只要 `available: True`，再继续折腾 `device=0`。

### 6.3 一个常见修复思路

```bash
source ~/yolo/.venv/bin/activate
pip uninstall -y torch torchvision
pip cache purge
```

然后按照 JetPack 对应版本，重新安装匹配的 Jetson wheel，而不是使用普通 PyPI 自动解析出来的通用 CUDA wheel。

## 7. `onnxruntime-gpu` 装不上：Jetson 上这是常见现象

在导出 TensorRT 时，前面出现过：

```text
ERROR: Could not find a version that satisfies the requirement onnxruntime-gpu
ERROR: No matching distribution found for onnxruntime-gpu
```

这在 Jetson 上很常见，因为 PyPI 上通常没有适用于 Jetson `aarch64` 的通用 `onnxruntime-gpu` wheel。

这不代表 TensorRT 一定不能导。

可以分两种处理：

### 7.1 手动安装 Jetson 可用 wheel

如果 Ultralytics 文档给了对应的 Jetson `onnxruntime_gpu` wheel，可以手动安装。

### 7.2 先绕开 simplify

有些情况下只是 ONNX simplify 这步在检查 `onnxruntime-gpu`。  
可以先试：

```bash
yolo export model=yolo26n.pt format=engine half=True device=0 simplify=False
```

## 8. TensorRT 导出时报 `CUDA initialization failure with error: 35`

这是前面最关键的一类问题。

典型报错：

```text
TensorRT: starting export with TensorRT 10.16.1.11...
createInferBuilder: Error Code 6: API Usage Error (CUDA initialization failure with error: 35)
TypeError: pybind11::init(): factory function returned nullptr
```

这通常说明：

- Python 里导入的 TensorRT 版本，和系统 JetPack 自带的 CUDA/驱动栈不匹配
- 或者虚拟环境里混入了 `pip install tensorrt` 之类的新版本包，覆盖了系统 TensorRT

### 8.1 一个非常重要的判断思路

如果你的系统是 `JetPack 6.x / R36.x`，但 Python 里导入的是一个明显更激进的新版本 TensorRT，例如：

- `TensorRT 10.16.1.11`

那就要高度怀疑是：

- venv 里装进了不该装的 pip TensorRT
- 它和系统自带 TensorRT 冲突了

### 8.2 先查清楚到底导入了谁

```bash
cat /etc/nv_tegra_release

python - <<'PY'
import tensorrt as trt
print(trt.__version__)
print(trt.__file__)
PY

trtexec --version
echo "$LD_LIBRARY_PATH"
dpkg -l | egrep 'python3-libnvinfer|tensorrt|libnvinfer'
```

### 8.3 常见修复路径

先卸掉 venv 里的 pip TensorRT：

```bash
source ~/yolo/.venv/bin/activate
pip uninstall -y tensorrt tensorrt-cu13 tensorrt-cu12 tensorrt_dispatch tensorrt_lean
deactivate
```

然后确保系统包存在：

```bash
sudo apt update
sudo apt install -y tensorrt tensorrt-libs python3-libnvinfer python3-libnvinfer-dev
```

再重新建一个可以访问系统包的虚拟环境：

```bash
cd ~/yolo
python3 -m venv --system-site-packages .venv
source .venv/bin/activate

unset LD_LIBRARY_PATH
unset CUDA_HOME
unset CUDA_PATH
```

最后再装 Jetson 兼容的 PyTorch 和 `ultralytics`。

### 8.4 导出前先做最小验证

```bash
python - <<'PY'
import torch
import tensorrt as trt
print("torch:", torch.__version__, "cuda:", torch.version.cuda, "available:", torch.cuda.is_available())
print("tensorrt:", trt.__version__, trt.__file__)
print("builder:", trt.Builder(trt.Logger()))
PY
```

只要这里能成功创建 `trt.Builder(...)`，再去导出 `engine`。

## 9. 推荐的排查顺序

遇到 Jetson 上的 YOLO 问题时，建议按这个顺序查：

1. 摄像头本身是否工作：`v4l2-ctl` / `gst-launch-1.0`
2. 模型文件是否完整：`file` / `ls -lh`
3. 当前是否有显示环境：`echo $DISPLAY`
4. PyTorch 是否识别 CUDA：`torch.cuda.is_available()`
5. TensorRT 版本是不是系统版本：`python import tensorrt` + `trtexec --version`
6. 再考虑导出 `engine`

这样可以把问题一层层拆开，不容易误判。

## 10. 一套更稳的最小工作流

### 10.1 先跑通摄像头 + YOLO

```bash
yolo predict model=yolo26n.pt source=0 save=True imgsz=640 device=0
```

### 10.2 再验证 CUDA 和 TensorRT

```bash
python - <<'PY'
import torch
import tensorrt as trt
print(torch.cuda.is_available())
print(trt.__version__)
PY
```

### 10.3 最后导出 TensorRT

```bash
yolo export model=yolo26n.pt format=engine half=True device=0 simplify=False
```

### 10.4 如果 Python 导出还有问题，先试系统 `trtexec`

```bash
yolo export model=yolo26n.pt format=onnx device=0 simplify=False
trtexec --onnx=yolo26n.onnx --saveEngine=yolo26n.engine --fp16
```

如果 `trtexec` 可以工作，而 Python 导出失败，问题就更可能在 Python 包环境，而不是系统 CUDA 本身。

## 11. 实际经验总结

- Jetson 上最容易踩坑的不是 YOLO 本身，而是 Python 包和系统 CUDA/TensorRT 的匹配关系
- `show=True` 报错很多时候只是没有桌面环境，不是模型推理失败
- `torch.cuda.device_count() == 1` 不代表 CUDA 就一定可用，关键要看 `torch.cuda.is_available()`
- `onnxruntime-gpu` 在 Jetson 上拿不到并不少见，不要看到这个就判断整条导出链路彻底坏了
- 只要怀疑 TensorRT 版本串了，先查 `trt.__file__` 和 `trtexec --version`
- 在 Jetson 上导出 `TensorRT engine`，尽量在目标板子本机上完成

## 12. 参考资料

- Ultralytics Jetson 指南: https://docs.ultralytics.com/guides/nvidia-jetson/
- Ultralytics Predict 模式: https://docs.ultralytics.com/modes/predict/
- Ultralytics TensorRT 导出: https://docs.ultralytics.com/integrations/tensorrt/
- NVIDIA Jetson Accelerated GStreamer: https://docs.nvidia.com/jetson/archives/r36.4.4/DeveloperGuide/SD/Multimedia/AcceleratedGstreamer.html
- NVIDIA JetPack 6.1 Release Notes: https://docs.nvidia.com/jetson/jetpack/6.1/release-notes/index.html
- NVIDIA JetPack 6.2 Release Notes: https://docs.nvidia.com/jetson/jetpack/6.2/release-notes/index.html
- NVIDIA Jetson PyTorch 安装说明: https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform/index.html
- NVIDIA Jetson PyTorch 发布说明: https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform-release-notes/pytorch-jetson-rel.html
- NVIDIA TensorRT 文档: https://docs.nvidia.com/deeplearning/tensorrt/latest/
- NVIDIA 开发者论坛，Cheese/JPEG 问题: https://forums.developer.nvidia.com/t/open-cheese-failed-in-jetpack-6-0/304314
- NVIDIA 开发者论坛，Cheese on 36.4: https://forums.developer.nvidia.com/t/cheese-does-not-work-well-on-36-4/322880

