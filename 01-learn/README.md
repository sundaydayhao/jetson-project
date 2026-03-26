# Jetson Nano 开发示例集

本项目为 NVIDIA Jetson Nano 开发者提供一系列实用示例，涵盖 GPIO 控制、摄像头采集、深度学习推理、传感器通信等常见应用场景。

## 环境要求

- NVIDIA Jetson Nano（2GB 或 4GB 版本）
- JetPack 4.6.x（包含 CUDA、cuDNN、TensorRT）
- Python 3.6+
- OpenCV 4.x（JetPack 自带）

## 示例列表

| 编号 | 目录 | 说明 |
|------|------|------|
| 01 | [01_gpio_led](./01_gpio_led/) | GPIO 控制 LED 闪烁 |
| 02 | [02_gpio_button](./02_gpio_button/) | GPIO 按键输入检测 |
| 03 | [03_camera_capture](./03_camera_capture/) | 摄像头图像采集与显示 |
| 04 | [04_face_detection](./04_face_detection/) | OpenCV 人脸检测 |
| 05 | [05_object_detection](./05_object_detection/) | SSD 目标检测（TensorRT） |
| 06 | [06_tensorrt_inference](./06_tensorrt_inference/) | TensorRT 模型推理 |
| 07 | [07_i2c_sensor](./07_i2c_sensor/) | I2C 传感器数据读取（MPU6050） |
| 08 | [08_pwm_servo](./08_pwm_servo/) | PWM 舵机控制 |
| 09 | [09_cuda_example](./09_cuda_example/) | CUDA 并行计算加速 |
| 10 | [10_system_monitor](./10_system_monitor/) | Jetson 系统资源监控 |

## 快速开始

```bash
# 克隆项目
git clone <repo_url>
cd jetson-learn

# 进入某个示例目录
cd 01_gpio_led

# 查看该示例的说明
cat README.md

# 安装依赖（如有）
pip3 install -r requirements.txt

# 运行示例
python3 main.py
```

## Jetson Nano 引脚图

```
                    J41 Header
         3.3V  [ 1] [ 2]  5V
   SDA1/GPIO2  [ 3] [ 4]  5V
   SCL1/GPIO3  [ 5] [ 6]  GND
        GPIO4  [ 7] [ 8]  UART_TX
          GND  [ 9] [10]  UART_RX
       GPIO17  [11] [12]  GPIO18/PWM0
       GPIO27  [13] [14]  GND
       GPIO22  [15] [16]  GPIO23
         3.3V  [17] [18]  GPIO24
  SPI_MOSI/10  [19] [20]  GND
  SPI_MISO/ 9  [21] [22]  GPIO25
  SPI_CLK/11   [23] [24]  SPI_CE0/ 8
          GND  [25] [26]  SPI_CE1/ 7
        ID_SD  [27] [28]  ID_SC
        GPIO5  [29] [30]  GND
        GPIO6  [31] [32]  GPIO12/PWM0
  GPIO13/PWM1  [33] [34]  GND
  GPIO19/MISO  [35] [36]  GPIO16
       GPIO26  [37] [38]  GPIO20/MOSI
          GND  [39] [40]  GPIO21/SCLK
```

## 注意事项

1. GPIO 操作需要 root 权限或将用户加入 `gpio` 组
2. 摄像头示例支持 USB 摄像头和 CSI 摄像头
3. TensorRT 示例需要提前转换模型
4. 运行前请确认硬件连接正确，避免短路损坏设备

## 参考资源

- [Jetson Nano 官方文档](https://developer.nvidia.com/embedded/learn/get-started-jetson-nano-devkit)
- [JetPack SDK](https://developer.nvidia.com/embedded/jetpack)
- [Jetson Zoo（预编译包）](https://elinux.org/Jetson_Zoo)
- [NVIDIA 深度学习示例](https://github.com/NVIDIA-AI-IOT)