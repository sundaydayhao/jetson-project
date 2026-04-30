# 示例 07：I2C 传感器数据读取（MPU6050）

## 功能说明

通过 I2C 总线读取 MPU6050 六轴惯性测量单元（IMU）的加速度计和陀螺仪数据，演示 Jetson Nano 的 I2C 通信。

## 硬件连接

```
MPU6050 模块    Jetson Nano J41
  VCC    ────── PIN 1  (3.3V)
  GND    ────── PIN 6  (GND)
  SCL    ────── PIN 5  (SCL1/GPIO3)
  SDA    ────── PIN 3  (SDA1/GPIO2)
  AD0    ────── GND    (I2C 地址 0x68)
```

> 注意：AD0 接 GND 时 I2C 地址为 0x68，接 VCC 时为 0x69

## MPU6050 寄存器说明

| 寄存器 | 地址 | 说明 |
|--------|------|------|
| PWR_MGMT_1 | 0x6B | 电源管理 |
| ACCEL_XOUT_H | 0x3B | 加速度 X 高字节 |
| GYRO_XOUT_H | 0x43 | 陀螺仪 X 高字节 |
| TEMP_OUT_H | 0x41 | 温度高字节 |

## 依赖安装

```bash
# 启用 I2C
sudo usermod -a -G i2c $USER
# 重新登录后生效

# 安装 smbus2
pip3 install smbus2

# 检测 I2C 设备
sudo i2cdetect -y -r 1
# 应该看到 0x68 地址
```

## 运行方式

```bash
# 读取传感器数据（默认模式）
python3 main.py

# 指定 I2C 总线号
python3 main.py --bus 1

# 指定 I2C 地址
python3 main.py --addr 0x68

# 数据记录模式（保存到 CSV）
python3 main.py --log

# 姿态角计算模式
python3 main.py --attitude
```

## 预期效果

- 实时显示加速度计（X/Y/Z）数据（单位：g）
- 实时显示陀螺仪（X/Y/Z）数据（单位：°/s）
- 实时显示温度（单位：°C）
- 可选：计算俯仰角（Pitch）和横滚角（Roll）

## 知识点

- I2C 协议基础
- smbus2 库使用
- MPU6050 寄存器操作
- 传感器数据校准
- 互补滤波器（姿态估计）