# 示例 08：PWM 舵机控制

## 功能说明

使用 Jetson Nano 的 PWM 引脚控制舵机，演示 PWM 信号生成、舵机角度控制和扫描运动。

## 硬件连接

```
舵机（SG90/MG996R）  Jetson Nano J41
  红线（VCC）  ────── 外部 5V 电源（不要用 Jetson 的 5V！）
  棕线（GND）  ────── PIN 6 (GND) + 外部电源 GND（共地）
  橙线（信号） ────── PIN 33 (GPIO13/PWM1)
```

> ⚠️ 重要：舵机需要较大电流，必须使用外部 5V 电源供电，
> 不能直接使用 Jetson Nano 的 5V 引脚（会导致系统不稳定）

## PWM 参数说明

```
标准舵机 PWM 参数：
  频率：50 Hz（周期 20ms）
  脉宽范围：
    0.5ms（500μs）  → 0°
    1.5ms（1500μs） → 90°（中位）
    2.5ms（2500μs） → 180°
```

## 依赖安装

```bash
pip3 install Jetson.GPIO
```

## 运行方式

```bash
# 舵机扫描演示（0° → 180° → 0°）
sudo python3 main.py

# 指定角度
sudo python3 main.py --angle 90

# 交互模式（手动输入角度）
sudo python3 main.py --interactive

# 多舵机控制
sudo python3 main.py --multi
```

## 预期效果

- 舵机从 0° 缓慢扫描到 180°，再回到 0°
- 交互模式下可实时输入角度控制舵机

## 知识点

- PWM 信号原理（频率、占空比、脉宽）
- Jetson.GPIO PWM 类使用
- 舵机控制原理
- 角度到脉宽的转换计算
- 多舵机协调控制