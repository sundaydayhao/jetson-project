# 示例 02：GPIO 按键输入检测

## 功能说明

通过 Jetson Nano 的 GPIO 引脚读取按键状态，演示数字输入检测，包括轮询和中断两种方式。

## 硬件连接

```
Jetson Nano J41 引脚:
  PIN 11 (GPIO17) ──── LED 正极 ──── 220Ω ──── GND
  PIN 13 (GPIO27) ──── 按键一端
  PIN 14 (GND)    ──── 按键另一端（内部上拉，按下为低电平）
```

## 电路图

```
  3.3V ──[10kΩ上拉]──┬── [PIN 13 GPIO27]
                     │
                   [按键]
                     │
                   [GND]

  [PIN 11 GPIO17] ──[220Ω]──[LED+]──[LED-]── [GND]
```

> 注意：Jetson.GPIO 支持内部上拉电阻，可以不接外部上拉电阻

## 依赖安装

```bash
sudo pip3 install Jetson.GPIO
```

## 运行方式

```bash
# 轮询模式（默认）
sudo python3 main.py

# 中断模式
sudo python3 main.py interrupt

# 事件检测模式
sudo python3 main.py event
```

## 预期效果

- 按下按键时 LED 亮起，松开时 LED 熄灭
- 终端显示按键状态变化
- 按 Ctrl+C 退出程序

## 知识点

- GPIO 输入模式配置
- 内部上拉/下拉电阻设置
- 轮询方式读取 GPIO 状态
- 中断（回调）方式检测 GPIO 变化
- 按键消抖处理