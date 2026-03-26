# 示例 01：GPIO 控制 LED 闪烁

## 功能说明

通过 Jetson Nano 的 GPIO 引脚控制 LED 灯的亮灭，演示基本的数字输出控制。

## 硬件连接

```
Jetson Nano J41 引脚:
  PIN 11 (GPIO17) ──── 220Ω 电阻 ──── LED 正极(长脚)
  PIN 9  (GND)    ──── LED 负极(短脚)
```

## 电路图

```
  [PIN 11 GPIO17] ---[220Ω]---[LED+]---[LED-]--- [GND PIN 9]
```

## 依赖安装

```bash
pip3 install Jetson.GPIO
```

或者使用系统自带版本：
```bash
sudo pip3 install Jetson.GPIO
```

将用户加入 gpio 组（避免每次使用 sudo）：
```bash
sudo groupadd -f -r gpio
sudo usermod -a -G gpio $USER
# 重新登录后生效
```

## 运行方式

```bash
# 方式1：直接运行（需要 sudo）
sudo python3 main.py

# 方式2：加入 gpio 组后运行
python3 main.py
```

## 预期效果

- LED 每隔 1 秒交替亮灭
- 按 Ctrl+C 退出程序，LED 自动关闭

## 知识点

- `Jetson.GPIO` 库的基本使用
- GPIO 引脚模式设置（BOARD / BCM）
- 数字输出控制
- 程序退出时的资源清理