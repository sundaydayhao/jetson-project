#!/usr/bin/env python3
"""
示例 01：GPIO 控制 LED 闪烁
硬件：Jetson Nano + LED + 220Ω 电阻
引脚：PIN 11 (GPIO17) 控制 LED
"""

import Jetson.GPIO as GPIO
import time
import sys

# ── 引脚配置 ──────────────────────────────────────────────────────────────────
LED_PIN = 11        # 使用 BOARD 编号，对应 GPIO17
BLINK_INTERVAL = 1  # 闪烁间隔（秒）


def setup():
    """初始化 GPIO"""
    # 使用物理引脚编号（BOARD 模式）
    GPIO.setmode(GPIO.BOARD)
    # 设置 LED 引脚为输出模式，初始状态为低电平（LED 熄灭）
    GPIO.setup(LED_PIN, GPIO.OUT, initial=GPIO.LOW)
    print(f"[初始化] GPIO 引脚 {LED_PIN} 已设置为输出模式")


def blink(count=None):
    """
    控制 LED 闪烁
    :param count: 闪烁次数，None 表示无限循环
    """
    blink_count = 0
    try:
        while True:
            # 点亮 LED
            GPIO.output(LED_PIN, GPIO.HIGH)
            print(f"[第 {blink_count + 1} 次] LED 亮")
            time.sleep(BLINK_INTERVAL)

            # 熄灭 LED
            GPIO.output(LED_PIN, GPIO.LOW)
            print(f"[第 {blink_count + 1} 次] LED 灭")
            time.sleep(BLINK_INTERVAL)

            blink_count += 1

            # 如果指定了次数，达到后退出
            if count is not None and blink_count >= count:
                print(f"\n已完成 {count} 次闪烁，程序结束")
                break

    except KeyboardInterrupt:
        print("\n[中断] 检测到 Ctrl+C，程序退出")


def cleanup():
    """清理 GPIO 资源"""
    GPIO.output(LED_PIN, GPIO.LOW)  # 确保 LED 熄灭
    GPIO.cleanup()
    print("[清理] GPIO 资源已释放")


def demo_pattern():
    """演示不同的闪烁模式"""
    print("\n=== 演示模式 1：快速闪烁（0.2秒）===")
    for _ in range(5):
        GPIO.output(LED_PIN, GPIO.HIGH)
        time.sleep(0.2)
        GPIO.output(LED_PIN, GPIO.LOW)
        time.sleep(0.2)

    print("\n=== 演示模式 2：慢速闪烁（1秒）===")
    for _ in range(3):
        GPIO.output(LED_PIN, GPIO.HIGH)
        time.sleep(1.0)
        GPIO.output(LED_PIN, GPIO.LOW)
        time.sleep(1.0)

    print("\n=== 演示模式 3：SOS 摩斯电码 ===")
    sos_pattern(repeat=2)


def sos_pattern(repeat=1):
    """
    发送 SOS 摩斯电码（... --- ...）
    短闪 = 0.2s，长闪 = 0.6s
    """
    dot = 0.2    # 短闪时长
    dash = 0.6   # 长闪时长
    gap = 0.2    # 符号间隔
    letter_gap = 0.6  # 字母间隔

    def flash(duration):
        GPIO.output(LED_PIN, GPIO.HIGH)
        time.sleep(duration)
        GPIO.output(LED_PIN, GPIO.LOW)
        time.sleep(gap)

    for _ in range(repeat):
        print("  S (···)")
        for _ in range(3):
            flash(dot)
        time.sleep(letter_gap)

        print("  O (───)")
        for _ in range(3):
            flash(dash)
        time.sleep(letter_gap)

        print("  S (···)")
        for _ in range(3):
            flash(dot)
        time.sleep(letter_gap * 2)


def main():
    print("=" * 50)
    print("  Jetson Nano GPIO LED 控制示例")
    print("=" * 50)
    print(f"  控制引脚: PIN {LED_PIN} (GPIO17)")
    print(f"  闪烁间隔: {BLINK_INTERVAL} 秒")
    print("  按 Ctrl+C 退出")
    print("=" * 50)

    setup()

    # 检查命令行参数
    if len(sys.argv) > 1:
        if sys.argv[1] == "demo":
            # 演示模式
            print("\n[演示模式] 展示不同闪烁效果")
            demo_pattern()
        elif sys.argv[1] == "sos":
            # SOS 模式
            print("\n[SOS 模式] 发送 SOS 摩斯电码")
            sos_pattern(repeat=3)
        else:
            # 指定次数
            try:
                count = int(sys.argv[1])
                print(f"\n[计数模式] 闪烁 {count} 次")
                blink(count=count)
            except ValueError:
                print(f"[错误] 无效参数: {sys.argv[1]}")
                print("用法: python3 main.py [次数|demo|sos]")
    else:
        # 默认：无限闪烁
        print("\n[正常模式] 开始无限闪烁（Ctrl+C 退出）\n")
        blink()

    cleanup()


if __name__ == "__main__":
    main()