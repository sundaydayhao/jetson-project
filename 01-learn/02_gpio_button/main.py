#!/usr/bin/env python3
"""
示例 02：GPIO 按键输入检测
硬件：Jetson Nano + 按键 + LED + 220Ω 电阻
引脚：PIN 13 (GPIO27) 读取按键，PIN 11 (GPIO17) 控制 LED
支持三种模式：轮询、中断、事件检测
"""

import Jetson.GPIO as GPIO
import time
import sys

# ── 引脚配置 ──────────────────────────────────────────────────────────────────
BUTTON_PIN = 13   # 按键引脚（BOARD 编号，对应 GPIO27）
LED_PIN = 11      # LED 引脚（BOARD 编号，对应 GPIO17）
DEBOUNCE_MS = 200  # 消抖时间（毫秒）

# 全局状态
press_count = 0
last_state = GPIO.HIGH


def setup():
    """初始化 GPIO"""
    GPIO.setmode(GPIO.BOARD)
    # 按键引脚：输入模式，启用内部上拉电阻（未按下时为高电平）
    GPIO.setup(BUTTON_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)
    # LED 引脚：输出模式，初始熄灭
    GPIO.setup(LED_PIN, GPIO.OUT, initial=GPIO.LOW)
    print(f"[初始化] 按键引脚: PIN {BUTTON_PIN}，LED 引脚: PIN {LED_PIN}")
    print("[初始化] 按键使用内部上拉电阻（按下为低电平）")


def polling_mode():
    """
    轮询模式：循环读取按键状态
    优点：简单直观
    缺点：占用 CPU，可能错过快速按键
    """
    global press_count, last_state
    print("\n[轮询模式] 开始检测按键（Ctrl+C 退出）\n")

    try:
        while True:
            current_state = GPIO.input(BUTTON_PIN)

            # 检测下降沿（按下）
            if current_state == GPIO.LOW and last_state == GPIO.HIGH:
                press_count += 1
                GPIO.output(LED_PIN, GPIO.HIGH)
                print(f"  ▼ 按键按下！（第 {press_count} 次）")

            # 检测上升沿（松开）
            elif current_state == GPIO.HIGH and last_state == GPIO.LOW:
                GPIO.output(LED_PIN, GPIO.LOW)
                print(f"  ▲ 按键松开")

            last_state = current_state
            time.sleep(0.05)  # 50ms 轮询间隔

    except KeyboardInterrupt:
        print(f"\n[统计] 共按下 {press_count} 次")


def interrupt_mode():
    """
    中断模式：使用回调函数响应按键事件
    优点：不占用 CPU，响应及时
    缺点：回调函数在独立线程中运行
    """
    global press_count
    print("\n[中断模式] 开始检测按键（Ctrl+C 退出）\n")

    def button_pressed_callback(channel):
        """按键按下回调（下降沿触发）"""
        global press_count
        press_count += 1
        GPIO.output(LED_PIN, GPIO.HIGH)
        print(f"  ▼ 按键按下！（第 {press_count} 次）[中断回调]")

    def button_released_callback(channel):
        """按键松开回调（上升沿触发）"""
        GPIO.output(LED_PIN, GPIO.LOW)
        print(f"  ▲ 按键松开 [中断回调]")

    # 注册下降沿中断（按下）
    GPIO.add_event_detect(
        BUTTON_PIN,
        GPIO.FALLING,
        callback=button_pressed_callback,
        bouncetime=DEBOUNCE_MS
    )

    # 注意：Jetson.GPIO 目前对同一引脚只支持一个 add_event_detect
    # 如需同时检测上升沿，可在回调中判断当前状态

    try:
        print("  等待按键事件...")
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print(f"\n[统计] 共按下 {press_count} 次")
    finally:
        GPIO.remove_event_detect(BUTTON_PIN)


def event_mode():
    """
    事件检测模式：使用 wait_for_edge 阻塞等待
    优点：精确等待特定边沿事件
    缺点：每次只能等待一个事件
    """
    global press_count
    print("\n[事件检测模式] 开始检测按键（Ctrl+C 退出）\n")

    try:
        while True:
            print("  等待按键按下...")
            # 阻塞等待下降沿（按下），超时 5 秒
            channel = GPIO.wait_for_edge(
                BUTTON_PIN,
                GPIO.FALLING,
                timeout=5000  # 毫秒
            )

            if channel is None:
                print("  [超时] 5 秒内未检测到按键")
                continue

            press_count += 1
            GPIO.output(LED_PIN, GPIO.HIGH)
            print(f"  ▼ 按键按下！（第 {press_count} 次）")

            # 等待松开（上升沿）
            GPIO.wait_for_edge(BUTTON_PIN, GPIO.RISING, timeout=10000)
            GPIO.output(LED_PIN, GPIO.LOW)
            print(f"  ▲ 按键松开")

    except KeyboardInterrupt:
        print(f"\n[统计] 共按下 {press_count} 次")


def long_press_demo():
    """
    长按/短按检测演示
    短按（< 1秒）：LED 闪烁一次
    长按（>= 1秒）：LED 持续亮 3 秒
    """
    print("\n[长按/短按模式] 开始检测（Ctrl+C 退出）\n")
    print("  短按（< 1秒）：LED 快闪")
    print("  长按（>= 1秒）：LED 长亮 3 秒\n")

    try:
        while True:
            # 等待按下
            GPIO.wait_for_edge(BUTTON_PIN, GPIO.FALLING, timeout=5000)
            press_time = time.time()
            GPIO.output(LED_PIN, GPIO.HIGH)

            # 等待松开
            GPIO.wait_for_edge(BUTTON_PIN, GPIO.RISING, timeout=10000)
            duration = time.time() - press_time

            if duration >= 1.0:
                print(f"  长按检测！持续 {duration:.2f} 秒 → LED 长亮 3 秒")
                time.sleep(3)
                GPIO.output(LED_PIN, GPIO.LOW)
            else:
                print(f"  短按检测！持续 {duration:.2f} 秒 → LED 快闪")
                GPIO.output(LED_PIN, GPIO.LOW)
                time.sleep(0.1)
                GPIO.output(LED_PIN, GPIO.HIGH)
                time.sleep(0.1)
                GPIO.output(LED_PIN, GPIO.LOW)

    except KeyboardInterrupt:
        print("\n[退出] 程序结束")


def cleanup():
    """清理 GPIO 资源"""
    GPIO.output(LED_PIN, GPIO.LOW)
    GPIO.cleanup()
    print("[清理] GPIO 资源已释放")


def main():
    print("=" * 50)
    print("  Jetson Nano GPIO 按键输入示例")
    print("=" * 50)
    print(f"  按键引脚: PIN {BUTTON_PIN} (GPIO27)")
    print(f"  LED 引脚: PIN {LED_PIN} (GPIO17)")
    print(f"  消抖时间: {DEBOUNCE_MS} ms")
    print("=" * 50)

    setup()

    mode = sys.argv[1] if len(sys.argv) > 1 else "polling"

    if mode == "interrupt":
        interrupt_mode()
    elif mode == "event":
        event_mode()
    elif mode == "longpress":
        long_press_demo()
    else:
        polling_mode()

    cleanup()


if __name__ == "__main__":
    main()