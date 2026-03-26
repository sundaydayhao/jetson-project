#!/usr/bin/env python3
"""
示例 08：PWM 舵机控制
使用 Jetson Nano GPIO PWM 控制标准舵机
支持单舵机扫描、角度控制和交互模式
"""

import Jetson.GPIO as GPIO
import time
import sys
import argparse

# ── 引脚配置 ──────────────────────────────────────────────────────────────────
SERVO_PIN_1 = 33   # BOARD 编号，对应 GPIO13/PWM1
SERVO_PIN_2 = 32   # BOARD 编号，对应 GPIO12/PWM0（多舵机示例）

# ── PWM 参数 ──────────────────────────────────────────────────────────────────
PWM_FREQ = 50       # PWM 频率 50Hz（周期 20ms）

# 舵机脉宽范围（单位：毫秒）
SERVO_MIN_PULSE = 0.5   # 0° 对应 0.5ms
SERVO_MAX_PULSE = 2.5   # 180° 对应 2.5ms
SERVO_MID_PULSE = 1.5   # 90° 对应 1.5ms

# 角度范围
SERVO_MIN_ANGLE = 0
SERVO_MAX_ANGLE = 180


def angle_to_duty_cycle(angle):
    """
    将角度转换为 PWM 占空比
    占空比 = 脉宽(ms) / 周期(ms) * 100%
    周期 = 1000ms / 50Hz = 20ms

    :param angle: 目标角度（0~180°）
    :return: 占空比（%）
    """
    angle = max(SERVO_MIN_ANGLE, min(SERVO_MAX_ANGLE, angle))
    pulse_ms = SERVO_MIN_PULSE + (SERVO_MAX_PULSE - SERVO_MIN_PULSE) * \
               (angle / SERVO_MAX_ANGLE)
    duty_cycle = pulse_ms / (1000.0 / PWM_FREQ) * 100.0
    return duty_cycle


def duty_cycle_to_angle(duty_cycle):
    """将占空比转换回角度"""
    pulse_ms = duty_cycle * (1000.0 / PWM_FREQ) / 100.0
    angle = (pulse_ms - SERVO_MIN_PULSE) / \
            (SERVO_MAX_PULSE - SERVO_MIN_PULSE) * SERVO_MAX_ANGLE
    return max(SERVO_MIN_ANGLE, min(SERVO_MAX_ANGLE, angle))


class Servo:
    """舵机控制类"""

    def __init__(self, pin, name="Servo"):
        """
        :param pin: GPIO 引脚号（BOARD 编号）
        :param name: 舵机名称（用于日志）
        """
        self.pin = pin
        self.name = name
        self.pwm = None
        self.current_angle = 90  # 初始角度（中位）
        self._setup()

    def _setup(self):
        """初始化 GPIO 和 PWM"""
        GPIO.setup(self.pin, GPIO.OUT)
        self.pwm = GPIO.PWM(self.pin, PWM_FREQ)
        # 从中位开始
        self.pwm.start(angle_to_duty_cycle(90))
        print(f"[{self.name}] 初始化完成，引脚 PIN {self.pin}，"
              f"频率 {PWM_FREQ}Hz，初始角度 90°")

    def set_angle(self, angle, smooth=False, step=2, delay=0.02):
        """
        设置舵机角度
        :param angle: 目标角度（0~180°）
        :param smooth: 是否平滑运动
        :param step: 平滑运动步长（度）
        :param delay: 平滑运动每步延迟（秒）
        """
        angle = max(SERVO_MIN_ANGLE, min(SERVO_MAX_ANGLE, int(angle)))

        if smooth and abs(angle - self.current_angle) > step:
            # 平滑运动
            direction = 1 if angle > self.current_angle else -1
            current = self.current_angle
            while abs(current - angle) > step:
                current += direction * step
                duty = angle_to_duty_cycle(current)
                self.pwm.ChangeDutyCycle(duty)
                time.sleep(delay)
            # 最终精确到目标角度
        duty = angle_to_duty_cycle(angle)
        self.pwm.ChangeDutyCycle(duty)
        self.current_angle = angle

    def get_angle(self):
        """获取当前角度"""
        return self.current_angle

    def center(self):
        """回到中位（90°）"""
        self.set_angle(90, smooth=True)
        print(f"[{self.name}] 回到中位 90°")

    def stop(self):
        """停止 PWM 输出（舵机断电，可自由转动）"""
        self.pwm.ChangeDutyCycle(0)

    def cleanup(self):
        """释放资源"""
        self.stop()
        self.pwm.stop()
        print(f"[{self.name}] 资源已释放")


def demo_scan(servo, start=0, end=180, step=1, delay=0.02):
    """
    扫描演示：从 start 扫描到 end，再回到 start
    :param step: 每步角度
    :param delay: 每步延迟（秒）
    """
    print(f"\n[扫描] {start}° → {end}° → {start}°")

    # 正向扫描
    for angle in range(start, end + 1, step):
        servo.set_angle(angle)
        time.sleep(delay)

    time.sleep(0.5)

    # 反向扫描
    for angle in range(end, start - 1, -step):
        servo.set_angle(angle)
        time.sleep(delay)

    print(f"[扫描] 完成")


def demo_positions(servo):
    """演示几个关键位置"""
    positions = [
        (0,   "最左（0°）"),
        (45,  "左45°"),
        (90,  "中位（90°）"),
        (135, "右135°"),
        (180, "最右（180°）"),
        (90,  "回中位（90°）"),
    ]

    print("\n[演示] 关键位置演示")
    for angle, desc in positions:
        print(f"  → {desc}")
        servo.set_angle(angle, smooth=True)
        time.sleep(1.0)


def demo_wave(servo, cycles=3):
    """波浪运动演示"""
    import math
    print(f"\n[波浪] 正弦波运动，{cycles} 个周期")

    steps = 100
    for cycle in range(cycles):
        for i in range(steps):
            # 正弦波：0° ~ 180°
            angle = 90 + 90 * math.sin(2 * math.pi * i / steps)
            servo.set_angle(int(angle))
            time.sleep(0.02)

    servo.set_angle(90)
    print("[波浪] 完成")


def interactive_mode(servo):
    """交互模式：手动输入角度"""
    print("\n[交互] 输入角度（0~180）控制舵机")
    print("  输入 'c' 回中位，'s' 扫描，'q' 退出\n")

    while True:
        try:
            user_input = input(f"  当前角度 {servo.get_angle()}°，输入命令: ").strip()

            if user_input.lower() == 'q':
                print("[退出] 交互模式结束")
                break
            elif user_input.lower() == 'c':
                servo.center()
            elif user_input.lower() == 's':
                demo_scan(servo)
            else:
                angle = float(user_input)
                servo.set_angle(angle, smooth=True)
                print(f"  → 移动到 {servo.get_angle()}°")

        except ValueError:
            print(f"  [错误] 无效输入，请输入 0~180 的数字")
        except KeyboardInterrupt:
            print("\n[中断] Ctrl+C")
            break


def multi_servo_demo():
    """多舵机协调控制演示"""
    print("\n[多舵机] 双舵机协调控制演示")

    servo1 = Servo(SERVO_PIN_1, name="舵机1")
    servo2 = Servo(SERVO_PIN_2, name="舵机2")

    try:
        # 同向运动
        print("\n  同向运动...")
        for angle in range(0, 181, 3):
            servo1.set_angle(angle)
            servo2.set_angle(angle)
            time.sleep(0.02)

        time.sleep(0.5)

        # 反向运动
        print("  反向运动...")
        for angle in range(0, 181, 3):
            servo1.set_angle(angle)
            servo2.set_angle(180 - angle)
            time.sleep(0.02)

        time.sleep(0.5)

        # 回中位
        servo1.center()
        servo2.center()
        time.sleep(1)

    finally:
        servo1.cleanup()
        servo2.cleanup()


def parse_args():
    parser = argparse.ArgumentParser(description="Jetson Nano PWM 舵机控制示例")
    parser.add_argument("--pin", type=int, default=SERVO_PIN_1,
                        help=f"舵机引脚（BOARD 编号，默认: {SERVO_PIN_1}）")
    parser.add_argument("--angle", type=float, default=None,
                        help="直接设置角度（0~180）")
    parser.add_argument("--mode", choices=["scan", "demo", "wave", "interactive", "multi"],
                        default="scan", help="运行模式（默认: scan）")
    return parser.parse_args()


def main():
    print("=" * 55)
    print("  Jetson Nano PWM 舵机控制示例")
    print("=" * 55)

    args = parse_args()

    print(f"  舵机引脚: PIN {args.pin}")
    print(f"  PWM 频率: {PWM_FREQ} Hz")
    print(f"  脉宽范围: {SERVO_MIN_PULSE}ms ~ {SERVO_MAX_PULSE}ms")
    print(f"  角度范围: {SERVO_MIN_ANGLE}° ~ {SERVO_MAX_ANGLE}°")
    print("=" * 55)

    # 初始化 GPIO
    GPIO.setmode(GPIO.BOARD)

    if args.mode == "multi":
        try:
            multi_servo_demo()
        finally:
            GPIO.cleanup()
        return

    servo = Servo(args.pin)

    try:
        if args.angle is not None:
            # 直接设置角度
            print(f"\n[设置] 移动到 {args.angle}°")
            servo.set_angle(args.angle, smooth=True)
            time.sleep(2)
        elif args.mode == "scan":
            demo_scan(servo)
        elif args.mode == "demo":
            demo_positions(servo)
        elif args.mode == "wave":
            demo_wave(servo)
        elif args.mode == "interactive":
            interactive_mode(servo)

    except KeyboardInterrupt:
        print("\n[中断] Ctrl+C 退出")
    finally:
        servo.cleanup()
        GPIO.cleanup()
        print("[清理] GPIO 资源已释放")


if __name__ == "__main__":
    main()