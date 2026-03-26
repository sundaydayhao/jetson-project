#!/usr/bin/env python3
"""
示例 07：I2C 传感器数据读取（MPU6050）
通过 I2C 总线读取六轴 IMU 传感器数据
支持加速度、陀螺仪、温度读取和姿态角计算
"""

import time
import math
import argparse
import csv
import os
from datetime import datetime

# ── MPU6050 寄存器地址 ────────────────────────────────────────────────────────
MPU6050_ADDR = 0x68          # I2C 地址（AD0=GND）

# 电源管理
PWR_MGMT_1   = 0x6B

# 采样率分频器
SMPLRT_DIV   = 0x19

# 配置寄存器
CONFIG       = 0x1A
GYRO_CONFIG  = 0x1B
ACCEL_CONFIG = 0x1C

# 加速度计数据寄存器（高字节）
ACCEL_XOUT_H = 0x3B
ACCEL_YOUT_H = 0x3D
ACCEL_ZOUT_H = 0x3F

# 温度数据寄存器
TEMP_OUT_H   = 0x41

# 陀螺仪数据寄存器（高字节）
GYRO_XOUT_H  = 0x43
GYRO_YOUT_H  = 0x45
GYRO_ZOUT_H  = 0x47

# 量程配置
ACCEL_RANGE_2G  = 0x00   # ±2g，灵敏度 16384 LSB/g
ACCEL_RANGE_4G  = 0x08   # ±4g，灵敏度 8192 LSB/g
ACCEL_RANGE_8G  = 0x10   # ±8g，灵敏度 4096 LSB/g
ACCEL_RANGE_16G = 0x18   # ±16g，灵敏度 2048 LSB/g

GYRO_RANGE_250  = 0x00   # ±250°/s，灵敏度 131 LSB/(°/s)
GYRO_RANGE_500  = 0x08   # ±500°/s，灵敏度 65.5 LSB/(°/s)
GYRO_RANGE_1000 = 0x10   # ±1000°/s，灵敏度 32.8 LSB/(°/s)
GYRO_RANGE_2000 = 0x18   # ±2000°/s，灵敏度 16.4 LSB/(°/s)

# 灵敏度系数
ACCEL_SENSITIVITY = {
    ACCEL_RANGE_2G:  16384.0,
    ACCEL_RANGE_4G:  8192.0,
    ACCEL_RANGE_8G:  4096.0,
    ACCEL_RANGE_16G: 2048.0,
}
GYRO_SENSITIVITY = {
    GYRO_RANGE_250:  131.0,
    GYRO_RANGE_500:  65.5,
    GYRO_RANGE_1000: 32.8,
    GYRO_RANGE_2000: 16.4,
}


class MPU6050:
    """MPU6050 六轴 IMU 传感器驱动"""

    def __init__(self, bus=1, address=MPU6050_ADDR):
        """
        初始化 MPU6050
        :param bus: I2C 总线号（Jetson Nano 通常为 1）
        :param address: I2C 地址（默认 0x68）
        """
        self.bus_num = bus
        self.address = address
        self.bus = None

        # 校准偏移量
        self.accel_offset = {"x": 0.0, "y": 0.0, "z": 0.0}
        self.gyro_offset = {"x": 0.0, "y": 0.0, "z": 0.0}

        # 当前量程配置
        self.accel_range = ACCEL_RANGE_2G
        self.gyro_range = GYRO_RANGE_250

        self._connect()
        self._initialize()

    def _connect(self):
        """连接 I2C 总线"""
        try:
            import smbus2
            self.bus = smbus2.SMBus(self.bus_num)
            print(f"[I2C] 连接总线 {self.bus_num}，地址 0x{self.address:02X}")
        except ImportError:
            raise ImportError(
                "smbus2 未安装，请运行: pip3 install smbus2"
            )
        except Exception as e:
            raise RuntimeError(
                f"I2C 连接失败: {e}\n"
                f"请检查:\n"
                f"  1. 硬件连接是否正确\n"
                f"  2. 是否已加入 i2c 组: sudo usermod -a -G i2c $USER\n"
                f"  3. 使用 i2cdetect -y -r {self.bus_num} 检测设备"
            )

    def _initialize(self):
        """初始化传感器"""
        # 唤醒传感器（清除睡眠位）
        self.bus.write_byte_data(self.address, PWR_MGMT_1, 0x00)
        time.sleep(0.1)

        # 设置采样率：1kHz / (1 + SMPLRT_DIV) = 100Hz
        self.bus.write_byte_data(self.address, SMPLRT_DIV, 0x09)

        # 设置低通滤波器（减少噪声）
        self.bus.write_byte_data(self.address, CONFIG, 0x06)

        # 设置加速度计量程 ±2g
        self.set_accel_range(ACCEL_RANGE_2G)

        # 设置陀螺仪量程 ±250°/s
        self.set_gyro_range(GYRO_RANGE_250)

        # 验证设备 ID
        who_am_i = self.bus.read_byte_data(self.address, 0x75)
        if who_am_i != 0x68:
            print(f"[警告] WHO_AM_I = 0x{who_am_i:02X}（期望 0x68）")
        else:
            print(f"[MPU6050] 初始化成功，WHO_AM_I = 0x{who_am_i:02X}")

    def set_accel_range(self, accel_range):
        """设置加速度计量程"""
        self.bus.write_byte_data(self.address, ACCEL_CONFIG, accel_range)
        self.accel_range = accel_range
        sensitivity = ACCEL_SENSITIVITY[accel_range]
        range_g = {0x00: 2, 0x08: 4, 0x10: 8, 0x18: 16}[accel_range]
        print(f"[MPU6050] 加速度计量程: ±{range_g}g，灵敏度: {sensitivity} LSB/g")

    def set_gyro_range(self, gyro_range):
        """设置陀螺仪量程"""
        self.bus.write_byte_data(self.address, GYRO_CONFIG, gyro_range)
        self.gyro_range = gyro_range
        sensitivity = GYRO_SENSITIVITY[gyro_range]
        range_dps = {0x00: 250, 0x08: 500, 0x10: 1000, 0x18: 2000}[gyro_range]
        print(f"[MPU6050] 陀螺仪量程: ±{range_dps}°/s，灵敏度: {sensitivity} LSB/(°/s)")

    def _read_raw_word(self, reg):
        """读取 16 位有符号整数（大端序）"""
        high = self.bus.read_byte_data(self.address, reg)
        low = self.bus.read_byte_data(self.address, reg + 1)
        value = (high << 8) | low
        # 转换为有符号数
        if value >= 0x8000:
            value -= 0x10000
        return value

    def read_accel_raw(self):
        """读取加速度计原始数据"""
        return {
            "x": self._read_raw_word(ACCEL_XOUT_H),
            "y": self._read_raw_word(ACCEL_YOUT_H),
            "z": self._read_raw_word(ACCEL_ZOUT_H),
        }

    def read_gyro_raw(self):
        """读取陀螺仪原始数据"""
        return {
            "x": self._read_raw_word(GYRO_XOUT_H),
            "y": self._read_raw_word(GYRO_YOUT_H),
            "z": self._read_raw_word(GYRO_ZOUT_H),
        }

    def read_temp_raw(self):
        """读取温度原始数据"""
        return self._read_raw_word(TEMP_OUT_H)

    def read_accel(self):
        """
        读取加速度（单位：g）
        返回校准后的值
        """
        raw = self.read_accel_raw()
        sensitivity = ACCEL_SENSITIVITY[self.accel_range]
        return {
            "x": raw["x"] / sensitivity - self.accel_offset["x"],
            "y": raw["y"] / sensitivity - self.accel_offset["y"],
            "z": raw["z"] / sensitivity - self.accel_offset["z"],
        }

    def read_gyro(self):
        """
        读取角速度（单位：°/s）
        返回校准后的值
        """
        raw = self.read_gyro_raw()
        sensitivity = GYRO_SENSITIVITY[self.gyro_range]
        return {
            "x": raw["x"] / sensitivity - self.gyro_offset["x"],
            "y": raw["y"] / sensitivity - self.gyro_offset["y"],
            "z": raw["z"] / sensitivity - self.gyro_offset["z"],
        }

    def read_temperature(self):
        """
        读取温度（单位：°C）
        公式：Temperature = raw / 340.0 + 36.53
        """
        raw = self.read_temp_raw()
        return raw / 340.0 + 36.53

    def read_all(self):
        """一次性读取所有数据（更高效）"""
        # 从 ACCEL_XOUT_H 开始连续读取 14 字节
        data = self.bus.read_i2c_block_data(self.address, ACCEL_XOUT_H, 14)

        def to_signed(high, low):
            val = (high << 8) | low
            return val - 0x10000 if val >= 0x8000 else val

        accel_raw = {
            "x": to_signed(data[0], data[1]),
            "y": to_signed(data[2], data[3]),
            "z": to_signed(data[4], data[5]),
        }
        temp_raw = to_signed(data[6], data[7])
        gyro_raw = {
            "x": to_signed(data[8], data[9]),
            "y": to_signed(data[10], data[11]),
            "z": to_signed(data[12], data[13]),
        }

        accel_sens = ACCEL_SENSITIVITY[self.accel_range]
        gyro_sens = GYRO_SENSITIVITY[self.gyro_range]

        return {
            "accel": {
                "x": accel_raw["x"] / accel_sens - self.accel_offset["x"],
                "y": accel_raw["y"] / accel_sens - self.accel_offset["y"],
                "z": accel_raw["z"] / accel_sens - self.accel_offset["z"],
            },
            "gyro": {
                "x": gyro_raw["x"] / gyro_sens - self.gyro_offset["x"],
                "y": gyro_raw["y"] / gyro_sens - self.gyro_offset["y"],
                "z": gyro_raw["z"] / gyro_sens - self.gyro_offset["z"],
            },
            "temperature": temp_raw / 340.0 + 36.53,
        }

    def calibrate(self, samples=200):
        """
        传感器校准（静止放置，水平面）
        计算陀螺仪零偏和加速度计偏移
        """
        print(f"\n[校准] 请将传感器水平静止放置...")
        print(f"[校准] 3 秒后开始采集 {samples} 个样本...")
        time.sleep(3)

        accel_sum = {"x": 0.0, "y": 0.0, "z": 0.0}
        gyro_sum = {"x": 0.0, "y": 0.0, "z": 0.0}

        for i in range(samples):
            data = self.read_all()
            for axis in ("x", "y", "z"):
                accel_sum[axis] += data["accel"][axis]
                gyro_sum[axis] += data["gyro"][axis]
            time.sleep(0.01)
            if (i + 1) % 50 == 0:
                print(f"  采集进度: {i+1}/{samples}")

        # 计算平均偏移
        self.gyro_offset = {
            axis: gyro_sum[axis] / samples for axis in ("x", "y", "z")
        }
        # 加速度计 Z 轴应为 1g（重力），X/Y 应为 0
        self.accel_offset = {
            "x": accel_sum["x"] / samples,
            "y": accel_sum["y"] / samples,
            "z": accel_sum["z"] / samples - 1.0,  # 减去重力
        }

        print(f"\n[校准完成]")
        print(f"  加速度偏移: X={self.accel_offset['x']:.4f}g, "
              f"Y={self.accel_offset['y']:.4f}g, "
              f"Z={self.accel_offset['z']:.4f}g")
        print(f"  陀螺仪偏移: X={self.gyro_offset['x']:.4f}°/s, "
              f"Y={self.gyro_offset['y']:.4f}°/s, "
              f"Z={self.gyro_offset['z']:.4f}°/s")

    def close(self):
        """关闭 I2C 连接"""
        if self.bus:
            self.bus.close()
            print("[I2C] 连接已关闭")


class AttitudeEstimator:
    """
    姿态估计器（互补滤波器）
    融合加速度计和陀螺仪数据，计算俯仰角和横滚角
    """

    def __init__(self, alpha=0.98):
        """
        :param alpha: 互补滤波系数（0~1，越大越信任陀螺仪）
        """
        self.alpha = alpha
        self.pitch = 0.0  # 俯仰角（绕 Y 轴）
        self.roll = 0.0   # 横滚角（绕 X 轴）
        self.last_time = time.time()

    def update(self, accel, gyro):
        """
        更新姿态角
        :param accel: 加速度数据 {"x": ..., "y": ..., "z": ...}（单位：g）
        :param gyro: 陀螺仪数据 {"x": ..., "y": ..., "z": ...}（单位：°/s）
        :return: (pitch, roll) 单位：度
        """
        now = time.time()
        dt = now - self.last_time
        self.last_time = now

        # 从加速度计计算角度
        accel_pitch = math.degrees(
            math.atan2(accel["y"], math.sqrt(accel["x"]**2 + accel["z"]**2))
        )
        accel_roll = math.degrees(
            math.atan2(-accel["x"], accel["z"])
        )

        # 互补滤波：融合陀螺仪积分和加速度计角度
        self.pitch = self.alpha * (self.pitch + gyro["x"] * dt) + \
                     (1 - self.alpha) * accel_pitch
        self.roll = self.alpha * (self.roll + gyro["y"] * dt) + \
                    (1 - self.alpha) * accel_roll

        return self.pitch, self.roll


def print_data(data, attitude=None):
    """格式化打印传感器数据"""
    accel = data["accel"]
    gyro = data["gyro"]
    temp = data["temperature"]

    print(f"\r  加速度(g): X={accel['x']:+7.3f}  Y={accel['y']:+7.3f}  Z={accel['z']:+7.3f}  "
          f"| 陀螺仪(°/s): X={gyro['x']:+8.2f}  Y={gyro['y']:+8.2f}  Z={gyro['z']:+8.2f}  "
          f"| 温度: {temp:5.1f}°C", end="")

    if attitude:
        pitch, roll = attitude
        print(f"  | 俯仰: {pitch:+7.2f}°  横滚: {roll:+7.2f}°", end="")


def run_normal(sensor, interval=0.1):
    """普通读取模式"""
    print("\n[读取] 开始读取传感器数据（Ctrl+C 退出）")
    print("  格式: 加速度(g) | 陀螺仪(°/s) | 温度(°C)\n")

    try:
        while True:
            data = sensor.read_all()
            print_data(data)
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\n\n[退出] 停止读取")


def run_attitude(sensor, interval=0.02):
    """姿态角计算模式"""
    print("\n[姿态] 开始计算姿态角（Ctrl+C 退出）")
    print("  互补滤波器 α=0.98\n")

    estimator = AttitudeEstimator(alpha=0.98)

    try:
        while True:
            data = sensor.read_all()
            pitch, roll = estimator.update(data["accel"], data["gyro"])
            print_data(data, attitude=(pitch, roll))
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\n\n[退出] 停止计算")


def run_log(sensor, output_file=None, interval=0.1, duration=None):
    """数据记录模式（保存到 CSV）"""
    if output_file is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"imu_log_{ts}.csv"

    print(f"\n[记录] 开始记录数据到: {output_file}")
    if duration:
        print(f"[记录] 记录时长: {duration} 秒")
    print("  按 Ctrl+C 停止\n")

    fieldnames = [
        "timestamp",
        "accel_x", "accel_y", "accel_z",
        "gyro_x", "gyro_y", "gyro_z",
        "temperature"
    ]

    record_count = 0
    start_time = time.time()

    try:
        with open(output_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            while True:
                elapsed = time.time() - start_time
                if duration and elapsed >= duration:
                    print(f"\n[完成] 已记录 {duration} 秒")
                    break

                data = sensor.read_all()
                row = {
                    "timestamp": f"{elapsed:.3f}",
                    "accel_x": f"{data['accel']['x']:.6f}",
                    "accel_y": f"{data['accel']['y']:.6f}",
                    "accel_z": f"{data['accel']['z']:.6f}",
                    "gyro_x": f"{data['gyro']['x']:.6f}",
                    "gyro_y": f"{data['gyro']['y']:.6f}",
                    "gyro_z": f"{data['gyro']['z']:.6f}",
                    "temperature": f"{data['temperature']:.2f}",
                }
                writer.writerow(row)
                record_count += 1

                if record_count % 10 == 0:
                    print(f"\r  已记录 {record_count} 条数据，"
                          f"耗时 {elapsed:.1f}s", end="")

                time.sleep(interval)

    except KeyboardInterrupt:
        print(f"\n\n[停止] 用户中断")
    finally:
        print(f"[完成] 共记录 {record_count} 条数据，保存到: {output_file}")


def parse_args():
    parser = argparse.ArgumentParser(description="MPU6050 I2C 传感器示例")
    parser.add_argument("--bus", type=int, default=1,
                        help="I2C 总线号（默认: 1）")
    parser.add_argument("--addr", type=lambda x: int(x, 0), default=0x68,
                        help="I2C 地址（默认: 0x68）")
    parser.add_argument("--mode", choices=["normal", "attitude", "log", "calibrate"],
                        default="normal", help="运行模式（默认: normal）")
    parser.add_argument("--interval", type=float, default=0.1,
                        help="采样间隔秒（默认: 0.1）")
    parser.add_argument("--log", type=str, default=None,
                        help="日志文件路径（log 模式）")
    parser.add_argument("--duration", type=float, default=None,
                        help="记录时长秒（log 模式，默认无限）")
    return parser.parse_args()


def main():
    print("=" * 60)
    print("  Jetson Nano I2C 传感器示例（MPU6050）")
    print("=" * 60)

    args = parse_args()

    print(f"  I2C 总线: {args.bus}")
    print(f"  I2C 地址: 0x{args.addr:02X}")
    print(f"  运行模式: {args.mode}")
    print("=" * 60)

    try:
        sensor = MPU6050(bus=args.bus, address=args.addr)
    except (ImportError, RuntimeError) as e:
        print(f"\n[错误] {e}")
        return

    try:
        if args.mode == "calibrate":
            sensor.calibrate()
        elif args.mode == "attitude":
            run_attitude(sensor, interval=args.interval)
        elif args.mode == "log":
            run_log(sensor, output_file=args.log,
                    interval=args.interval, duration=args.duration)
        else:
            run_normal(sensor, interval=args.interval)
    finally:
        sensor.close()


if __name__ == "__main__":
    main()