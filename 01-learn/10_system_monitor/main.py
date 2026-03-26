#!/usr/bin/env python3
"""
示例 10：Jetson Nano 系统资源监控
实时监控 CPU、GPU、内存、温度、功耗等系统指标
支持终端显示和 Web 界面
"""

import os
import sys
import time
import argparse
import csv
import subprocess
from datetime import datetime

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    print("[警告] psutil 未安装，部分功能不可用: pip3 install psutil")


# ── ANSI 颜色代码 ──────────────────────────────────────────────────────────────
class Color:
    RESET  = "\033[0m"
    BOLD   = "\033[1m"
    RED    = "\033[91m"
    GREEN  = "\033[92m"
    YELLOW = "\033[93m"
    BLUE   = "\033[94m"
    CYAN   = "\033[96m"
    WHITE  = "\033[97m"
    GRAY   = "\033[90m"

    @staticmethod
    def colored(text, color):
        return f"{color}{text}{Color.RESET}"

    @staticmethod
    def bar(value, max_val=100, width=20, warn=70, crit=90):
        """生成进度条"""
        pct = min(value / max_val, 1.0)
        filled = int(pct * width)
        bar_str = "█" * filled + "░" * (width - filled)
        if value >= crit:
            color = Color.RED
        elif value >= warn:
            color = Color.YELLOW
        else:
            color = Color.GREEN
        return f"{color}[{bar_str}]{Color.RESET} {value:5.1f}%"


# ── Jetson 特有信息读取 ────────────────────────────────────────────────────────
class JetsonStats:
    """读取 Jetson Nano 特有的系统信息"""

    THERMAL_ZONES = {
        "CPU":  "/sys/devices/virtual/thermal/thermal_zone0/temp",
        "GPU":  "/sys/devices/virtual/thermal/thermal_zone1/temp",
        "PLL":  "/sys/devices/virtual/thermal/thermal_zone2/temp",
        "PMIC": "/sys/devices/virtual/thermal/thermal_zone3/temp",
        "Fan":  "/sys/devices/virtual/thermal/thermal_zone4/temp",
        "AO":   "/sys/devices/virtual/thermal/thermal_zone5/temp",
    }

    GPU_FREQ_PATH = "/sys/devices/57000000.gpu/devfreq/57000000.gpu"

    POWER_PATHS = {
        "VDD_IN":  "/sys/bus/i2c/drivers/ina3221x/6-0040/iio:device0/in_power0_input",
        "VDD_CPU": "/sys/bus/i2c/drivers/ina3221x/6-0040/iio:device0/in_power1_input",
        "VDD_GPU": "/sys/bus/i2c/drivers/ina3221x/6-0040/iio:device0/in_power2_input",
    }

    @staticmethod
    def read_file(path, default=None):
        try:
            with open(path) as f:
                return f.read().strip()
        except (FileNotFoundError, PermissionError):
            return default

    @classmethod
    def get_temperatures(cls):
        temps = {}
        for name, path in cls.THERMAL_ZONES.items():
            val = cls.read_file(path)
            if val is not None:
                try:
                    temps[name] = int(val) / 1000.0
                except ValueError:
                    pass
        return temps

    @classmethod
    def get_gpu_freq(cls):
        cur   = cls.read_file(f"{cls.GPU_FREQ_PATH}/cur_freq")
        max_f = cls.read_file(f"{cls.GPU_FREQ_PATH}/max_freq")
        min_f = cls.read_file(f"{cls.GPU_FREQ_PATH}/min_freq")
        return {
            "current": int(cur)   // 1_000_000 if cur   else 0,
            "max":     int(max_f) // 1_000_000 if max_f else 0,
            "min":     int(min_f) // 1_000_000 if min_f else 0,
        }

    @classmethod
    def get_gpu_usage(cls):
        paths = [
            "/sys/devices/57000000.gpu/load",
            "/sys/class/devfreq/57000000.gpu/device/load",
        ]
        for path in paths:
            val = cls.read_file(path)
            if val is not None:
                try:
                    return float(val) / 10.0
                except ValueError:
                    pass
        return 0.0

    @classmethod
    def get_power(cls):
        power = {}
        for name, path in cls.POWER_PATHS.items():
            val = cls.read_file(path)
            if val is not None:
                try:
                    power[name] = int(val)
                except ValueError:
                    pass
        return power

    @classmethod
    def get_fan_speed(cls):
        paths = [
            "/sys/devices/pwm-fan/hwmon/hwmon3/fan1_input",
            "/sys/class/hwmon/hwmon3/fan1_input",
        ]
        for path in paths:
            val = cls.read_file(path)
            if val is not None:
                try:
                    return int(val)
                except ValueError:
                    pass
        return None

    @classmethod
    def get_nvpmodel(cls):
        try:
            result = subprocess.run(
                ["nvpmodel", "-q"],
                capture_output=True, text=True, timeout=2
            )
            for line in result.stdout.split("\n"):
                if "NV Power Mode" in line:
                    return line.split(":")[-1].strip()
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
        return "Unknown"


# ── 系统监控主类 ───────────────────────────────────────────────────────────────
class SystemMonitor:

    def __init__(self, interval=1.0, log_file=None):
        self.interval = interval
        self.log_file = log_file
        self.running = False
        self.prev_net = None
        self.prev_net_time = None
        self.csv_writer = None
        self.csv_file_handle = None

        if log_file:
            self._init_log()

    def _init_log(self):
        self.csv_file_handle = open(self.log_file, "w", newline="")
        fieldnames = [
            "timestamp", "cpu_pct", "cpu_freq_mhz",
            "gpu_pct", "gpu_freq_mhz",
            "mem_used_mb", "mem_pct",
            "temp_cpu", "temp_gpu",
            "power_total_mw",
        ]
        self.csv_writer = csv.DictWriter(self.csv_file_handle, fieldnames=fieldnames)
        self.csv_writer.writeheader()
        print(f"[日志] 数据将记录到: {self.log_file}")

    def collect(self):
        data = {"timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

        if HAS_PSUTIL:
            data["cpu_pct"] = psutil.cpu_percent(interval=None, percpu=True)
            data["cpu_pct_total"] = sum(data["cpu_pct"]) / len(data["cpu_pct"])
            cpu_freq = psutil.cpu_freq()
            data["cpu_freq"] = {
                "current": cpu_freq.current if cpu_freq else 0,
                "max":     cpu_freq.max     if cpu_freq else 0,
            }

            mem = psutil.virtual_memory()
            data["memory"] = {
                "total_mb":     mem.total     // 1024 // 1024,
                "used_mb":      mem.used      // 1024 // 1024,
                "available_mb": mem.available // 1024 // 1024,
                "percent":      mem.percent,
            }

            swap = psutil.swap_memory()
            data["swap"] = {
                "total_mb": swap.total // 1024 // 1024,
                "used_mb":  swap.used  // 1024 // 1024,
                "percent":  swap.percent,
            }

            disk = psutil.disk_usage("/")
            data["disk"] = {
                "total_gb": disk.total // 1024 // 1024 // 1024,
                "used_gb":  disk.used  // 1024 // 1024 // 1024,
                "percent":  disk.percent,
            }

            net = psutil.net_io_counters()
            now = time.time()
            if self.prev_net and self.prev_net_time:
                dt = now - self.prev_net_time
                data["network"] = {
                    "rx_mbps": (net.bytes_recv - self.prev_net.bytes_recv) / dt / 1024 / 1024,
                    "tx_mbps": (net.bytes_sent - self.prev_net.bytes_sent) / dt / 1024 / 1024,
                }
            else:
                data["network"] = {"rx_mbps": 0.0, "tx_mbps": 0.0}
            self.prev_net = net
            self.prev_net_time = now

        data["gpu_pct"]      = JetsonStats.get_gpu_usage()
        data["gpu_freq"]     = JetsonStats.get_gpu_freq()
        data["temperatures"] = JetsonStats.get_temperatures()
        data["power"]        = JetsonStats.get_power()
        data["fan_rpm"]      = JetsonStats.get_fan_speed()
        return data

    def display_terminal(self, data):
        os.system("clear" if os.name == "posix" else "cls")

        ts = data["timestamp"]
        print(f"{Color.BOLD}{Color.CYAN}{'='*65}{Color.RESET}")
        print(f"{Color.BOLD}{Color.CYAN}  Jetson Nano 系统监控  "
              f"{Color.GRAY}{ts}{Color.RESET}")
        print(f"{Color.BOLD}{Color.CYAN}{'='*65}{Color.RESET}")

        # CPU
        if "cpu_pct" in data:
            print(f"\n{Color.BOLD}【CPU】{Color.RESET}")
            for i, pct in enumerate(data["cpu_pct"]):
                print(f"  核心 {i}: {Color.bar(pct)}")
            freq = data["cpu_freq"]
            print(f"  频率: {freq['current']:.0f} MHz / {freq['max']:.0f} MHz")

        # GPU
        print(f"\n{Color.BOLD}【GPU】{Color.RESET}")
        print(f"  使用率: {Color.bar(data['gpu_pct'])}")
        gf = data["gpu_freq"]
        if gf["current"] > 0:
            print(f"  频率: {gf['current']} MHz / {gf['max']} MHz")

        # 内存
        if "memory" in data:
            mem = data["memory"]
            print(f"\n{Color.BOLD}【内存】{Color.RESET}")
            print(f"  RAM:  {Color.bar(mem['percent'])}  "
                  f"{mem['used_mb']:,} / {mem['total_mb']:,} MB")
            swap = data["swap"]
            if swap["total_mb"] > 0:
                print(f"  Swap: {Color.bar(swap['percent'])}  "
                      f"{swap['used_mb']:,} / {swap['total_mb']:,} MB")

        # 温度
        temps = data["temperatures"]
        if temps:
            print(f"\n{Color.BOLD}【温度】{Color.RESET}")
            for name, temp in temps.items():
                warn = 75 if name in ("CPU", "GPU") else 80
                crit = 85
                if temp >= crit:
                    color = Color.RED
                    flag = " ⚠ 过热!"
                elif temp >= warn:
                    color = Color.YELLOW
                    flag = " △ 偏高"
                else:
                    color = Color.GREEN
                    flag = ""
                print(f"  {name:6s}: {color}{temp:5.1f}°C{Color.RESET}{flag}")

        # 功耗
        power = data["power"]
        if power:
            print(f"\n{Color.BOLD}【功耗】{Color.RESET}")
            for name, mw in power.items():
                print(f"  {name:10s}: {mw/1000:.2f} W")

        # 磁盘
        if "disk" in data:
            disk = data["disk"]
            print(f"\n{Color.BOLD}【磁盘】{Color.RESET}")
            print(f"  /: {Color.bar(disk['percent'])}  "
                  f"{disk['used_gb']} / {disk['total_gb']} GB")

        # 网络
        if "network" in data:
            net = data["network"]
            print(f"\n{Color.BOLD}【网络】{Color.RESET}")
            print(f"  ↓ {net['rx_mbps']:.2f} MB/s  ↑ {net['tx_mbps']:.2f} MB/s")

        # 风扇
        if data.get("fan_rpm") is not None:
            print(f"\n{Color.BOLD}【风扇】{Color.RESET}")
            print(f"  转速: {data['fan_rpm']} RPM")

        print(f"\n{Color.GRAY}  刷新间隔: {self.interval}s  按 Ctrl+C 退出{Color.RESET}")

    def log_data(self, data):
        if not self.csv_writer:
            return
        row = {
            "timestamp":      data["timestamp"],
            "cpu_pct":        f"{data.get('cpu_pct_total', 0):.1f}",
            "cpu_freq_mhz":   f"{data.get('cpu_freq', {}).get('current', 0):.0f}",
            "gpu_pct":        f"{data['gpu_pct']:.1f}",
            "gpu_freq_mhz":   f"{data['gpu_freq']['current']}",
            "mem_used_mb":    data.get("memory", {}).get("used_mb", 0),
            "mem_pct":        f"{data.get('memory', {}).get('percent', 0):.1f}",
            "temp_cpu":       f"{data['temperatures'].get('CPU', 0):.1f}",
            "temp_gpu":       f"{data['temperatures'].get('GPU', 0):.1f}",
            "power_total_mw": data["power"].get("VDD_IN", 0),
        }
        self.csv_writer.writerow(row)
        self.csv_file_handle.flush()

    def run(self):
        self.running = True
        print("[监控] 启动系统监控...")

        if HAS_PSUTIL:
            psutil.cpu_percent(interval=None, percpu=True)
            time.sleep(0.5)

        try:
            while self.running:
                data = self.collect()
                self.display_terminal(data)
                if self.csv_writer:
                    self.log_data(data)
                time.sleep(self.interval)
        except KeyboardInterrupt:
            print("\n\n[停止] 监控已停止")
        finally:
            if self.csv_file_handle:
                self.csv_file_handle.close()
                print(f"[日志] 数据已保存到: {self.log_file}")

    def run_web(self, host="0.0.0.0", port=5000):
        """启动 Web 监控界面（浏览器访问 http://<jetson-ip>:5000）"""
        try:
            from flask import Flask, jsonify, render_template_string
        except ImportError:
            print("[错误] Flask 未安装: pip3 install flask")
            return

        app = Flask(__name__)
        monitor = self

        HTML_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>Jetson Nano Monitor</title>
  <style>
    body{font-family:monospace;background:#1a1a2e;color:#eee;padding:20px}
    h1{color:#00d4ff}
    .card{background:#16213e;border-radius:8px;padding:15px;margin:10px 0}
    .bar-bg{background:#333;border-radius:4px;height:18px;margin:4px 0}
    .bar-fill{height:18px;border-radius:4px}
    .g{background:#00b894}.y{background:#fdcb6e}.r{background:#e17055}
    table{width:100%;border-collapse:collapse}
    td{padding:4px 8px}
    .lbl{color:#74b9ff;width:110px}
    .warn{color:#fdcb6e}.crit{color:#e17055}
  </style>
  <script>setTimeout(()=>location.reload(), 2000)</script>
</head>
<body>
  <h1>&#x1F680; Jetson Nano 系统监控</h1>
  <p style="color:#636e72">{{ stats.timestamp }}</p>

  <div class="card"><h3>CPU</h3>
  {% for i, p in enumerate(stats.cpu_pct) %}
    <div>核心{{i}}: {{p|round(1)}}%
      <div class="bar-bg"><div class="bar-fill {{
        'r' if p>90 else 'y' if p>70 else 'g' }}" style="width:{{p}}%"></div></div>
    </div>
  {% endfor %}
  <small>频率: {{stats.cpu_freq.current|round|int}} MHz</small>
  </div>

  <div class="card"><h3>GPU</h3>
  {{stats.gpu_pct|round(1)}}%
  <div class="bar-bg"><div class="bar-fill {{
    'r' if stats.gpu_pct>90 else 'y' if stats.gpu_pct>70 else 'g' }}"
    style="width:{{stats.gpu_pct}}%"></div></div>
  <small>频率: {{stats.gpu_freq.current}} MHz</small>
  </div>

  {% if stats.memory %}
  <div class="card"><h3>内存</h3>
  {{stats.memory.percent|round(1)}}%  ({{stats.memory.used_mb}} / {{stats.memory.total_mb}} MB)
  <div class="bar-bg"><div class="bar-fill {{
    'r' if stats.memory.percent>90 else 'y' if stats.memory.percent>70 else 'g' }}"
    style="width:{{stats.memory.percent}}%"></div></div>
  </div>
  {% endif %}

  <div class="card"><h3>温度</h3>
  <table>
  {% for name, temp in stats.temperatures.items() %}
  <tr>
    <td class="lbl">{{name}}</td>
    <td class="{{ 'crit' if temp>85 else 'warn' if temp>75 else '' }}">
      {{temp|round(1)}} °C
    </td>
  </tr>
  {% endfor %}
  </table></div>

  {% if stats.power %}
  <div class="card"><h3>功耗</h3>
  <table>
  {% for name, mw in stats.power.items() %}
  <tr><td class="lbl">{{name}}</td><td>{{(mw/1000)|round(2)}} W</td></tr>
  {% endfor %}
  </table></div>
  {% endif %}
</body>
</html>"""

        @app.route("/")
        def index():
            data = monitor.collect()
            # 将 dict 包装为对象以便模板访问
            class Obj:
                def __init__(self, d):
                    for k, v in d.items():
                        if isinstance(v, dict):
                            setattr(self, k, Obj(v))
                        else:
                            setattr(self, k, v)
            return render_template_string(
                HTML_TEMPLATE,
                stats=Obj(data),
                enumerate=enumerate
            )

        @app.route("/api/stats")
        def api_stats():
            data = monitor.collect()
            return jsonify(data)

        print(f"\n[Web] 监控界面已启动")
        print(f"  本地访问: http://localhost:{port}")
        print(f"  远程访问: http://<Jetson-IP>:{port}")
        print(f"  API 接口: http://localhost:{port}/api/stats")
        print(f"  按 Ctrl+C 停止\n")
        app.run(host=host, port=port, debug=False)


# ── 辅助工具函数 ───────────────────────────────────────────────────────────────
def show_tegrastats(duration=10):
    """
    调用 tegrastats 工具（Jetson 官方监控工具）
    输出原始 tegrastats 数据
    """
    print(f"\n[tegrastats] 运行 {duration} 秒（Ctrl+C 停止）\n")
    try:
        proc = subprocess.Popen(
            ["tegrastats", "--interval", "1000"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )
        start = time.time()
        for line in proc.stdout:
            print(line.strip())
            if time.time() - start >= duration:
                break
        proc.terminate()
    except FileNotFoundError:
        print("[错误] tegrastats 未找到（仅在 Jetson 设备上可用）")


def parse_args():
    parser = argparse.ArgumentParser(description="Jetson Nano 系统资源监控")
    parser.add_argument("--interval", type=float, default=1.0,
                        help="刷新间隔秒（默认: 1.0）")
    parser.add_argument("--web", action="store_true",
                        help="启动 Web 监控界面")
    parser.add_argument("--port", type=int, default=5000,
                        help="Web 服务端口（默认: 5000）")
    parser.add_argument("--log", type=str, default=None,
                        help="记录数据到 CSV 文件")
    parser.add_argument("--tegrastats", action="store_true",
                        help="使用 tegrastats 工具监控")
    parser.add_argument("--duration", type=int, default=10,
                        help="tegrastats 运行时长（秒）")
    return parser.parse_args()


def main():
    print("=" * 65)
    print("  Jetson Nano 系统资源监控")
    print("=" * 65)

    if not HAS_PSUTIL:
        print("[提示] 安装 psutil 获取完整功能: pip3 install psutil")

    args = parse_args()

    if args.tegrastats:
        show_tegrastats(args.duration)
        return

    monitor = SystemMonitor(interval=args.interval, log_file=args.log)

    if args.web:
        monitor.run_web(port=args.port)
    else:
        monitor.run()


if __name__ == "__main__":
    main()