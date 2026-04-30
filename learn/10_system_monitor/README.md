# 示例 10：Jetson 系统资源监控

## 功能说明

实时监控 Jetson Nano 的系统资源，包括 CPU、GPU、内存、温度、功耗等，并提供终端和 Web 两种显示方式。

## 监控指标

| 指标 | 说明 |
|------|------|
| CPU 使用率 | 4 核心各自的使用率 |
| CPU 频率 | 当前/最大频率 |
| GPU 使用率 | Maxwell GPU 负载 |
| GPU 频率 | 当前 GPU 时钟 |
| 内存使用 | RAM 使用量和占比 |
| 交换分区 | Swap 使用情况 |
| 温度 | CPU/GPU/板载温度 |
| 功耗 | 总功耗（需要 INA3221） |
| 磁盘 | 存储使用情况 |
| 网络 | 网络收发速率 |

## 依赖安装

```bash
pip3 install psutil
# Web 界面（可选）
pip3 install flask
```

## 运行方式

```bash
# 终端实时监控（默认）
python3 main.py

# 指定刷新间隔（秒）
python3 main.py --interval 2

# 启动 Web 监控界面（浏览器访问 http://jetson-ip:5000）
python3 main.py --web

# 记录数据到 CSV
python3 main.py --log monitor.csv

# 显示详细信息
python3 main.py --verbose
```

## 预期效果

- 终端实时显示系统资源使用情况
- 温度超过阈值时显示警告
- Web 模式下可远程监控

## 知识点

- Jetson 特有的系统信息读取（/sys/devices/...）
- psutil 库使用
- tegrastats 工具解析
- 终端 UI 绘制（ANSI 颜色）
- Flask Web 服务器