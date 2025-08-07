import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np

"""
Assumes that several perfomance runs have been made
Author: andreas.wassmer@fhnw.ch
"""


@dataclass
class LogfileInfo:
    filename: str
    freq_channels: int
    duration: float

    def is_gpu_run(self) -> bool:
        return "gpu" in str(self.filename)


def efficency(logfile: LogfileInfo) -> float:
    """Calculate channels per minute
    Not used yet, here for future reference
    """
    if logfile.duration == 0:
        return 0
    return logfile.freq_channels / logfile.duration


def extract_log_data(logfile_path) -> Tuple[int, float]:
    """
    Extracts the number of frequency channels and the calculation time
    from a log file. The function expects a log file from 'karabo_benchmark.py'
    """
    frequency_channels = 0
    duration_minutes = 0

    with open(logfile_path, "r") as file:
        for line in file:
            if "Number of freqeuncy channels" in line:
                # Extrahiere die letzte Zahl in der Zeile
                match = re.search(r"(\d+)\s*$", line)
                if match:
                    frequency_channels = int(match.group(1))
            elif "DURATION" in line:
                # Extrahiere den float-Wert vor "mins"
                match = re.search(r"DURATION:\s*([0-9.]+)\s*mins", line)
                if match:
                    duration_minutes = float(match.group(1))

    return frequency_channels, duration_minutes


log_infos_cpu = []
log_infos_gpu = []
log_dir = Path(".")

log_files: List[Path] = list(log_dir.glob("logfiles/*.log"))
for logfile in log_files:
    print(f"Processing file {logfile}")
    freq_channels, duration = extract_log_data(logfile)
    if "gpu" in str(logfile):
        log_infos_gpu.append(LogfileInfo(str(logfile), freq_channels, duration))
    else:
        log_infos_cpu.append(LogfileInfo(str(logfile), freq_channels, duration))

log_infos_cpu.sort(key=lambda x: x.freq_channels)
log_infos_gpu.sort(key=lambda x: x.freq_channels)

x_cpu = [info.duration for info in log_infos_cpu]
x_gpu = [info.duration for info in log_infos_gpu]
y = [info.freq_channels for info in log_infos_cpu]

# let's fit a straight line to match the data points
z_cpu = np.polyfit(x_cpu, y, 1)
z_gpu = np.polyfit(x_gpu, y, 1)
y_cpu_fit = [i * z_cpu[0] + z_cpu[1] for i in x_cpu]
y_gpu_fit = [i * z_gpu[0] + z_gpu[1] for i in x_gpu]

# calculate the speed up
speedup = []
for i in range(len(x_cpu)):
    speedup.append(x_cpu[i] / x_gpu[i])

plt.rcParams["savefig.dpi"] = 300
f, ax = plt.subplots(1, 1, figsize=(8, 4))
ax.plot(x_cpu, y, "o", color="g", label="CPU only")
ax.plot(x_cpu, y_cpu_fit, "--", color="g")
ax.plot(x_gpu, y, "o", color="b", label="with GPU")
ax.plot(x_gpu, y_gpu_fit, "--", color="b")
ax.set_ylabel("Number of Frequency Channels")
ax.set_xlabel("Computation Time (min)")
ax.set_title("Benchmark CPU/GPU")
ax.set_yscale("log")
# ax.grid()
ax.legend()
plt.tight_layout()
plt.savefig("CPU_vs_GPU.png")
plt.show()


f, ax = plt.subplots(1, 1, figsize=(8, 6))
ax.plot(y, speedup, "o-", color="b", label="Speedup (CPU only = 1)")
ax.set_ylabel("Speedup")
ax.set_xlabel("Number of Frequency Channels")
ax.set_title("Benchmark CPU/GPU")
ax.set_ylim(1, 5)
ax.grid()
ax.legend()
plt.tight_layout()
plt.savefig("benchmark_speedup.png")
plt.show()
