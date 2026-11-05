"""The sampler: one CSV row per tick with device-level and per-group GPU memory, utilisation, PCIe and RAM.

Times are Unix epoch seconds (time.time()) so they line up with every other log of the run.
Per-process GPU memory comes from NVML's compute + graphics process lists; a process in both lists is counted
once (NVML reports the same per-process total in both). 'unattributed' = device used - sum of listed processes
(driver/context overhead and processes NVML does not list).
"""

from __future__ import annotations

import csv
import threading
import time
from pathlib import Path

import psutil
import pynvml

from gpumon import procs

MIB = 2 ** 20
COLUMNS = ["time", "used_mib", "free_mib", "total_mib", "gpu_util", "mem_util", "pcie_tx_mbs", "pcie_rx_mbs",
           "power_w", "sm_clock_mhz", "temp_c", "gpu_sim_mib", "gpu_agent_mib", "gpu_other_mib", "gpu_unattributed_mib",
           "ram_used_mib", "ram_available_mib", "rss_sim_mib", "rss_agent_mib", "n_gpu_procs"]


class Sampler:
    """Background thread writing <out>/samples.csv at `hz` until stop(). `pcie`: also read PCIe throughput (each
    read takes ~20 ms inside the driver, so it bounds the rate at ~20 Hz)."""

    def __init__(self, out_dir: Path, hz: float = 10.0, gpu_index: int = 0, pcie: bool = True) -> None:
        self.path = Path(out_dir) / "samples.csv"
        self.period, self.pcie = 1.0 / hz, pcie
        pynvml.nvmlInit()
        self.h = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
        name = pynvml.nvmlDeviceGetName(self.h)
        self.device = name.decode() if isinstance(name, bytes) else name
        self.total_mib = pynvml.nvmlDeviceGetMemoryInfo(self.h).total / MIB
        self._stop = threading.Event()
        self._groups: dict[int, str] = {}
        self._ram_procs: dict[str, list[psutil.Process]] = {"sim": [], "agent": []}
        self._ram_refresh = 0.0
        self.thread = threading.Thread(target=self._run, name="gpumon-sampler", daemon=True)
        self.peak_used_mib = 0.0
        self.rows = 0

    def start(self) -> "Sampler":
        self.thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        self.thread.join(5)
        try:
            pynvml.nvmlShutdown()
        except pynvml.NVMLError:
            pass

    # ------------------------------------------------------------------------------------------------ one tick
    def _gpu_procs(self) -> dict[int, float]:
        mem: dict[int, float] = {}
        for fn in (pynvml.nvmlDeviceGetComputeRunningProcesses, pynvml.nvmlDeviceGetGraphicsRunningProcesses):
            try:
                for p in fn(self.h):
                    used = p.usedGpuMemory or 0  # None when the driver cannot attribute it
                    mem[p.pid] = max(mem.get(p.pid, 0), used / MIB)
            except pynvml.NVMLError:
                pass
        return mem

    def _rss(self, group: str, now: float) -> float:
        if now >= self._ram_refresh:  # re-scan the process table once a second, not every tick
            self._ram_procs = {g: procs.find(g) for g in ("sim", "agent")}
            self._ram_refresh = now + 1.0
        total = 0
        for p in self._ram_procs[group]:
            try:
                total += p.memory_info().rss
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        return total / MIB

    def sample(self) -> dict:
        now = time.time()
        info = pynvml.nvmlDeviceGetMemoryInfo(self.h)
        util = pynvml.nvmlDeviceGetUtilizationRates(self.h)
        row = {"time": round(now, 3), "used_mib": info.used / MIB, "free_mib": info.free / MIB,
               "total_mib": info.total / MIB, "gpu_util": util.gpu, "mem_util": util.memory}
        if self.pcie:
            try:  # KB/s over the driver's last ~20 ms window
                row["pcie_tx_mbs"] = pynvml.nvmlDeviceGetPcieThroughput(self.h, pynvml.NVML_PCIE_UTIL_TX_BYTES) / 1024
                row["pcie_rx_mbs"] = pynvml.nvmlDeviceGetPcieThroughput(self.h, pynvml.NVML_PCIE_UTIL_RX_BYTES) / 1024
            except pynvml.NVMLError:
                pass
        for key, fn in (("power_w", lambda: pynvml.nvmlDeviceGetPowerUsage(self.h) / 1000),
                        ("sm_clock_mhz", lambda: pynvml.nvmlDeviceGetClockInfo(self.h, pynvml.NVML_CLOCK_SM)),
                        ("temp_c", lambda: pynvml.nvmlDeviceGetTemperature(self.h, pynvml.NVML_TEMPERATURE_GPU))):
            try:
                row[key] = fn()
            except pynvml.NVMLError:
                pass
        by_group = {"sim": 0.0, "agent": 0.0, "other": 0.0}
        gpu = self._gpu_procs()
        for pid, mib in gpu.items():
            by_group[procs.classify(pid, self._groups)] += mib
        for g, v in by_group.items():
            row[f"gpu_{g}_mib"] = v
        row["gpu_unattributed_mib"] = max(0.0, row["used_mib"] - sum(by_group.values()))
        row["n_gpu_procs"] = len(gpu)
        vm = psutil.virtual_memory()
        row["ram_used_mib"], row["ram_available_mib"] = (vm.total - vm.available) / MIB, vm.available / MIB
        row["rss_sim_mib"], row["rss_agent_mib"] = self._rss("sim", now), self._rss("agent", now)
        return {k: (round(v, 1) if isinstance(v, float) and k != "time" else v) for k, v in row.items()}

    def _run(self) -> None:
        with open(self.path, "w", newline="", buffering=1) as f:
            w = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
            w.writeheader()
            nxt = time.monotonic()
            while not self._stop.is_set():
                try:
                    row = self.sample()
                    w.writerow(row)
                    self.rows += 1
                    self.peak_used_mib = max(self.peak_used_mib, row["used_mib"])
                except pynvml.NVMLError as e:  # keep sampling; a driver hiccup must not end the recording
                    f.write(f"# nvml error at {time.time():.3f}: {e}\n")
                nxt += self.period
                delay = nxt - time.monotonic()
                if delay < 0:  # fell behind (slow tick): restart the schedule instead of bursting
                    nxt, delay = time.monotonic(), 0
                self._stop.wait(delay)
