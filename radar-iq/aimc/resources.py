"""Bounded compute bursts, thermal backoff and file-based run controls."""

from contextlib import contextmanager
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import resource
import subprocess
import time

import torch


class StopRequested(Exception):
    pass


@dataclass
class ResourceConfig:
    threads: int = 2
    duty_cycle: float = 0.25
    min_idle_seconds: float = 0.005
    gpu_memory_fraction: float = 0.20
    max_temperature_c: int = 65
    resume_temperature_c: int = 58
    min_available_memory_mb: int = 2048
    nice: int = 10

    def __post_init__(self):
        if self.threads < 1 or not 0 < self.duty_cycle <= 0.5:
            raise ValueError("Use positive threads and a duty cycle in (0,0.5].")
        if not 0 < self.gpu_memory_fraction <= 0.4 or self.min_idle_seconds < 0:
            raise ValueError("Invalid memory fraction or idle interval.")
        if not 0 < self.resume_temperature_c < self.max_temperature_c <= 80:
            raise ValueError("Invalid thermal thresholds.")

    def to_dict(self):
        return asdict(self)


def available_memory_mb():
    fields = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
    return int(fields["MemAvailable"].split()[0]) / 1024


class ResourceGovernor:
    def __init__(self, config, out_dir, device="cpu"):
        self.config = config
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.device = torch.device(device)
        torch.set_num_threads(config.threads)
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        current = os.getpriority(os.PRIO_PROCESS, 0)
        if current < config.nice:
            os.nice(config.nice - current)
        if self.device.type == "cuda":
            torch.cuda.set_per_process_memory_fraction(config.gpu_memory_fraction, self.device)
        self.last_monitor = 0.0
        self.active_seconds = 0.0
        self.idle_seconds = 0.0
        self.started = time.monotonic()
        self.cooling = False

    def gpu_status(self):
        if self.device.type != "cuda":
            return {}
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=temperature.gpu,utilization.gpu,memory.used,power.draw",
             "--format=csv,noheader,nounits", "-i", str(self.device.index or 0)],
            capture_output=True, text=True, timeout=8, check=True,
        )
        values = result.stdout.strip().split(",")
        return {"gpu_temperature_c": float(values[0]), "gpu_utilization_percent": float(values[1]),
                "gpu_total_memory_used_mb": float(values[2]),
                "gpu_power_w": None if "N/A" in values[3] else float(values[3])}

    def controls(self):
        if (self.out_dir / "STOP").exists():
            raise StopRequested("STOP file found; saving a resumable checkpoint.")
        while (self.out_dir / "PAUSE").exists():
            if (self.out_dir / "STOP").exists():
                raise StopRequested("STOP file found during pause.")
            time.sleep(0.2)

    def monitor(self, force=False):
        self.controls()
        now = time.monotonic()
        if not force and now - self.last_monitor < 5:
            return
        self.last_monitor = now
        gpu = self.gpu_status()
        while (gpu.get("gpu_temperature_c", 0) >= self.config.max_temperature_c
               or (self.cooling and gpu.get("gpu_temperature_c", 0) > self.config.resume_temperature_c)
               or available_memory_mb() < self.config.min_available_memory_mb):
            self.cooling = gpu.get("gpu_temperature_c", 0) >= self.config.resume_temperature_c
            self.controls()
            self.write_status(gpu, "cooling_or_waiting_for_memory")
            time.sleep(2)
            gpu = self.gpu_status()
        self.cooling = False
        self.write_status(gpu, "running")

    def write_status(self, gpu, state):
        elapsed = max(time.monotonic() - self.started, 1e-9)
        status = {"state": state, "pid": os.getpid(), "elapsed_seconds": elapsed,
                  "active_seconds": self.active_seconds, "idle_seconds": self.idle_seconds,
                  "active_fraction": self.active_seconds / elapsed,
                  "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
                  "available_memory_mb": available_memory_mb(), "limits": self.config.to_dict(), **gpu}
        temporary = self.out_dir / "resources.json.tmp"
        temporary.write_text(json.dumps(status, indent=2) + "\n")
        temporary.replace(self.out_dir / "resources.json")
        with (self.out_dir / "resources.jsonl").open("a") as stream:
            stream.write(json.dumps(status) + "\n")

    @contextmanager
    def burst(self):
        self.monitor()
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        started = time.monotonic()
        try:
            yield
        finally:
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            duration = time.monotonic() - started
            self.active_seconds += duration
            idle = max(self.config.min_idle_seconds, duration * (1 / self.config.duty_cycle - 1))
            time.sleep(idle)
            self.idle_seconds += idle
