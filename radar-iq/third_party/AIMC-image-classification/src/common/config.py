from __future__ import annotations

import copy
import math
import os
from dataclasses import dataclass, asdict, field
from contextlib import nullcontext

from datetime import datetime
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd
import torch
from torch import amp
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
import torch.optim as optim
from tqdm import tqdm
import torch.nn.functional as F


def _amp_cfg_for_device(device: str):
    if device == "cuda":
        use_bf16 = torch.cuda.is_bf16_supported()
        amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
        scaler = amp.GradScaler(enabled=(amp_dtype is torch.float16))
        ctx = lambda: amp.autocast(device_type="cuda", dtype=amp_dtype)
    elif device == "cpu":
        scaler = amp.GradScaler(enabled=False)
        ctx = nullcontext
    else:
        scaler = amp.GradScaler(enabled=False)
        ctx = lambda: amp.autocast(device_type=device)
    return ctx, scaler


@dataclass(frozen=False)
class DatasetConfig:
    modulation_list: list[str]
    noise_label: str = "noise"
    use_noise: bool = True
    snr_range: list[int] = None
    window_count: int = 500
    window_size: int = 1000
    window_step_size: int = 1000
    k_ratio: dict[int, float] = None
    history_length: int = 1
    stride: int = 1
    threshold_time_frac: float = 0.02
    train_noise_pos_ratio: float = 0.5
    val_noise_pos_ratio: float = 0.05
    data_path: str = "data\\"

    def __post_init__(self):
        if self.snr_range is None:
            object.__setattr__(self, "snr_range", [+6, +3, 0, -3, -6])
        if self.k_ratio is None:
            object.__setattr__(self, "k_ratio", {1: 1.0})

    @property
    def train_data_path(self) -> str:
        return os.path.join(self.data_path, "train\\")

    @property
    def test_data_path(self) -> str:
        return os.path.join(self.data_path, "test\\")

    @property
    def snr_str(self) -> str:
        return "".join(str(s) for s in self.snr_range)


@dataclass(frozen=False)
class TrainConfig:
    seed: int = 42
    max_epochs: int = 40
    batch_size_train: int = 32
    batch_size_val: int = 64
    learning_rate: float = 1e-3
    patience: int = 10
    min_delta: float = 0.003
    update_every: int = 10
    ema_alpha: float = 0.1
    backbone: str = "convnext_tiny"
    pretrained_pt: str | None = "pretrained_convnext_tiny.pt"
    out_dir: str = "out\\convnext_power\\"
    ckpt_dir: str = "checkpoints\\convnext_power\\"
    cm_every_epochs: int = 5
    criterion: nn.Module = None
    optimizer: optim.Optimizer = None
    scheduler: optim.lr_scheduler.LRScheduler = None
    train_split: float = 0.8


def write_run_config_txt(
    run_dir: str,
    ds_cfg: DatasetConfig,
    train_cfg: TrainConfig,
    extra: dict[str, Any] | None = None,
):
    """
    Writes a human-readable config record for the run.
    """
    cfg_path = os.path.join(run_dir, "configs.txt")

    def _fmt_block(title: str, d: dict) -> str:
        lines = [f"[{title}]"]
        for k in sorted(d.keys()):
            lines.append(f"{k} = {d[k]}")
        return "\n".join(lines)

    ds_dict = asdict(ds_cfg)
    tr_dict = asdict(train_cfg)

    blocks = [
        _fmt_block("DatasetConfig", ds_dict),
        "",
        _fmt_block("TrainConfig", tr_dict),
    ]

    if extra:
        blocks += ["", _fmt_block("RunMeta", dict(extra))]

    text = "\n".join(blocks) + "\n"
    with open(cfg_path, "w", encoding="utf-8") as f:
        f.write(text)

    return cfg_path


def build_model_and_amp(
    model: nn.Module, device: torch.device
):

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    model = model.to(device).to(memory_format=torch.channels_last)

    device_str = "cuda" if device.type == "cuda" else "cpu"
    ctx, scaler = _amp_cfg_for_device(device_str)
    return model, ctx, scaler
