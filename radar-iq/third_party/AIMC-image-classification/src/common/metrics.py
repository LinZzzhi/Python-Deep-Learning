from __future__ import annotations

import copy
import math
import os
from dataclasses import dataclass, asdict, field

from datetime import datetime
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
import torch.optim as optim
from tqdm import tqdm
import torch.nn.functional as F

def make_run_dir(base_out_dir: str, suffix_info: str) -> tuple[str, str]:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_suffix = suffix_info.replace(os.sep, "_").replace(":", "_")
    run_dir = os.path.join(base_out_dir, f"{ts}{safe_suffix}")
    Path(run_dir).mkdir(parents=True, exist_ok=True)
    return run_dir, ts


class MetricsWriter:
    def __init__(self, run_dir: str, suffix_info: str, snr_keys: list[int]):
        Path(run_dir).mkdir(parents=True, exist_ok=True)
        self.csv_path = os.path.join(run_dir, f"metrics_{suffix_info}.csv")

        base_cols = ["epoch", "time", "train_loss", "val_loss_macro", "accuracy_macro"]
        snr_cols = []
        for s in snr_keys:
            snr_cols += [f"loss_{s:+d}dB", f"accuracy_{s:+d}dB"]
        self.columns = base_cols + snr_cols
        self.snr_keys = snr_keys

    def append(
        self,
        ep: int,
        train_loss: float,
        stats_by_snr: dict,
        val_loss_macro: float,
        acc_macro: float,
    ):
        row = {
            "epoch": ep,
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "train_loss": float(train_loss),
            "val_loss_macro": float(val_loss_macro),
            "accuracy_macro": float(acc_macro),
        }
        for s in self.snr_keys:
            row[f"loss_{s:+d}dB"] = float(stats_by_snr[s]["loss"])
            row[f"accuracy_{s:+d}dB"] = float(stats_by_snr[s]["accuracy"])

        df = pd.DataFrame([row], columns=self.columns)
        write_header = not os.path.exists(self.csv_path)
        df.to_csv(self.csv_path, mode="a", header=write_header, index=False)