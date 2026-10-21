from __future__ import annotations

import copy
import json
import math
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional  # <<< CHANGED

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.common.confusion_matrix import (
    unwrap_batch_for_single_stream,
    ensure_seq_images,
)

# =========================
# Model output normalisation
# =========================

def unpack_model_output(out: Any) -> Dict[str, Optional[torch.Tensor]]:  # <<< CHANGED
    """
    Normalise model output into:
      {"logits": Tensor, "features": Tensor|None, "raw": original_out}
    Accepts:
      - logits
      - (features, logits)
      - (anything, logits) e.g. (_, logits)
      - dict with keys like {"logits": ..., "features": ...}
    """
    if isinstance(out, dict):
        logits = out.get("logits", None)
        features = out.get("features", None)
        if logits is None:
            raise ValueError("Model output dict must contain 'logits'.")
        return {"logits": logits, "features": features, "raw": out}

    if isinstance(out, (tuple, list)):
        if len(out) == 0:
            raise ValueError("Model returned an empty tuple/list.")
        if len(out) == 1:
            return {"logits": out[0], "features": None, "raw": out}
        # assume last element is logits (covers (features, logits) and (_, logits))
        return {
            "logits": out[-1],
            "features": out[0] if len(out) >= 2 else None,
            "raw": out,
        }

    # plain tensor == logits
    return {"logits": out, "features": None, "raw": out}

def compute_loss(loss_fn, *, y, logits, features=None):  # <<< CHANGED
    try:
        return loss_fn(logits, y)                 # common case
    except TypeError:
        return loss_fn(features, y, logits)       # feature-aware loss


@torch.no_grad()
def eval_frames(model: nn.Module, loader, device, loss_fn_ce: nn.Module, desc: str):
    model.eval()
    total_loss, n_frames, correct = 0.0, 0, 0
    C: int | None = None

    for batch in tqdm(loader, desc=desc, leave=False):
        xb, yb, _, msk, _ = unwrap_batch_for_single_stream(batch, prefer="power")
        xb = xb.to(device, non_blocking=True).float()
        yb = yb.to(device, non_blocking=True).float()
        msk = msk.to(device, non_blocking=True).bool()

        xb = ensure_seq_images(xb, yb)
        B, T, _, H, W = xb.shape
        _, _, C = yb.shape

        flat_mask = msk.view(B * T)
        xbt = xb.view(B * T, 3, H, W)[flat_mask]
        ybt_full = yb.view(B * T, C)[flat_mask]

        if xbt.numel() == 0:
            continue

        ybt = ybt_full.argmax(dim=-1)

        # >>> CHANGED: normalise any model output shape into logits/features <<<
        out = model(xbt)                          # <<< CHANGED
        pack = unpack_model_output(out)           # <<< CHANGED
        logits_bt = pack["logits"]                # <<< CHANGED
        features_bt = pack["features"]            # <<< CHANGED

        # >>> CHANGED: loss wrapper supports (logits,y) OR (features,y,logits) <<<
        loss = compute_loss(                      # <<< CHANGED
            loss_fn_ce,
            y=ybt,
            logits=logits_bt,
            features=features_bt,
        )

        total_loss += loss.item() * xbt.size(0)
        n_frames += xbt.size(0)
        correct += (logits_bt.argmax(dim=-1) == ybt).sum().item()

    avg_loss = 0.0 if n_frames == 0 else (total_loss / n_frames)
    acc = 0.0 if n_frames == 0 else (correct / n_frames)
    thresholds = (
        torch.zeros(C, device=device)
        if C is not None
        else torch.tensor([], device=device)
    )
    return {"loss": float(avg_loss), "accuracy": float(acc), "thresholds": thresholds}


def validate_across_snrs(*, model, val_loaders_by_snr, device, loss_fn_ce, ep: int):
    snr_keys = sorted(val_loaders_by_snr.keys())
    stats_by_snr = {}
    for s in snr_keys:
        stats_by_snr[s] = eval_frames(
            model=model,
            loader=val_loaders_by_snr[s],
            device=device,
            loss_fn_ce=loss_fn_ce,
            desc=f"val@{s:+d}dB frames ep{ep:02d}",
        )

    def macro(field: str) -> float:
        return (
            float(np.mean([float(stats_by_snr[s][field]) for s in snr_keys]))
            if snr_keys
            else 0.0
        )

    return stats_by_snr, macro("loss"), macro("accuracy")