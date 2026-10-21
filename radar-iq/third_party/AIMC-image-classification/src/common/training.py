from __future__ import annotations

import copy
import math
import os
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import amp
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

from src.common.confusion_matrix import (
    unwrap_batch_for_single_stream,
    ensure_seq_images,
)

def unpack_model_output(out: Any) -> Dict[str, Optional[torch.Tensor]]:
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
        return {"logits": out[-1], "features": out[0] if len(out) >= 2 else None, "raw": out}

    # plain tensor == logits
    return {"logits": out, "features": None, "raw": out}

def compute_loss(loss_fn, *, y, logits, features=None):
    try:
        return loss_fn(logits, y)                 # common case
    except TypeError:
        return loss_fn(features, y, logits)       # feature-aware loss

def train_one_epoch(
    *,
    model: nn.Module,
    loader,
    device,
    opt,
    loss_fn_ce,
    ctx,
    scaler,
    update_every: int,
    ema_alpha: float,
    ep: int,
    max_epochs: int,
):
    model.train()
    running, n_seen = 0.0, 0
    ema_loss = None

    pbar = tqdm(loader, desc=f"Epoch {ep}/{max_epochs}", leave=False)
    for i, batch in enumerate(pbar):
        xb, yb, _, msk, _ = unwrap_batch_for_single_stream(batch, prefer="power")
        xb = xb.to(device, non_blocking=True).float()
        yb = yb.to(device, non_blocking=True).float()
        msk = msk.to(device, non_blocking=True).bool()

        xb = ensure_seq_images(xb, yb)
        B, T, _, H, W = xb.shape
        C = yb.size(-1)

        flat = msk.view(B * T)

        # Select only valid frames for supervision
        xbt = xb.view(B * T, 3, H, W)[flat]
        ybt = yb.view(B * T, C)[flat].argmax(-1)

        # Skip batch if nothing valid
        if xbt.numel() == 0:
            continue

        opt.zero_grad(set_to_none=True)
        with ctx():
            # >>> CHANGED: normalise any model output shape into logits/features <<<
            out = model(xbt)                              # <<< CHANGED
            pack = unpack_model_output(out)               # <<< CHANGED
            logits = pack["logits"]                       # <<< CHANGED
            features = pack["features"]                   # <<< CHANGED

            # >>> CHANGED: loss wrapper supports (logits,y) OR (features,y,logits) <<<
            loss = compute_loss(                           # <<< CHANGED
                loss_fn_ce,
                y=ybt,
                logits=logits,
                features=features,
            )

        if scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            opt.step()

        batch_frames = xbt.size(0)
        running += loss.item() * batch_frames
        n_seen += batch_frames

        ema_loss = (
            loss.item()
            if ema_loss is None
            else (1 - ema_alpha) * ema_loss + ema_alpha * loss.item()
        )
        if (i % update_every) == 0:
            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                ema=f"{ema_loss:.4f}",
                lr=f"{opt.param_groups[0]['lr']:.2e}",
            )

    return running / max(1, n_seen)