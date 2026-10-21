from __future__ import annotations

import copy
import math
import os
from dataclasses import dataclass, asdict, field

from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional  # <<< CHANGED

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
import torch.optim as optim
from tqdm import tqdm
import torch.nn.functional as F


def ensure_seq_images(xb: torch.Tensor, yb: torch.Tensor) -> torch.Tensor:
    xb = xb.contiguous()
    B, T, _ = yb.shape  # yb is (B, T, C)

    if xb.dim() == 5:
        if xb.shape[2] == 3:
            return xb
        if xb.shape[2] == 1:
            return xb.repeat(1, 1, 3, 1, 1)
        if xb.shape[-1] == 3:
            return xb.permute(0, 1, 4, 2, 3).contiguous()
        if xb.shape[2] not in (1, 3):
            return xb.unsqueeze(2).repeat(1, 1, 3, 1, 1)
        raise ValueError(f"Unexpected 5D xb shape {tuple(xb.shape)}")

    if xb.dim() == 4:
        if xb.shape[1] == 3:
            N, _, H, W = xb.shape
            assert N == B * T, f"N={N} != B*T={B*T}"
            return xb.view(B, T, 3, H, W).contiguous()
        if xb.shape[1] == 1:
            xb = xb.repeat(1, 3, 1, 1)
            N, _, H, W = xb.shape
            assert N == B * T, f"N={N} != B*T={B*T}"
            return xb.view(B, T, 3, H, W).contiguous()
        if xb.shape[-1] == 3:
            xb = xb.permute(0, 3, 1, 2).contiguous()
            N, _, H, W = xb.shape
            assert N == B * T, f"N={N} != B*T={B*T}"
            return xb.view(B, T, 3, H, W).contiguous()
        raise ValueError(f"Unexpected 4D xb shape {tuple(xb.shape)}")

    raise ValueError(f"Unexpected xb rank {xb.dim()} with shape {tuple(xb.shape)}")


def unwrap_batch_for_single_stream(batch, prefer: str = "power"):
    if len(batch) == 5:
        xb, yb, yend, msk, meta = batch
        return xb, yb, yend, msk, meta
    if len(batch) == 6:
        xb_p, xb_q, yb, yend, msk, meta = batch
        xb = xb_p if prefer == "power" else xb_q
        return xb, yb, yend, msk, meta
    raise ValueError(f"Unexpected batch tuple length {len(batch)}")


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


@torch.no_grad()
def confusion_matrix_frames(
    model: nn.Module,
    loader,
    device: torch.device,
    num_classes: int,
    prefer: str = "power",
):
    """
    Frame-level confusion matrix over *valid frames only* (msk=True).
    Returns: np.ndarray shape (num_classes, num_classes), rows=true, cols=pred.
    """
    model.eval()
    cm = torch.zeros((num_classes, num_classes), dtype=torch.int64, device="cpu")

    for batch in tqdm(loader, desc="cm_frames", leave=False):
        xb, yb, _, msk, _ = unwrap_batch_for_single_stream(batch, prefer=prefer)
        xb = xb.to(device, non_blocking=True).float()
        yb = yb.to(device, non_blocking=True).float()
        msk = msk.to(device, non_blocking=True).bool()

        xb = ensure_seq_images(xb, yb)
        B, T, _, H, W = xb.shape
        C = yb.size(-1)

        flat_mask = msk.view(B * T)
        xbt = xb.view(B * T, 3, H, W)[flat_mask]
        ybt_full = yb.view(B * T, C)[flat_mask]

        if xbt.numel() == 0:
            continue

        y_true = ybt_full.argmax(dim=-1).detach().cpu()

        # >>> CHANGED: handle models that return logits OR (features, logits) etc. <<<
        out = model(xbt)                            # <<< CHANGED
        pack = unpack_model_output(out)             # <<< CHANGED
        logits_bt = pack["logits"]                  # <<< CHANGED
        y_pred = logits_bt.argmax(dim=-1).detach().cpu()  # <<< CHANGED

        # bincount trick for speed
        idx = y_true * num_classes + y_pred
        cm += torch.bincount(idx, minlength=num_classes * num_classes).view(
            num_classes, num_classes
        )

    return cm.numpy()


def save_confusion_matrix_csv(
    cm: np.ndarray,
    *,
    run_dir: str | os.PathLike,
    filename: str | os.PathLike,
    id_to_class: dict[int, str] | None = None,
):
    """
    Saves CSV under run_dir/confusion_matrices/<filename>.
    """
    run_dir = os.fspath(run_dir)  # Path -> str
    filename = os.fspath(filename)  # Path -> str

    out_dir = os.path.join(run_dir, "confusion_matrices")
    os.makedirs(out_dir, exist_ok=True)

    if id_to_class:
        labels = [id_to_class[i] for i in range(cm.shape[0])]
        df = pd.DataFrame(cm, index=labels, columns=labels)
    else:
        df = pd.DataFrame(cm)

    path = os.path.join(out_dir, filename)  # now guaranteed to be str
    df.to_csv(path)
    return path