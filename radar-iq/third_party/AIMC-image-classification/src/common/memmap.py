from __future__ import annotations

import copy
import json
import math
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import matplotlib.pyplot as plt


def _load_meta(memmap_dir: str) -> dict:
    meta_path = os.path.join(memmap_dir, "meta.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"meta.json not found in: {memmap_dir}")
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _open_memmap(memmap_dir: str, rel_path: str, dtype, shape):
    path = os.path.join(memmap_dir, rel_path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"memmap file missing: {path}")
    return np.memmap(path, dtype=dtype, mode="r", shape=shape)


@dataclass
class MemmapPowerStore:
    memmap_dir: str
    N: int
    H: int
    W: int
    C: int
    power_u8: np.memmap
    yend_u8: np.memmap
    snr_i16: np.memmap
    mod_i16: np.memmap
    stream_i32: np.memmap
    k_u8: np.memmap
    meta: dict = field(default_factory=dict)  # <-- add this

    @staticmethod
    def open(memmap_dir: str) -> "MemmapPowerStore":
        meta = _load_meta(memmap_dir)
        N = int(meta["N"])
        H = int(meta["H"])
        W = int(meta["W"])
        C = int(meta["C"])

        files = meta["files"]

        store = MemmapPowerStore(
            memmap_dir=memmap_dir,
            N=N,
            H=H,
            W=W,
            C=C,
            power_u8=_open_memmap(
                memmap_dir, files["power_u8"]["path"], np.uint8, (N, H, W)
            ),
            yend_u8=_open_memmap(
                memmap_dir, files["yend_u8"]["path"], np.uint8, (N, C)
            ),
            snr_i16=_open_memmap(memmap_dir, files["snr_i16"]["path"], np.int16, (N,)),
            mod_i16=_open_memmap(memmap_dir, files["mod_i16"]["path"], np.int16, (N,)),
            stream_i32=_open_memmap(
                memmap_dir, files["stream_i32"]["path"], np.int32, (N,)
            ),
            k_u8=_open_memmap(memmap_dir, files["k_u8"]["path"], np.uint8, (N,)),
            meta=meta,  # <-- add this
        )
        return store


def _filter_indices_by_snr(
    store: MemmapPowerStore, idx: np.ndarray, snr_value: int
) -> np.ndarray:
    snr_value = int(snr_value)
    return idx[(store.snr_i16[idx] == snr_value)]


class MemmapPowerFrameDataset(Dataset):
    def __init__(self, store: MemmapPowerStore):
        self.store = store
        self.C = store.C
        self.class_to_id: dict[str, int] = {}
        self.id_to_class: dict[int, str] = {}

    def __len__(self) -> int:
        return self.store.N

    def __getitem__(self, i: int):
        img = self.store.power_u8[i]
        xb = torch.from_numpy(np.asarray(img)).unsqueeze(0)
        y = self.store.yend_u8[i]
        yb = torch.from_numpy(np.asarray(y)).to(torch.float32).unsqueeze(0)
        yend = yb[0].clone()
        msk = torch.ones((1,), dtype=torch.bool)
        meta = {
            "snr": int(self.store.snr_i16[i]),
            "mod_id": int(self.store.mod_i16[i]),
            "stream_id": int(self.store.stream_i32[i]),
            "K": int(self.store.k_u8[i]),
        }
        return xb, yb, yend, msk, meta


class IndexedDataset(Dataset):
    def __init__(self, base: Dataset, indices: np.ndarray):
        self.base = base
        self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self) -> int:
        return int(self.indices.shape[0])

    def __getitem__(self, j: int):
        i = int(self.indices[j])
        return self.base[i]


def collate_memmap_singleframe(batch):
    x_list, yb_list, yend_list, msk_list, meta_list = zip(*batch)
    xb = torch.stack(x_list, dim=0)
    yb = torch.stack(yb_list, dim=0)
    yend = torch.stack(yend_list, dim=0)
    msk = torch.stack(msk_list, dim=0)
    meta = list(meta_list)
    return xb, yb, yend, msk, meta


def build_loaders_memmap(ds_cfg, train_cfg, rng: np.random.Generator):
    import os
    import numpy as np
    from torch.utils.data import Dataset, DataLoader

    root = os.path.normpath(ds_cfg.data_path)
    train_dir = os.path.join(root, "train")
    test_dir = os.path.join(root, "test")

    if not os.path.isdir(train_dir):
        raise FileNotFoundError(f"Expected train directory at: {train_dir}")
    if not os.path.isdir(test_dir):
        raise FileNotFoundError(f"Expected test directory at: {test_dir}")

    train_store = MemmapPowerStore.open(train_dir)
    test_store = MemmapPowerStore.open(test_dir)

    if train_store.C != test_store.C:
        raise RuntimeError(
            f"Train/Test memmap C mismatch: train={train_store.C}, test={test_store.C}"
        )
    if train_store.H != test_store.H or train_store.W != test_store.W:
        raise RuntimeError("Train/Test memmap spatial shape mismatch.")

    Cfull = int(train_store.C)

    present_snrs_test = set(int(x) for x in np.unique(test_store.snr_i16))
    wanted_snrs = set(int(s) for s in ds_cfg.snr_range)
    missing = wanted_snrs - present_snrs_test
    if missing:
        raise RuntimeError(
            f"Test memmap missing SNRs {sorted(missing)}. Present={sorted(present_snrs_test)}"
        )

    stored_mods = None
    for st in (train_store, test_store):
        if isinstance(st.meta, dict) and isinstance(
            st.meta.get("modulation_list", None), list
        ):
            stored_mods = st.meta["modulation_list"]
            break

    if stored_mods is None:
        if Cfull != len(ds_cfg.modulation_list):
            raise RuntimeError(
                "meta.json does not contain 'modulation_list', so I can't safely map selected modulations.\n"
                f"Got memmap C={Cfull} but ds_cfg.modulation_list has len={len(ds_cfg.modulation_list)}.\n"
                "Either export memmap with meta['modulation_list'] or set ds_cfg.modulation_list "
                "to the full stored order (len == C)."
            )
        stored_mods = list(ds_cfg.modulation_list)

    if len(stored_mods) != Cfull:
        raise RuntimeError(
            f"meta['modulation_list'] length mismatch: len={len(stored_mods)} but C={Cfull}."
        )

    stored_mod_to_id = {str(m): int(i) for i, m in enumerate(stored_mods)}

    selected_mods = list(ds_cfg.modulation_list)
    if len(selected_mods) == 0:
        raise RuntimeError("ds_cfg.modulation_list is empty; nothing to train on.")

    missing_mods = [m for m in selected_mods if str(m) not in stored_mod_to_id]
    if missing_mods:
        raise RuntimeError(
            f"Selected modulations not present in memmap meta: {missing_mods}\n"
            f"Memmap contains: {stored_mods}"
        )

    selected_stored_ids = np.array(
        [stored_mod_to_id[str(m)] for m in selected_mods], dtype=np.int64
    )
    selected_stored_ids_sorted = np.sort(selected_stored_ids)

    class_to_id = {str(m): int(i) for i, m in enumerate(selected_mods)}
    id_to_class = {int(i): str(m) for m, i in class_to_id.items()}

    old_to_new = {
        int(stored_mod_to_id[str(m)]): int(class_to_id[str(m)]) for m in selected_mods
    }

    def _filter_idx_by_mod(store: MemmapPowerStore, idx: np.ndarray) -> np.ndarray:
        mid = store.mod_i16[idx].astype(np.int64, copy=False)
        keep = np.isin(mid, selected_stored_ids_sorted, assume_unique=False)
        return idx[keep]

    train_base = MemmapPowerFrameDataset(train_store)
    test_base = MemmapPowerFrameDataset(test_store)

    train_base.class_to_id = class_to_id
    train_base.id_to_class = id_to_class
    test_base.class_to_id = class_to_id
    test_base.id_to_class = id_to_class

    class RemapAndSliceDataset(Dataset):
        def __init__(
            self,
            base: MemmapPowerFrameDataset,
            y_cols: np.ndarray,
            old_to_new_map: dict[int, int],
        ):
            self.base = base
            self.y_cols = np.asarray(
                y_cols, dtype=np.int64
            )  # stored ids used as column indices
            self.old_to_new_map = {int(k): int(v) for k, v in old_to_new_map.items()}
            self.class_to_id = base.class_to_id
            self.id_to_class = base.id_to_class

        def __len__(self):
            return len(self.base)

        def __getitem__(self, i: int):
            xb, yb, yend, msk, meta = self.base[i]

            yb = yb[:, self.y_cols]
            yend = yend[self.y_cols]

            old = int(meta.get("mod_id", -1))
            meta = dict(meta)
            meta["mod_id_full"] = old

            meta["mod_id"] = int(self.old_to_new_map.get(old, -1))

            return xb, yb, yend, msk, meta

    y_cols_in_user_order = np.array(
        [stored_mod_to_id[str(m)] for m in selected_mods], dtype=np.int64
    )

    train_base2 = RemapAndSliceDataset(train_base, y_cols_in_user_order, old_to_new)
    test_base2 = RemapAndSliceDataset(test_base, y_cols_in_user_order, old_to_new)

    # --- TRAIN: all frames from train/ for selected mods ---
    train_idx_all = _filter_idx_by_mod(
        train_store, np.arange(train_store.N, dtype=np.int64)
    )
    train_ds = IndexedDataset(train_base2, train_idx_all)

    train_loader = DataLoader(
        train_ds,
        batch_size=train_cfg.batch_size_train,
        shuffle=True,
        num_workers=getattr(train_cfg, "num_workers", 0),
        pin_memory=getattr(train_cfg, "pin_memory", False),
        persistent_workers=(
            getattr(train_cfg, "num_workers", 0) > 0
            and getattr(train_cfg, "persistent_workers", False)
        ),
        prefetch_factor=(
            getattr(train_cfg, "prefetch_factor", 2)
            if getattr(train_cfg, "num_workers", 0) > 0
            else None
        ),
        collate_fn=collate_memmap_singleframe,
    )

    test_idx_all = _filter_idx_by_mod(
        test_store, np.arange(test_store.N, dtype=np.int64)
    )

    val_loaders_by_snr = {}
    for s in ds_cfg.snr_range:
        idx_s = _filter_indices_by_snr(test_store, test_idx_all, int(s))
        val_ds = IndexedDataset(test_base2, idx_s)
        val_loader = DataLoader(
            val_ds,
            batch_size=train_cfg.batch_size_val,
            shuffle=False,
            num_workers=getattr(train_cfg, "num_workers", 0),
            pin_memory=getattr(train_cfg, "pin_memory", False),
            persistent_workers=(
                getattr(train_cfg, "num_workers", 0) > 0
                and getattr(train_cfg, "persistent_workers", False)
            ),
            prefetch_factor=(
                getattr(train_cfg, "prefetch_factor", 2)
                if getattr(train_cfg, "num_workers", 0) > 0
                else None
            ),
            collate_fn=collate_memmap_singleframe,
        )
        val_loaders_by_snr[int(s)] = val_loader

    return train_base2, train_loader, val_loaders_by_snr

def _get_stored_mod_to_id(store) -> dict[str, int]:
    mods = store.meta.get("modulation_list", None)
    if not isinstance(mods, list) or len(mods) == 0:
        raise RuntimeError(
            "meta.json is missing 'modulation_list'. "
            "Re-export with meta['modulation_list'] so we can map '4fsk' -> stored id safely."
        )
    return {str(m): int(i) for i, m in enumerate(mods)}

def extract_one_image_per_snr(
    memmap_dir: str,
    modulation: str,
    snrs=(6, 3, 0, -3, -6),
    out_dir: str | None = None,
    seed: int = 0,
):
    """
    Extracts ONE power spectrogram image per requested SNR for the given modulation,
    saving PNGs to out_dir, and returns a dict snr -> (index, image_array_uint8).

    memmap_dir: .../train or .../test (folder containing meta.json + memmaps)
    modulation: e.g. "4fsk"
    snrs      : iterable of ints
    """
    store = MemmapPowerStore.open(memmap_dir)
    mod_to_id = _get_stored_mod_to_id(store)

    if str(modulation) not in mod_to_id:
        raise KeyError(
            f"Modulation '{modulation}' not found in memmap meta modulation_list.\n"
            f"Available examples: {list(mod_to_id.keys())[:20]} ..."
        )

    target_mod_id = mod_to_id[str(modulation)]
    snrs = [int(s) for s in snrs]

    # indices for the modulation across all SNRs
    idx_mod = np.flatnonzero(store.mod_i16[:] == target_mod_id)

    rng = np.random.default_rng(seed)
    results = {}

    if out_dir is None:
        out_dir = os.path.join(memmap_dir, f"extract_{modulation}")
    os.makedirs(out_dir, exist_ok=True)

    for s in snrs:
        idx_s = idx_mod[store.snr_i16[idx_mod] == s]
        if idx_s.size == 0:
            results[s] = None
            continue

        chosen = int(rng.choice(idx_s))
        img = np.asarray(store.power_u8[chosen])  # (H,W) uint8

        # save PNG
        out_path = os.path.join(out_dir, f"{modulation}_snr_{s:+d}dB_idx_{chosen}.png")
        plt.figure()
        plt.imshow(img, cmap="gray", vmin=0, vmax=255)
        plt.title(f"{modulation} @ SNR {s:+d} dB (idx {chosen})")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(out_path, dpi=200)
        plt.close()

        results[s] = (chosen, img)

    # handy summary
    print(f"Saved to: {out_dir}")
    for s in snrs:
        if results[s] is None:
            print(f"SNR {s:+d} dB: no samples found")
        else:
            print(f"SNR {s:+d} dB: idx={results[s][0]}  shape={results[s][1].shape} dtype={results[s][1].dtype}")

    return results

def export_sample_folder(
    memmap_dir: str,
    *,
    modulation: str,
    snr: int,
    n_samples: int = 20,
    out_dir: str | None = None,
    seed: int = 42,
    flip_vertical: bool = True,
    cmap: str = "turbo",
    add_colorbar: bool = True,
) -> dict[str, Any]:
    """
    Export a folder of sample spectrogram images for ONE modulation at ONE SNR.

    Saves PNGs:
      <out_dir>/<modulation>_snr_<+dB>_sample_<k>_idx_<idx>.png

    Returns a small summary dict with chosen indices and counts.
    """
    store = MemmapPowerStore.open(memmap_dir)
    mod_to_id = _get_stored_mod_to_id(store)

    if str(modulation) not in mod_to_id:
        raise KeyError(
            f"Modulation '{modulation}' not found in memmap meta modulation_list.\n"
            f"Available examples: {list(mod_to_id.keys())[:20]} ..."
        )

    target_mod_id = int(mod_to_id[str(modulation)])
    snr = int(snr)
    n_samples = int(n_samples)

    # Filter candidate indices
    idx_mod = np.flatnonzero(store.mod_i16[:] == target_mod_id)
    idx_s = idx_mod[store.snr_i16[idx_mod] == snr]

    if idx_s.size == 0:
        raise RuntimeError(f"No samples found for modulation={modulation} at SNR={snr} dB.")

    rng = np.random.default_rng(seed)

    k = min(n_samples, int(idx_s.size))
    chosen = rng.choice(idx_s, size=k, replace=False).astype(np.int64)

    if out_dir is None:
        out_dir = os.path.join(memmap_dir, f"samples_{modulation}_snr_{snr:+d}dB")
    os.makedirs(out_dir, exist_ok=True)

    # Save images
    for j, idx in enumerate(chosen):
        img = np.asarray(store.power_u8[int(idx)])  # (H,W) uint8

        if flip_vertical:
            img = np.flipud(img)

        out_path = os.path.join(
            out_dir,
            f"{modulation}_snr_{snr:+d}dB_sample_{j:03d}_idx_{int(idx)}.png",
        )

        plt.figure(figsize=(6, 5))
        im = plt.imshow(img, cmap=cmap, vmin=0, vmax=255, aspect="auto")
        plt.title(f"{modulation} | SNR {snr:+d} dB | idx {int(idx)}")
        plt.axis("off")
        if add_colorbar:
            cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
            cbar.set_label("Power (normalised)")
        plt.tight_layout()
        plt.savefig(out_path, dpi=200)
        plt.close()

    summary = {
        "memmap_dir": memmap_dir,
        "out_dir": out_dir,
        "modulation": str(modulation),
        "snr_db": snr,
        "requested": n_samples,
        "saved": int(k),
        "available": int(idx_s.size),
        "indices": [int(x) for x in chosen.tolist()],
    }

    print(f"Saved {summary['saved']} / {summary['requested']} samples to: {out_dir}")
    return summary