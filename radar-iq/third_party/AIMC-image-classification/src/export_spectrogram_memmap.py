"""
AIMC-Spec Power Spectrogram Memmap Exporter
===========================================

This module exports IQ stream data from the AIMC-Spec HDF5 dataset format
into fixed-size memory-mapped (memmap) files containing power spectrogram
images and aligned per-frame labels/metadata.

It is intended to be used after the HDF5 dataset files have already been
generated, including:
    - the training subset HDF5 files
    - the testing HDF5 files

Like the other dataset preparation scripts in this project, this file is
not normally run directly. Its `main()` function is imported and executed
through the project-level `main.py` entrypoint.

----------------------------------------------------------------------
Where This File Sits in the Pipeline
----------------------------------------------------------------------

This script is part of the dataset conversion pipeline:

1. Generate or prepare emitter HDF5 files
2. Build the reduced training subset HDF5 files (if required)
3. Use this script to export those HDF5 files into memmap files
4. Train models using the memmap-based dataset loader

In other words, this file does not create the original emitter HDF5 files.
It converts an already-prepared HDF5 dataset into a memmap representation
that is faster and more practical for spectrogram-based training.

----------------------------------------------------------------------
What This File Produces
----------------------------------------------------------------------

Given an input AIMC-Spec HDF5 dataset directory, this module creates a
memmap export directory containing:

    power_u8.dat     : (N, 224, 224) uint8
        Power spectrogram images stored as 8-bit grayscale arrays.

    yend_u8.dat      : (N, C) uint8
        Multi-hot end-label targets repeated per frame.

    snr_i16.dat      : (N,) int16
        SNR value for each frame.

    mod_i16.dat      : (N,) int16
        Integer modulation class ID for each frame.

    stream_i32.dat   : (N,) int32
        Stream identifier for each frame.

    k_u8.dat         : (N,) uint8
        Number of active emitters in the stream. This exporter expects K=1.

    meta.json
        Metadata describing file paths, shapes, dtypes, and the class mapping.

These files are written into a split-specific folder such as:

    <out_root>/train/
    <out_root>/test/
    <out_root>/val/

depending on how the script is configured.

----------------------------------------------------------------------
Expected Inputs
----------------------------------------------------------------------

This exporter assumes the dataset already exists in the AIMC-Spec HDF5
layout used by the project. In practice, this usually means:

    C:\\Apps\\Code\\AIMC_Spec_v2_train\\
    C:\\Apps\\Code\\AIMC_Spec_v2_test\\

or similar paths containing one HDF5 file per modulation type.

The loader used in this script (`prepMultiHotStreamLoader`) reads those
HDF5 files, generates IQ stream segments, and passes batches into the
export process.

----------------------------------------------------------------------
How the Export Works
----------------------------------------------------------------------

The export is performed in two passes over the loader:

1. Counting pass
   The script first iterates over the loader to determine the total
   number of output frames `N`, because memmap files must be allocated
   with a fixed shape before writing.

2. Writing pass
   The script iterates over the loader again, converts each IQ segment
   into a power spectrogram, resizes it to 224 x 224, converts it to
   uint8, and writes the image plus aligned labels/metadata into the
   memmap files.

This two-pass design is necessary because NumPy memmaps are not dynamically
resized like normal Python lists.

----------------------------------------------------------------------
Important Assumptions
----------------------------------------------------------------------

- This exporter is designed for power spectrogram export only.
- It expects single-emitter streams only:
      K = 1
- The collate function must return batches in the form:
      (x_bt, ys, yend, msk, metas, T)
- Spectrograms are generated in batch using
      batched_iq_to_power_spectrogram(...)
- Output images are resized to 224 x 224.
- The modulation-to-ID mapping must match the number of label columns C.

If the dataset is multi-emitter or uses a different metadata format,
this script will need to be adjusted.

----------------------------------------------------------------------
How to Use This File
----------------------------------------------------------------------

This file is usually called through `main.py`, which imports and executes
the `main()` function defined here.

Typical usage flow:

1. Set `split` to the dataset split you want to export
   (for example `"train"` or `"test"`)

2. Set `ds_cfg.data_path` to the directory containing the relevant HDF5 files

3. Set `out_root` to the base directory where the memmap files should be written

4. Run the project entrypoint so that this module's `main()` function is called

The current `main()` function is configured to:
- define the modulation list
- build a dataset configuration
- build the stream loader
- infer the class mapping
- export the power spectrogram memmaps

----------------------------------------------------------------------
Example Intent
----------------------------------------------------------------------

A common use case is:

- Create the HDF5 training subset files first
- Create the HDF5 test files
- Run this exporter once for the training split
- Run this exporter again for the test split

This results in separate memmap directories that can be consumed directly
by later training code.

----------------------------------------------------------------------
Project-Specific Notes
----------------------------------------------------------------------

- This file follows the repo's existing path and import conventions.
- Compression is not used here because memmap files are written as raw
  fixed-size binary arrays for fast random access.
- The loader is built with `persistent_workers=False` to avoid issues
  during the two-pass export process.
- If emitter counts per modulation are not supplied, the current script
  exports all available emitters for the configured split.

----------------------------------------------------------------------
Key Functions
----------------------------------------------------------------------

collate_stream_segments_for_export(...)
    Converts loader batches into a format suitable for batched STFT export.

_count_total_frames_and_C(...)
    Performs the counting pass to determine total number of frames and
    the label dimension C.

spec01_to_uint8_224(...)
    Resizes a [0,1] spectrogram to 224 x 224 and converts it to uint8.

export_loader_to_memmap_power_only(...)
    Main export routine that writes all memmap files and meta.json.

main()
    Example project entrypoint for exporting one dataset split.

----------------------------------------------------------------------
Summary
----------------------------------------------------------------------

Use this module after the AIMC-Spec HDF5 dataset files already exist.
Its job is to convert those HDF5-based IQ streams into memmap-based
power spectrogram files for faster downstream model training.
"""

from __future__ import annotations

import os
import json
import numpy as np
import torch
import torch.nn.functional as F

from src.STFT_CNN.STFT_CNN_MemMap import DatasetConfig
from src.common.MultiHotStreamLoader import prepMultiHotStreamLoader as prepLoader
from src.common.spectrogram import batched_iq_to_power_spectrogram


def _meta_get_required(meta, keys, name: str):
    for k in keys:
        if k in meta:
            return meta[k]
    raise KeyError(
        f"meta missing required '{name}', tried keys={keys}, meta_keys={list(meta.keys())}"
    )


def collate_stream_segments_for_export(batch):
    """
    Returns IQ stacked so we can do batched STFT once per batch.
    batch: list of (x_seq, y_seq, y_last, mask, meta)

    Output:
      x_bt : (B*T, 2, W) float32 contiguous
      ys   : stacked
      yend : stacked
      msk  : stacked
      metas: list
      T    : int
    """
    xs, ys, yend, msk, metas = zip(*batch)

    def _norm_x(x):
        if not torch.is_tensor(x):
            x = torch.as_tensor(x)
        x = x.detach()
        if x.ndim != 3:
            raise ValueError(f"expected (T,2,W) or (T,W,2), got {tuple(x.shape)}")
        if x.shape[1] == 2:
            pass
        elif x.shape[-1] == 2:
            x = x.movedim(-1, 1)
        else:
            raise ValueError(f"missing IQ dim, got {tuple(x.shape)}")
        return x.contiguous().to(torch.float32)

    xs = tuple(_norm_x(x) for x in xs)  # each: (T,2,W)
    B = len(xs)
    T = int(xs[0].shape[0])
    W = int(xs[0].shape[-1])

    # --- FAST STACK: (B,T,2,W) -> (B*T,2,W)
    x_b_t_2_w = torch.stack(xs, dim=0)  # (B,T,2,W)
    x_bt = x_b_t_2_w.reshape(B * T, 2, W).contiguous()  # (B*T,2,W)

    ys = torch.stack(ys, dim=0)
    yend = torch.stack(yend, dim=0)
    msk = torch.stack(msk, dim=0)

    return x_bt, ys, yend, msk, list(metas), T


def _count_total_frames_and_C(loader):
    """
    Counting pass only (no STFT). Returns (N_total_frames, C).
    """
    total = 0
    C = None
    for batch in loader:
        x_bt, ys, yend, msk, metas, T = batch
        B = int(yend.shape[0])
        T = int(T)
        total += B * T
        if C is None:
            C = int(yend.shape[1])
    if C is None:
        raise RuntimeError("Could not infer C (yend.shape[1]) from loader.")
    return int(total), int(C)


def _make_emitter_split_by_mod_from_counts(
    modulation_list: list[str],
    emitter_counts_by_mod: dict[str, int],
    *,
    train_frac: float = 0.8,
    seed: int = 42,
):
    """
    Returns (train_keep_by_mod, val_keep_by_mod) where each is:
      { mod: [indices-within-that-mod-block] }

    This matches prepMultiHotStreamLoader's emitter_keep_by_mod expectation.
    """
    rng = np.random.default_rng(seed)
    train_keep: dict[str, list[int]] = {}
    val_keep: dict[str, list[int]] = {}

    for mod in modulation_list:
        n = int(emitter_counts_by_mod.get(mod, 0))
        if n <= 0:
            train_keep[mod] = []
            val_keep[mod] = []
            continue

        perm = rng.permutation(n)
        n_train = int(np.floor(train_frac * n))
        train_keep[mod] = np.sort(perm[:n_train]).tolist()
        val_keep[mod] = np.sort(perm[n_train:]).tolist()

    return train_keep, val_keep



def spec01_to_uint8_224(spec_ft: torch.Tensor, size: int = 224) -> torch.Tensor:
    """
    spec_ft: (F,T) float in [0,1]
    returns: (224,224) uint8 on CPU
    """
    if spec_ft.ndim != 2:
        raise ValueError(f"expected (F,T), got {tuple(spec_ft.shape)}")

    x = spec_ft.unsqueeze(0).unsqueeze(0)  # (1,1,F,T)
    x = F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)
    x = x.squeeze(0).squeeze(0)  # (224,224)
    x = (x.clamp(0, 1) * 255.0).round().to(torch.uint8)
    return x.cpu()


@torch.no_grad()
def export_loader_to_memmap_power_only(
    loader,
    out_dir: str,
    *,
    mod_to_id: dict[str, int],
    n_fft=64,
    win_len=64,
    hop_len=32,
):
    """
    Expects batches from collate_stream_segments_for_export:
      (x_bt, ys, yend, msk, metas, T)

    Writes memmaps:
      power_u8   : (N,224,224) uint8
      yend_u8    : (N,C) uint8
      snr_i16    : (N,) int16
      mod_i16    : (N,) int16
      stream_i32 : (N,) int32
      k_u8       : (N,) uint8
      meta.json
    """

    os.makedirs(out_dir, exist_ok=True)

    # memmap is fixed-size => count first
    N_total, C = _count_total_frames_and_C(loader)
    H = W = 224

    power_path = os.path.join(out_dir, "power_u8.dat")
    yend_path = os.path.join(out_dir, "yend_u8.dat")
    snr_path = os.path.join(out_dir, "snr_i16.dat")
    mod_path = os.path.join(out_dir, "mod_i16.dat")
    stream_path = os.path.join(out_dir, "stream_i32.dat")
    k_path = os.path.join(out_dir, "k_u8.dat")
    meta_path = os.path.join(out_dir, "meta.json")

    power_mm = np.memmap(power_path, dtype=np.uint8, mode="w+", shape=(N_total, H, W))
    yend_mm = np.memmap(yend_path, dtype=np.uint8, mode="w+", shape=(N_total, C))
    snr_mm = np.memmap(snr_path, dtype=np.int16, mode="w+", shape=(N_total,))
    mod_mm = np.memmap(mod_path, dtype=np.int16, mode="w+", shape=(N_total,))
    stream_mm = np.memmap(stream_path, dtype=np.int32, mode="w+", shape=(N_total,))
    k_mm = np.memmap(k_path, dtype=np.uint8, mode="w+", shape=(N_total,))

    written = 0

    # Iterate again (DataLoader is reusable per epoch)
    for batch in loader:
        x_bt, ys, yend, msk, metas, T = batch
        T = int(T)
        B = int(yend.shape[0])
        BT = B * T

        # --- batched POWER ---
        p_bt = batched_iq_to_power_spectrogram(
            x_bt,
            n_fft=n_fft,
            win_len=win_len,
            hop_len=hop_len,
            center_dc=True,
            eps=1e-12,
            dyn_range_db=None,
            normalise_noise=True,
            noise_floor_mode="fixed",
            noise_floor_db=1.0,
            to_db=True,
            db_min=-100.0,
            db_max=20.0,
            return_mode="snr01",
        )  # (BT, F, TT)

        # Convert to uint8 images (SAFE for CUDA/CPU)
        p_imgs = (
            torch.stack(
                [spec01_to_uint8_224(p_bt[i]) for i in range(p_bt.shape[0])],
                dim=0,
            )
            .cpu()
            .numpy()
        )  # (BT,224,224) uint8

        # Repeat yend rows T times -> (BT,C)
        yend_rep = yend.repeat_interleave(T, dim=0).to(torch.uint8).cpu().numpy()

        # Per-frame meta
        snr_rep = np.empty((BT,), dtype=np.int16)
        mod_rep = np.empty((BT,), dtype=np.int16)
        sid_rep = np.empty((BT,), dtype=np.int32)
        k_rep = np.empty((BT,), dtype=np.uint8)

        out_idx = 0
        for b in range(B):
            meta = metas[b]
            snr = int(_meta_get_required(meta, ("snr", "snr_db"), "snr"))
            K = int(_meta_get_required(meta, ("K",), "K"))
            perm = _meta_get_required(meta, ("perm",), "perm")

            if not isinstance(perm, (tuple, list)) or len(perm) < 1:
                raise ValueError(f"meta['perm'] malformed: {perm!r}")
            if K != 1:
                raise ValueError(f"export expects K=1; got K={K}, perm={perm}")

            mod = str(perm[0])
            if mod not in mod_to_id:
                raise KeyError(
                    f"mod '{mod}' not in mod_to_id keys={list(mod_to_id.keys())}"
                )
            mod_id = int(mod_to_id[mod])

            stream_id = int(_meta_get_required(meta, ("stream_id",), "stream_id"))

            snr_rep[out_idx : out_idx + T] = snr
            mod_rep[out_idx : out_idx + T] = mod_id
            sid_rep[out_idx : out_idx + T] = stream_id
            k_rep[out_idx : out_idx + T] = K
            out_idx += T

        # First-batch sanity check (alignment + repetition)
        if written == 0:
            assert (
                p_imgs.shape[0] == BT
            ), f"p_imgs has {p_imgs.shape[0]} rows, expected BT={BT}"
            assert (
                yend_rep.shape[0] == BT
            ), f"yend_rep has {yend_rep.shape[0]} rows, expected BT={BT}"
            assert np.all(
                sid_rep[0:T] == sid_rep[0]
            ), "stream_id not repeating across first T frames"
            assert np.all(
                snr_rep[0:T] == snr_rep[0]
            ), "snr not repeating across first T frames"
            assert np.all(
                mod_rep[0:T] == mod_rep[0]
            ), "mod_id not repeating across first T frames"
            assert np.all(
                k_rep[0:T] == k_rep[0]
            ), "K not repeating across first T frames"
            assert np.all(
                yend_rep[0:T] == yend_rep[0]
            ), "yend not repeating across first T frames"
            print("✅ export sanity: first-stream repetition passed")

        end = written + BT
        if end > N_total:
            raise RuntimeError(f"Overflow: writing {end} > allocated {N_total}")

        power_mm[written:end] = p_imgs
        yend_mm[written:end] = yend_rep
        snr_mm[written:end] = snr_rep
        mod_mm[written:end] = mod_rep
        stream_mm[written:end] = sid_rep
        k_mm[written:end] = k_rep

        written = end

        if written % 10_000 == 0:
            power_mm.flush()
            yend_mm.flush()
            snr_mm.flush()
            mod_mm.flush()
            stream_mm.flush()
            k_mm.flush()
            print(f"written {written:,}/{N_total:,} frames -> {out_dir}")

    power_mm.flush()
    yend_mm.flush()
    snr_mm.flush()
    mod_mm.flush()
    stream_mm.flush()
    k_mm.flush()

    # --- build a stable id->name mapping for meta.json ---
    if len(mod_to_id) != C:
        raise RuntimeError(
            f"export: len(mod_to_id)={len(mod_to_id)} but counted C={C}. "
            "These must match, because yend_u8 has C columns."
        )

    id_to_mod = [None] * C
    for m, i in mod_to_id.items():
        i = int(i)
        if i < 0 or i >= C:
            raise RuntimeError(f"export: mod_to_id['{m}'] has invalid id {i} for C={C}")
        if id_to_mod[i] is not None:
            raise RuntimeError(
                f"export: duplicate class id {i} for mods {id_to_mod[i]} and {m}"
            )
        id_to_mod[i] = str(m)

    if any(v is None for v in id_to_mod):
        missing = [i for i, v in enumerate(id_to_mod) if v is None]
        raise RuntimeError(f"export: id_to_mod missing entries at ids={missing}")

    meta = {
        "N": N_total,
        "H": H,
        "W": W,
        "C": C,
        # ✅ THIS is what build_loaders_memmap needs to safely subset later
        "modulation_list": id_to_mod,  # index = class id
        "mod_to_id": {k: int(v) for k, v in mod_to_id.items()},
        "files": {
            "power_u8": {
                "path": "power_u8.dat",
                "dtype": "uint8",
                "shape": [N_total, H, W],
            },
            "yend_u8": {"path": "yend_u8.dat", "dtype": "uint8", "shape": [N_total, C]},
            "snr_i16": {"path": "snr_i16.dat", "dtype": "int16", "shape": [N_total]},
            "mod_i16": {"path": "mod_i16.dat", "dtype": "int16", "shape": [N_total]},
            "stream_i32": {
                "path": "stream_i32.dat",
                "dtype": "int32",
                "shape": [N_total],
            },
            "k_u8": {"path": "k_u8.dat", "dtype": "uint8", "shape": [N_total]},
        },
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"✅ done. total frames written: {written:,} -> {out_dir}")
    return out_dir


# ─────────────────────────────────────────────────────────────────────────────
# __main__  (EXPORT POWER SPECS -> MEMMAP uint8, with 80/20 split)
# ─────────────────────────────────────────────────────────────────────────────
def main():
    import numpy as np

    modulation_list = [
        ###### FM ######
        "unmodulated",
        "lfm_up",
        "lfm_down",
        "dlfm_up_down",
        "NLFM_up",
        "NLFM_down",
        "mlfm",
        "dlfm_down_up",
        ###### PM ######
        "bpsk",
        "qpsk",
        "barker_11",
        "barker_13",
        "p1",
        "p2",
        "p3",
        "p4",
        ### Not Used ###
        # FM
        "triangle",
        "exp",
        "bfsk",
        "4fsk",
        "eqfm",
        "sfm",
        "costas",
        # PM
        "barker_2_1",
        "barker_2_2",
        "barker_3",
        "barker_4_1",
        "barker_4_2",
        "barker_5",
        "barker_7",
    ]

    split = "test"

    ds_cfg = DatasetConfig(
        modulation_list=modulation_list,
        noise_label="noise",
        snr_range=[
            6,
            3,
            0,
            -3,
            -6,
        ],
        #
        window_count=1,
        window_size=1000,
        window_step_size=1000,
        #
        k_ratio={1: 1.0},
        history_length=1,
        stride=1,
        threshold_time_frac=0.02,
        train_noise_pos_ratio=0.5,
        val_noise_pos_ratio=0.05,
        data_path=f"C:\\Apps\\Code\\AIMC_Spec_v2_{split}\\",
        #
        use_noise=False,  # toggle
    )

    # Where to write the exported uint8 power spectrograms (MEMMAP root)
    out_root = r"C:\\Apps\\Code\\power_specs_memmap\\"
    out_train = os.path.join(out_root, split)

    rng = np.random.default_rng(42)

    # --- Build 80/20 emitter split using only emitter counts per mod ---
    # Since prepMultiHotStreamLoader builds emitters internally, we can't
    # directly access index_map here without importing deeper internals.
    #
    # Workaround: run one "dry" loader build (no export) to get summary counts.
    # If you already have a reliable way to get emitter counts per mod, replace
    # this block and pass train/val emitter_keep_by_mod.
    #
    # Pragmatic approach: use emitter_keep_by_mod=None by default (export all),
    # OR set emitter_keep_by_mod externally if you already compute it elsewhere.
    #
    # Here we assume you DO have emitter_keep_by_mod support and you will pass it.
    #
    # If you want me to wire this perfectly: paste the import path for
    # load_emitters_from_mod_list (or where index_map comes from) in your repo.

    # For now, do a deterministic per-mod split using a placeholder emitter counts dict:
    # >>> REPLACE emitter_counts_by_mod with real counts if you have them <<<
    # If you don't replace it, leave emitter_keep_by_mod=None below and export all.
    emitter_counts_by_mod = {m: 0 for m in modulation_list}  # <-- REPLACE

    train_keep_by_mod, _ = _make_emitter_split_by_mod_from_counts(
        modulation_list=modulation_list,
        emitter_counts_by_mod=emitter_counts_by_mod,
        train_frac=1,
        seed=42,
    )

    # Common kwargs (EXPORT MODE)
    common_loader_kwargs = dict(
        modulation_list=ds_cfg.modulation_list,
        snr_range=ds_cfg.snr_range,
        filepath=ds_cfg.data_path,
        batch_size=64,
        history_length=ds_cfg.history_length,
        stride=ds_cfg.stride,
        window_count=ds_cfg.window_count,
        window_size=ds_cfg.window_size,
        window_step_size=ds_cfg.window_step_size,
        k_ratio=ds_cfg.k_ratio,
        seed=42,
        rng=rng,
        split="train",
        balance_mode="global",
        noise_pos_ratio=0.0,
        resample_seed=None,
        shuffle=False,
        num_workers=4,
        persistent_workers=False,
        prefetch_factor=2,
        pin_memory=False,
        image_mode=False,
        collate_fn=collate_stream_segments_for_export,
        progress_steps=True,
        summary_prints=True,
        threshold_time_frac=ds_cfg.threshold_time_frac,
        add_noise_class=False,
        noise_label=ds_cfg.noise_label,
        drop_ambiguous=False,
        snr_levels=tuple(ds_cfg.snr_range),
        balance_snr=False,
    )

    # Build ONCE
    train_ds, train_loader = prepLoader(
        **common_loader_kwargs,
        emitter_keep_by_mod=(train_keep_by_mod if any(emitter_counts_by_mod.values()) else None),
    )

    # Map mod string -> id (use dataset mapping if available)
    mod_to_id = getattr(train_ds, "class_to_id", None)

    if not mod_to_id:
        mod_to_id = {m: i for i, m in enumerate(ds_cfg.modulation_list)}

    export_loader_to_memmap_power_only(
        train_loader,
        out_dir=out_train,
        mod_to_id=mod_to_id,
        n_fft=64,
        win_len=64,
        hop_len=32,
    )

    print(f"\n✅ Export complete:")
    print(f"  {split} -> {out_train}")


if __name__ == "__main__":
    main()
