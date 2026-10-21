"""
AIMC-Spec HDF5 Subset Builder
=============================

This module provides utilities for reading, writing, and sub-sampling
radar emitter datasets stored in HDF5 format. It is primarily used to
construct a smaller training subset from a larger emitter dataset while
preserving the expected AIMC-Spec data schema.

This file is not intended to be executed directly. Instead, its
`main()` function is imported and invoked from the project-level
`main.py` entrypoint.

------------------------------------------------------------
Purpose
------------------------------------------------------------

Large AIMC-Spec datasets can contain tens of thousands of emitters per
modulation type. Training experiments often require a reduced subset
to speed up iteration while maintaining a consistent train/validation
ratio.

This script performs the following steps:

1. Reads emitter HDF5 files from a source directory.
2. Randomly samples emitters from each modulation class.
3. Writes a new HDF5 file containing only the sampled training emitters.
4. Preserves waveform data and metadata using the AIMC-Spec emitter schema.
5. Uses Blosc compression (via `hdf5plugin`) for efficient storage.

------------------------------------------------------------
Expected Input Directory Structure
------------------------------------------------------------

The source directory must contain one HDF5 file per modulation type:

    <all_tr_path>/
        unmodulated.h5
        lfm_up.h5
        lfm_down.h5
        ...
        p4.h5

Each file contains emitter groups:

    emitter_0/
        I                       (dataset, float32)
        Q                       (dataset, float32)
        attrs:
            seed
            rf
            pri
            pw
            pri_idx
            pw_idx
            intra_pulse_mod_bw
            intra_pulse_mod_info (optional)

------------------------------------------------------------
Output Directory Structure
------------------------------------------------------------

The subset directory will contain the same per-modulation files,
but with only the sampled training emitters:

    <subset_tr_path>/
        unmodulated.h5
        lfm_up.h5
        lfm_down.h5
        ...

Each file will contain `train_per_mod` emitters.

------------------------------------------------------------
Sampling Logic
------------------------------------------------------------

The subset size is derived from the desired train/validation ratio:

    ratio = train / (train + validation)

Given:
    validation_per_mod = test_per_mod

We compute:

    train_per_mod = round(validation_per_mod * ratio / (1 - ratio))

Example (default settings):

    ratio = 0.8
    test_per_mod = 1000

Result:

    train_per_mod = 4000
    total_needed_per_mod = 5000

Only the training subset is written by this script, but sampling is
performed in a way that preserves a disjoint validation split.

------------------------------------------------------------
Key Functions
------------------------------------------------------------

save_emitters_to_h5(...)
    Writes a list of emitter dictionaries to a compressed HDF5 file.

_read_emitter_group(...)
    Loads a single emitter group from an HDF5 file into a Python dict.

make_train_subset_from_iq_h5(...)
    Core function that samples emitters from each modulation file and
    writes the training subset.

main()
    Example entrypoint demonstrating how the subset builder is called.
    In the full project, this function is imported and executed from
    `main.py`.

------------------------------------------------------------
Dependencies
------------------------------------------------------------

Required packages:

    numpy
    h5py
    hdf5plugin
    
Blosc compression is enabled automatically when `hdf5plugin` is imported.

------------------------------------------------------------
Notes
------------------------------------------------------------

• Emitters are sampled without replacement using NumPy's random generator.
• The random seed ensures deterministic subset generation.
• Metadata fields are preserved exactly from the source dataset.
• `intra_pulse_mod_info` supports flexible storage (scalar, array, or JSON).

------------------------------------------------------------
Author Context
------------------------------------------------------------

This module is part of the AIMC-Spec dataset generation and processing
pipeline used for radar intra-pulse modulation classification research.
"""


# ─────────────────────────────────────────────
# Standard Library
# ─────────────────────────────────────────────
import json
from pathlib import Path
from typing import Any, Dict, List

# ─────────────────────────────────────────────
# Third-Party Libraries
# ─────────────────────────────────────────────
import h5py
import hdf5plugin  # activates Blosc compression/decompression
import matplotlib
import numpy as np

# Use non-GUI backend for matplotlib
matplotlib.use("Agg")

# Small helper to make dicts JSON-safe (handles NumPy scalars/arrays)
def _to_jsonable(obj):
    if isinstance(obj, (np.generic,)):  # NumPy scalar -> Python scalar
        return obj.item()
    if isinstance(obj, np.ndarray):  # ndarray -> list
        return obj.tolist()
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    return obj


def save_emitters_to_h5(emitters, filename):
    """
    Save a list of emitter dicts into one HDF5 file with Blosc compression.

    Expected emitter keys (new structure):
      - "I": list/array float32
      - "Q": list/array float32
      - "seed": int
      - "rf": float
      - "pri": float
      - "pw": float
      - "pri_idx": int
      - "pw_idx": int
      - "intra_pulse_mod_bw": float
      - "intra_pulse_mod_info": scalar | array-like | dict (stored flexibly)

    Removed from old structure:
      - signal_noise, time_offset, intra_pulse_mod_type, *_interpulse, pulses
    """
    print(f"Saving to: {filename}")

    # Blosc/LZ4 @ level 9
    _comp = dict(
        compression=32001, compression_opts=(0, 0, 0, 0, 9, 1, 1), shuffle=True
    )

    with h5py.File(filename, "w") as f:
        for idx, emitter in enumerate(emitters):
            g = f.create_group(f"emitter_{idx}")

            # --- I/Q waveforms ---
            I = np.asarray(emitter["I"], dtype=np.float32)
            Q = np.asarray(emitter["Q"], dtype=np.float32)

            if I.shape != Q.shape:
                raise ValueError(
                    f"Emitter {idx}: I and Q shape mismatch: {I.shape} vs {Q.shape}"
                )

            g.create_dataset("I", data=I, **_comp)
            g.create_dataset("Q", data=Q, **_comp)

            # --- Scalar metadata (always present in new schema) ---
            g.attrs["seed"] = int(emitter["seed"])
            g.attrs["rf"] = float(emitter["rf"])
            g.attrs["pri"] = float(emitter["pri"])
            g.attrs["pw"] = float(emitter["pw"])
            g.attrs["pri_idx"] = int(emitter["pri_idx"])
            g.attrs["pw_idx"] = int(emitter["pw_idx"])
            g.attrs["intra_pulse_mod_bw"] = float(emitter["intra_pulse_mod_bw"])

            # --- intra_pulse_mod_info: flexible storage ---
            # If it's dict -> JSON attribute
            # If it's array-like (and not tiny) -> dataset
            # If it's scalar / short list -> attribute
            info = emitter.get("intra_pulse_mod_info", None)

            if info is None:
                # Nothing to store; skip
                continue

            # Try to interpret as numeric array first
            stored = False
            try:
                arr = np.asarray(info)
                # If it looks numeric and has >1 element, store as dataset
                if arr.dtype.kind in "iufc" and arr.size > 1:
                    g.create_dataset(
                        "intra_pulse_mod_info",
                        data=arr.astype(np.float32, copy=False),
                        **_comp,
                    )
                    stored = True
                elif arr.dtype.kind in "iufc" and arr.size == 1:
                    # Single numeric -> attribute
                    g.attrs["intra_pulse_mod_info"] = float(arr.ravel()[0])
                    stored = True
            except Exception:
                pass

            if not stored:
                # Fallback: if dict/list/other, store JSON in attribute
                try:
                    g.attrs["intra_pulse_mod_info"] = json.dumps(_to_jsonable(info))
                except TypeError:
                    # Final fallback: string repr to avoid failing the whole save
                    g.attrs["intra_pulse_mod_info"] = str(info)


def _read_emitter_group(g: h5py.Group) -> Dict[str, Any]:
    """
    Load one emitter group from your schema into a dict compatible with save_emitters_to_h5().
    Expects datasets: I, Q
    Expects attrs: seed, rf, pri, pw, pri_idx, pw_idx, intra_pulse_mod_bw
    Optional: intra_pulse_mod_info as dataset or attr (string/number/JSON)
    """
    emitter: Dict[str, Any] = {}

    # Required waveforms
    emitter["I"] = np.asarray(g["I"], dtype=np.float32)
    emitter["Q"] = np.asarray(g["Q"], dtype=np.float32)

    # Required attrs
    for k in ("seed", "rf", "pri", "pw", "pri_idx", "pw_idx", "intra_pulse_mod_bw"):
        if k not in g.attrs:
            raise KeyError(f"Emitter group missing attr '{k}'")
        v = g.attrs[k]
        # Cast to native python types
        if k in ("seed", "pri_idx", "pw_idx"):
            emitter[k] = int(v)
        else:
            emitter[k] = float(v)

    # Optional intra_pulse_mod_info
    if "intra_pulse_mod_info" in g:
        emitter["intra_pulse_mod_info"] = np.asarray(g["intra_pulse_mod_info"])
    elif "intra_pulse_mod_info" in g.attrs:
        emitter["intra_pulse_mod_info"] = g.attrs["intra_pulse_mod_info"]

    return emitter


def _count_emitters_in_file(h5: h5py.File) -> int:
    # Your save code uses groups named emitter_<idx>
    # We'll count those safely rather than trusting contiguous numbering.
    return sum(1 for k in h5.keys() if k.startswith("emitter_"))


def make_train_subset_from_iq_h5(
    all_tr_path: str,
    subset_tr_path: str,
    *,
    modulation_list: List[str],
    seed: int,
    ratio: float = 0.8,
    test_per_mod: int = 1000,
    overwrite: bool = False,
) -> Dict[str, Dict[str, int]]:
    """
    Build a subset training set by sampling emitters from each <modulation>.h5.

    Input layout:
      all_tr_path/<mod>.h5
        emitter_0/
          I, Q datasets
          attrs: seed, rf, pri, pw, pri_idx, pw_idx, intra_pulse_mod_bw
          optional: intra_pulse_mod_info

    Output layout:
      subset_tr_path/<mod>.h5
        contains ONLY 'train_per_mod' emitters (sampled without replacement)

    Sampling math:
      val_per_mod = test_per_mod (fixed 1000 by default)
      total_needed = ceil(val_per_mod / (1-ratio))
      train_per_mod = total_needed - val_per_mod

    Returns:
      summary dict with counts per modulation.
    """
    if not (0.0 < ratio < 1.0):
        raise ValueError("ratio must be in (0,1)")
    if test_per_mod <= 0:
        raise ValueError("test_per_mod must be > 0")

    val_per_mod = int(test_per_mod)

    # More numerically stable
    train_per_mod = int(round(val_per_mod * ratio / (1.0 - ratio)))

    total_needed = train_per_mod + val_per_mod
    
    if train_per_mod <= 0:
        raise ValueError(
            f"Computed train_per_mod={train_per_mod} (ratio={ratio}, test_per_mod={test_per_mod}). "
            "Choose a larger ratio or test_per_mod."
        )

    in_dir = Path(all_tr_path)
    out_dir = Path(subset_tr_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(int(seed))

    summary: Dict[str, Dict[str, int]] = {
        "_meta": {
            "seed": int(seed),
            "ratio_times_1000": int(round(ratio * 1000)),
            "test_per_mod": int(test_per_mod),
            "val_per_mod": int(val_per_mod),
            "total_needed_per_mod": int(total_needed),
            "train_per_mod": int(train_per_mod),
        }
    }

    for mod in modulation_list:
        src_fp = in_dir / f"{mod}.h5"
        dst_fp = out_dir / f"{mod}.h5"

        if not src_fp.exists():
            # skip silently? I'd rather be explicit:
            raise FileNotFoundError(f"Missing source file: {src_fp}")

        if dst_fp.exists() and not overwrite:
            raise FileExistsError(f"Output file exists (set overwrite=True): {dst_fp}")

        with h5py.File(src_fp, "r") as src:
            n_emitters = _count_emitters_in_file(src)
            if n_emitters < total_needed:
                raise ValueError(
                    f"{src_fp}: has {n_emitters} emitters, but need at least {total_needed} "
                    f"(train {train_per_mod} + heldout {val_per_mod})"
                )

            # Choose a pool big enough to support the implied ratio.
            # We only SAVE the train subset here, but we sample a disjoint split:
            perm = rng.permutation(n_emitters)
            train_ids = perm[:train_per_mod]
            # heldout_ids = perm[train_per_mod:train_per_mod + val_per_mod]  # not written here

            # Load selected emitters
            emitters = []
            for idx in train_ids.tolist():
                gname = f"emitter_{idx}"
                if gname not in src:
                    # If numbering isn't contiguous, fall back to sorted keys:
                    # (rare, but safe)
                    emitter_keys = sorted(
                        [k for k in src.keys() if k.startswith("emitter_")]
                    )
                    if idx >= len(emitter_keys):
                        raise KeyError(
                            f"{src_fp}: emitter index out of range after fallback"
                        )
                    gname = emitter_keys[idx]
                emitters.append(_read_emitter_group(src[gname]))

        # Write subset file using YOUR saver (must be in scope / imported)
        save_emitters_to_h5(emitters, str(dst_fp))

        summary[mod] = {
            "source_emitters": int(n_emitters),
            "saved_train_emitters": int(train_per_mod),
            "implied_val_emitters": int(val_per_mod),
            "implied_total_needed": int(total_needed),
        }

    return summary


def main():
    split = (
        "train"  # or "test" if you wanted (but this function is for training subset)
    )

    modulation_list = [
        "unmodulated",
        "lfm_up",
        "lfm_down",
        "dlfm_up_down",
        "NLFM_up",
        "NLFM_down",
        "mlfm",
        "dlfm_down_up",
        "bpsk",
        "qpsk",
        "barker_11",
        "barker_13",
        "p1",
        "p2",
        "p3",
        "p4",
        "triangle",
        "exp",
        "bfsk",
        "4fsk",
        "eqfm",
        "sfm",
        "costas",
        "barker_2_1",
        "barker_2_2",
        "barker_3",
        "barker_4_1",
        "barker_4_2",
        "barker_5",
        "barker_7",
    ]

    # CHANGE THIS
    all_tr_path = rf"C:\Apps\Code\AIMC_Spec_v2_{split}_ALL"
    subset_tr_path = rf"C:\Apps\Code\AIMC_Spec_v2_{split}"

    summary = make_train_subset_from_iq_h5(
        all_tr_path,
        subset_tr_path,
        modulation_list=modulation_list,
        seed=42,
        ratio=0.8,
        test_per_mod=1000,
        overwrite=True,
    )

    print(summary["_meta"])
    # print per-mod counts if you want
    print({k: v["saved_train_emitters"] for k, v in summary.items() if k != "_meta"})


# if __name__ == "__main__":
#     main()
