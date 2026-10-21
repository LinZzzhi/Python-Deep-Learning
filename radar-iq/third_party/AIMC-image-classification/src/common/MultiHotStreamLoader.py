# --- Standard library ---
import itertools
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional
from time import time
import sys

# --- Third-party libraries ---
import h5py
import hdf5plugin  # activates Blosc compression/decompression
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from torch import Tensor

# Import here to avoid circulars if any
from src.common.spectrogram import (
    iq_to_power_spectrogram,
    iq_to_phase_spectrogram,
    spec_to_resnet_input,
)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# #### Data Classes
@dataclass(frozen=True)
class ModInst:
    mod: str
    emitter: int
    toa: int


@dataclass(frozen=True)
class WindowKey:
    split: str
    start: int
    end: int
    win_len: int
    mods: List[ModInst]
    snr: int

    def to_key(self) -> str:
        mods_sorted = sorted(self.mods, key=lambda m: (m.mod, m.emitter, m.toa))
        payload = {
            "split": self.split,
            "start": self.start,
            "end": self.end,
            "win": self.win_len,
            "mods": [asdict(m) for m in mods_sorted],
            "snr": self.snr,
        }
        return json.dumps(payload, separators=(",", ":"), sort_keys=True)

    @staticmethod
    def from_key(s: str) -> "WindowKey":
        o = json.loads(s)
        mods = [ModInst(**m) for m in o["mods"]]
        return WindowKey(
            split=o["split"],
            start=o["start"],
            end=o["end"],
            win_len=o["win"],
            mods=mods,
            snr=o["snr"],
        )


def add_noise(rng, real, imag=0, snr_db=0.0, noise_power_db=0.0, verify=False, eps=1e-12):
    """
    Robust AWGN adder with pulse-only power estimate.

    - Compute signal power ONLY over non-zero samples (|I|+|Q| > 0), so padding zeros
      don't taint the SNR. If there are no non-zero samples, fall back to noise-only.
    - `snr_db` is the desired SNR in dB relative to `noise_power_db`.
    - Returns noisy (real, imag) as float32.
    """
    real = np.asarray(real, dtype=np.float32)
    imag = np.asarray(imag, dtype=np.float32)
    signal = real.astype(np.float32) + 1j * imag.astype(np.float32)

    N = signal.size
    noise_lin = 10.0 ** (noise_power_db / 10.0)

    # Pulse-only mask: any non-zero I or Q counts as "signal present"
    nz = (np.abs(signal.real) > 0) | (np.abs(signal.imag) > 0)
    has_pulse = bool(nz.any())

    if not has_pulse:
        # ---- Pure-noise path (no scaling, ignore snr_db) ----
        noise = np.sqrt(noise_lin / 2.0) * (rng.normal(size=N) + 1j * rng.normal(size=N))
        noisy = noise
        if verify:
            act_noise_p = float(np.mean(np.abs(noise) ** 2))
            act_noise_db = 10.0 * np.log10(max(act_noise_p, eps))
            print(f"[noise-only] target_noise={noise_power_db:.2f} dB | actual_noise={act_noise_db:.2f} dB")
    else:
        # ---- Scale ONLY the pulse region to achieve target SNR ----
        sig_power = float(np.mean(np.abs(signal[nz]) ** 2))  # pulse-only average
        desired_sig_power = noise_lin * (10.0 ** (snr_db / 10.0))
        scale = np.sqrt(max(desired_sig_power, eps) / max(sig_power, eps))
        signal_scaled = signal * scale

        noise = np.sqrt(noise_lin / 2.0) * (rng.normal(size=N) + 1j * rng.normal(size=N))
        noisy = signal_scaled + noise

        if verify:
            act_sig_p = float(np.mean(np.abs(signal_scaled[nz]) ** 2))
            act_noise_p = float(np.mean(np.abs(noise) ** 2))
            act_snr_db = 10.0 * np.log10(max(act_sig_p, eps) / max(act_noise_p, eps))
            print(f"Desired SNR={snr_db:.2f} dB | Actual SNR={act_snr_db:.2f} dB")

    return np.real(noisy).astype(np.float32, copy=False), np.imag(noisy).astype(np.float32, copy=False)


# --- NEW: PRI-driven pulse placement over a continuous stream ---

def _pulse_spans_for_emitter(em: dict, total_samples: int, rng: np.random.Generator) -> list[tuple[int, int]]:
    """
    Returns a list of (p0, p1) sample indices in stream coordinates [0, total_samples),
    where p0 = toa + k*PRI and p1 = p0 + PW for all k s.t. p0 < total_samples.

    IMPORTANT CHANGE:
      - ToA is now drawn in [0, min(PRI, total_samples)) so the initial offset
        can span the *entire stream*, not just a single 20 µs window.
    """
    PW  = int(len(em["I"]))      # pulse width in *samples*
    PRI = int(em.get("pri", 0))  # PRI in *samples*

    if PRI <= 0:
        # Defensive: treat invalid PRI as non-repeating (push beyond stream)
        PRI = total_samples + 1

    # NEW: cap for ToA is min(PRI, total_samples)
    cap = min(PRI, total_samples//2)
    # Draw ToA uniformly in [0, cap); ensures at most one pulse in [0, ToA),
    # and aligns the initial offset to either a PRI-bound or stream-end bound.
    toa = int(rng.integers(0, max(1, cap)))

    spans: list[tuple[int, int]] = []
    p0 = toa
    while p0 < total_samples:
        p1 = p0 + PW
        spans.append((p0, p1))
        p0 += PRI
    return spans


def _build_repeating_stream_windows(
    mods: list[tuple[str, int]],                # [(mod_name, emitter_global_id), ...] length = K
    emitters: list[dict],
    *,
    snr: int,
    window_count: int,
    window_size: int,
    window_step_size: int,
    seed: int,
    split: str,
) -> list[WindowKey]:
    """
    Build a stream as a *continuous* signal of L = window_count * window_step_size samples.
    For each emitter, place pulses at toa + n*PRI across the full L; for each window, add
    a ModInst for *every* pulse that overlaps the window with ToA relative to window start.
    """
    rng = np.random.default_rng([seed, abs(snr), len(mods), window_count, window_size])
    L = window_count * window_step_size

    # Precompute global spans per emitter
    spans_by_em = {}
    for (mname, em_id) in mods:
        spans_by_em[em_id] = _pulse_spans_for_emitter(emitters[em_id], L, rng)

    stream: list[WindowKey] = []
    ws, we = 0, window_size
    for widx in range(window_count):
        insts: list[ModInst] = []
        for (mname, em_id) in mods:
            # add *all* pulses whose span intersects [ws, we)
            for (p0, p1) in spans_by_em[em_id]:
                if p1 <= ws or p0 >= we:
                    continue  # no overlap
                # ToA for this window is relative to ws; clip left side if needed
                toa_rel = max(0, p0 - ws)
                insts.append(ModInst(mname, em_id, int(toa_rel)))
        stream.append(WindowKey(
            split=split,
            start=ws, end=we, win_len=window_size,
            mods=insts, snr=int(snr),
        ))
        ws += window_step_size
        we += window_step_size
    return stream
####################


def extract_emitters_from_file(file_name: str, full_data: bool = False):
    emitters = []
    with h5py.File(file_name, "r") as f:
        emitter_keys = sorted(
            f.keys(),
            key=lambda x: int(x.split("_")[-1]) if x.startswith("emitter_") else x,
        )
        for emitter_key in emitter_keys:
            em = f[emitter_key]

            I = em["I"][:]
            Q = em["Q"][:]

            if full_data:
                emitters.append(
                    {
                        "I": I,
                        "Q": Q,
                        "seed": em.attrs["seed"],
                        "rf": em.attrs["rf"],
                        "pri": em.attrs["pri"],
                        "pri_idx": em.attrs["pri_idx"],
                        "pw": em.attrs["pw"],
                        "pw_idx": em.attrs["pw_idx"],
                        "intra_pulse_mod_bw": em.attrs["intra_pulse_mod_bw"],
                        "intra_pulse_mod_info": em.attrs["intra_pulse_mod_info"],
                    }
                )
            else:
                emitters.append(
                    {
                        "I": I,
                        "Q": Q,
                        "pri": em.attrs["pri_idx"],
                    }
                )

    return emitters


def extract_emitters_from_file_with_mod(
    file_path: str, mod_name: str, full_data: bool = False
):
    """
    Reads one <mod>.h5 and returns emitters with fields compatible with PulseTrainWindowDataset:
      - I, Q               : np.ndarray (the single stored pulse)
      - pri                : int (uses 'pri_idx' if present)
      - toa_idx            : int (defaults to 0 if missing)
      - mod                : str (modulation name)
      - signal_noise       : float (optional, defaults 0.0 if missing)
      - (optionally carries extra attrs if full_data=True)
    """
    out = []
    with h5py.File(file_path, "r") as f:
        # Keep canonical emitter order
        emitter_keys = sorted(
            f.keys(),
            key=lambda x: int(x.split("_")[-1]) if x.startswith("emitter_") else x,
        )
        for ek in emitter_keys:
            em = f[ek]
            I = em["I"][:]  # single pulse
            Q = em["Q"][:]

            # pull PRI and ToA (index form)
            pri = (
                int(em.attrs["pri_idx"])
                if "pri_idx" in em.attrs
                else int(em.attrs.get("pri", 0))
            )
            toa_idx = int(em.attrs.get("toa_idx", 0))
            sig_noise = float(em.attrs.get("signal_noise", 0.0))

            base = {
                "I": I,
                "Q": Q,
                "pri": pri,
                "toa_idx": toa_idx,
                "signal_noise": sig_noise,
                "mod": mod_name,
            }

            if full_data:
                base.update(
                    {
                        "seed": em.attrs.get("seed", None),
                        "rf": em.attrs.get("rf", None),
                        "pri_nominal": em.attrs.get("pri", None),
                        "pw": em.attrs.get("pw", None),
                        "pw_idx": em.attrs.get("pw_idx", None),
                        "intra_pulse_mod_bw": em.attrs.get("intra_pulse_mod_bw", None),
                        "intra_pulse_mod_info": em.attrs.get(
                            "intra_pulse_mod_info", None
                        ),
                    }
                )

            out.append(base)
    return out


def load_emitters_from_mod_list(
    filepath: str, modulation_list, *, full_data: bool = False
):
    """
    Coalesce emitters from multiple <mod>.h5 files into one list.

    Returns:
      emitters: list[dict]  (each has I, Q, pri, toa_idx, mod, signal_noise, ...)
      index_map: dict[mod -> (start_idx, end_idx)]   # handy for debugging/splits
    """
    emitters = []
    index_map = {}
    for mod in modulation_list:
        file_path = f"{filepath}{mod}.h5"  # adjust if your path scheme differs
        start = len(emitters)
        emitters.extend(
            extract_emitters_from_file_with_mod(file_path, mod, full_data=full_data)
        )
        end = len(emitters)
        index_map[mod] = (start, end)
    return emitters, index_map


def make_label_to_id(modulation_list, include_noise=True, start_at=0):
    """
    Map labels to ints; 'noise' is 0 by default, then mods follow.
    """
    mapping = {}
    cur = start_at
    if include_noise:
        mapping["noise"] = cur
        cur += 1
    for m in modulation_list:
        mapping[m] = cur
        cur += 1
    return mapping


# #### Helpers


def estimate_pos_weight(
    dataset,
    *,
    num_samples: int | None = None,
    batch_size: int = 256,
    num_workers: int = 2,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """
    Estimate BCEWithLogitsLoss `pos_weight` for a multi-label dataset.

    pos_weight[c] = (#negatives for class c) / (#positives for class c)

    Parameters
    ----------
    dataset : torch.utils.data.Dataset
        Must return (xs, ys, yend, msk, meta) where yend is (B,C) last-step label.
    num_samples : int | None
        If given, randomly sample at most this many items; else scan whole dataset.
    batch_size : int
        Mini-batch size for the temporary DataLoader.
    num_workers : int
        Workers for the temporary DataLoader.
    device : str or torch.device
        Where to accumulate counts (keep "cpu" if dataset is large).

    Returns
    -------
    torch.Tensor of shape (C,)  to pass to `BCEWithLogitsLoss(pos_weight=...)`.
    """
    loader = DataLoader(
        dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False
    )

    total_pos = None
    total_count = 0
    sampled = 0

    for _, _, yend, _, _ in loader:
        y = yend.to(device, non_blocking=True)
        if total_pos is None:
            total_pos = torch.zeros(y.size(1), device=device, dtype=torch.float64)
        total_pos += y.sum(dim=0).double()
        total_count += y.size(0)
        sampled += y.size(0)
        if num_samples is not None and sampled >= num_samples:
            break

    total_neg = total_count - total_pos
    # avoid division by zero
    pos_weight = (total_neg / torch.clamp(total_pos, min=1.0)).float()
    return pos_weight


def _global_idx_for_mod(
    mod: str, base_eid: int, index_map: Dict[str, Tuple[int, int]]
) -> int:
    """
    Map a per-mod base emitter id (0..E-1) to the global emitter index in `emitters`.
    `index_map[mod] == (start, end)` is the slice for that modulation in the flat emitters list.
    """
    s, e = index_map[mod]
    E = e - s
    if not (0 <= base_eid < E):
        raise IndexError(
            f"base emitter id {base_eid} out of range for mod '{mod}' (E={E})"
        )
    return s + base_eid


def list_unique_classes(emitters: List[dict]) -> Tuple[List[str], Dict[str, int]]:
    mods = sorted({em["mod"] for em in emitters})
    class_to_id = {m: i for i, m in enumerate(mods)}
    return mods, class_to_id


def build_emitter_arrays(emitters: List[dict]):
    E = len(emitters)
    pw = np.array([len(em["I"]) for em in emitters], dtype=np.int32)
    pri = np.array([int(em["pri"]) for em in emitters], dtype=np.int32)
    return pw, pri


def rand_toa(max_toa_inclusive: int, rng: np.random.Generator) -> int:
    return int(rng.integers(0, max_toa_inclusive + 1)) if max_toa_inclusive >= 0 else 0


def _downsample_partition_uniform(
    part_map: dict, combos_target: int, rng: np.random.Generator
):
    """
    part_map: dict[perm -> list[emitter-combos]]
      - K2: perm=(A,B), combos = [(emA, emB), ...]
      - K3: perm=(A,B,C), combos = [(emA, emB, emC), ...]
    combos_target: total number of combos to keep across ALL permutations.
    Returns a new dict with same keys but truncated lists.
    """
    if combos_target <= 0:
        return {k: [] for k in part_map.keys()}

    perms = sorted(part_map.keys())
    total_available = sum(len(v) for v in part_map.values())
    if combos_target >= total_available:
        # nothing to cull
        return part_map

    # split target as evenly as possible across permutations
    base = combos_target // len(perms)
    extra = combos_target % len(perms)

    kept = {}
    for i, p in enumerate(perms):
        want = base + (1 if i < extra else 0)
        avail = part_map[p]
        # shuffle once per perm then take the slice
        arr = avail.copy()
        rng.shuffle(arr)
        kept[p] = arr[: min(want, len(arr))]
    return kept


def class_map_from_emitters(
    emitters, add_noise_class: bool = False, noise_label: str = "__noise__"
):
    mods = sorted({em["mod"] for em in emitters})
    if add_noise_class and noise_label not in mods:
        mods.append(noise_label)  # put noise at the end
    class_to_id = {m: i for i, m in enumerate(mods)}
    id_to_class = {i: m for m, i in class_to_id.items()}
    noise_id = class_to_id[noise_label] if add_noise_class else None
    return class_to_id, id_to_class, noise_id


def _labels_from_key_time_threshold(
    w: WindowKey,
    emitters: List[dict],
    class_to_id: Dict[str, int],
    threshold_time_frac: float = 0.10,
    *,
    add_noise_class: bool = False,
    noise_class_id: int | None = None,
) -> np.ndarray:
    W = w.win_len
    need = int(math.ceil(W * threshold_time_frac))
    C = len(class_to_id)
    y = np.zeros(C, dtype=np.float32)

    for m in w.mods:
        em = emitters[m.emitter]
        c = class_to_id.get(em["mod"])
        if c is None:
            continue
        s = int(m.toa)
        e = min(W, s + len(em["I"]))
        ov = max(0, e - s)
        if ov >= need:
            y[c] = 1.0

    if add_noise_class and (y.sum() == 0.0) and noise_class_id is not None:
        y[noise_class_id] = 1.0

    return y


def build_k1_streams_all_emitters(
    emitters,
    modulation_list,
    snr_range,
    window_count,
    window_size,
    window_step_size,
    split,
    index_map,  # (kept)
    seed=1234,
):
    k1 = {}
    for snr in snr_range:
        for mod in modulation_list:
            s, e = index_map[mod]
            for em in range(s, e):
                mods = [(mod, em)]  # K=1
                stream = _build_repeating_stream_windows(
                    mods, emitters,
                    snr=snr,
                    window_count=window_count,
                    window_size=window_size,
                    window_step_size=window_step_size,
                    seed=seed,
                    split=split,
                )
                k1[(mod, em, int(snr))] = stream
    return k1

def _compute_k_targets(
    n_k1_streams: int, ratio: dict, snr_count: int, loose_round_up=True
):
    """
    Returns desired counts as (#streams for K=2, #streams for K=3)
    and also as (#emitter pairs, #emitter triplets) so we can cull partitions
    before building. Each emitter-combo → |snr_range| streams (one per SNR).
    """
    # total streams implied by K1 and ratio
    total_streams_target = int(
        round(n_k1_streams / ratio.get(1, 1.0) * sum(ratio.values()))
    )
    k2_streams_target = total_streams_target * ratio.get(2, 0.0)
    k3_streams_target = total_streams_target * ratio.get(3, 0.0)

    if loose_round_up:
        k2_streams_target = int(math.ceil(k2_streams_target))
        k3_streams_target = int(math.ceil(k3_streams_target))
    else:
        k2_streams_target = int(math.floor(k2_streams_target))
        k3_streams_target = int(math.floor(k3_streams_target))

    # each emitter pair/triplet becomes |snr_range| streams (one stream per SNR)
    # convert to emitter-combo counts (round, keep >=0)
    def to_combos(streams_target):
        if snr_count <= 0:
            return 0
        # prefer rounding up to avoid starving rare perms
        return max(0, int(math.ceil(streams_target / snr_count)))

    k2_pairs_target = to_combos(k2_streams_target)
    k3_triplets_target = to_combos(k3_streams_target)
    return (k2_streams_target, k3_streams_target, k2_pairs_target, k3_triplets_target)


def partition_emitters_k2_pairs(
    num_emitters: int,
    class_pairs_ordered: List[Tuple[str, str]],
    seed: int = 1234,
) -> Dict[Tuple[str, str], List[Tuple[int, int]]]:
    """
    For each ordered pair (A,B), split range(num_emitters) into two disjoint halves and
    zip them to make emitter pairs. No emitter repeats inside the pair.
    Returns mapping (A,B) -> list of (emA, emB).
    """
    rng = np.random.default_rng(seed)
    pairs_map = {}
    base_emitters = list(range(num_emitters))

    for A, B in class_pairs_ordered:
        idxs = base_emitters.copy()
        rng.shuffle(idxs)
        half = len(idxs) // 2
        left = idxs[:half]  # for A
        right = idxs[half:]  # for B (may be +1 if odd)
        # to equalise lengths, allow the shorter side to repeat last element once
        if len(left) < len(right):
            left = left + [left[-1]]
        elif len(right) < len(left):
            right = right + [right[-1]]
        pairs_map[(A, B)] = list(zip(left, right))
    return pairs_map


def global_partition_k2(num_emitters: int, class_pairs_ordered, seed: int = 42):
    rng = np.random.default_rng(seed)
    S = np.arange(num_emitters)
    rng.shuffle(S)
    half = len(S) // 2
    left = S[:half].tolist()
    right = S[half:].tolist()
    # pad shorter side by repeating last once (when odd)
    if len(left) < len(right):
        left = left + [left[-1]]
    if len(right) < len(left):
        right = right + [right[-1]]
    # return per-permutation positional lists (aligned by index)
    pairs_map = {}
    for A, B in class_pairs_ordered:
        pairs_map[(A, B)] = list(zip(left, right))  # same positions for all perms
    return pairs_map  # each list length = n_pos2_total_raw


def partition_emitters_k3_triplets(
    num_emitters: int,
    class_triplets_ordered: List[Tuple[str, str, str]],
    seed: int = 1234,
) -> Dict[Tuple[str, str, str], List[Tuple[int, int, int]]]:
    rng = np.random.default_rng(seed)
    trip_map = {}
    base_emitters = list(range(num_emitters))

    a_len = math.ceil(num_emitters / 3.0)
    b_len = num_emitters - a_len
    b_each = b_len // 2
    c_len = b_len - b_each  # ensures a_len + b_each + c_len == num_emitters

    for A, B, C in class_triplets_ordered:
        idxs = base_emitters.copy()
        rng.shuffle(idxs)
        A_list = idxs[:a_len]
        rest = idxs[a_len:]
        B_list = rest[:b_each]
        C_list = rest[b_each:]

        # bump B and C to a_len by **repeating one emitter once** if needed
        if len(B_list) < a_len:
            B_list = B_list + [B_list[-1]] * (a_len - len(B_list))
        if len(C_list) < a_len:
            C_list = C_list + [C_list[-1]] * (a_len - len(C_list))

        trip_map[(A, B, C)] = list(zip(A_list, B_list, C_list))
    return trip_map


def global_partition_k3(num_emitters: int, class_triplets_ordered, seed: int = 42):
    rng = np.random.default_rng(seed)
    S = np.arange(num_emitters)
    rng.shuffle(S)
    a_len = math.ceil(num_emitters / 3.0)
    b_len = num_emitters - a_len
    b_each = b_len // 2
    c_len = b_len - b_each
    A = S[:a_len].tolist()
    rest = S[a_len:].tolist()
    B = rest[:b_each]
    C = rest[b_each:]
    # pad B,C up to len(A) by repeating last (exactly what you asked)
    if len(B) < len(A):
        B = B + [B[-1]] * (len(A) - len(B))
    if len(C) < len(A):
        C = C + [C[-1]] * (len(A) - len(C))
    trip_map = {}
    for perm in class_triplets_ordered:
        trip_map[perm] = list(zip(A, B, C))  # same positions for all perms
    return trip_map  # each list length = n_pos3_total_raw


def build_streams_k2_from_pairs(
    pairs_map, emitters, snr_range,
    window_count, window_size, window_step_size,
    split, seed=1234,
) -> List[List[WindowKey]]:
    k2_streams: List[List[WindowKey]] = []
    for snr in snr_range:
        for (A, B), pairs in pairs_map.items():
            for emA, emB in pairs:
                mods = [(A, emA), (B, emB)]
                stream = _build_repeating_stream_windows(
                    mods, emitters,
                    snr=snr,
                    window_count=window_count,
                    window_size=window_size,
                    window_step_size=window_step_size,
                    seed=seed, split=split
                )
                k2_streams.append(stream)
    return k2_streams


def build_streams_k3_from_triplets(
    trip_map, emitters, snr_range,
    window_count, window_size, window_step_size,
    split, seed=1234,
) -> List[List[WindowKey]]:
    k3_streams: List[List[WindowKey]] = []
    for snr in snr_range:
        for (A, B, C), trips in trip_map.items():
            for emA, emB, emC in trips:
                mods = [(A, emA), (B, emB), (C, emC)]
                stream = _build_repeating_stream_windows(
                    mods, emitters,
                    snr=snr,
                    window_count=window_count,
                    window_size=window_size,
                    window_step_size=window_step_size,
                    seed=seed, split=split
                )
                k3_streams.append(stream)
    return k3_streams


def build_k2_from_positions(
    pairs_map, n_pos_keep, emitters, snr_range,
    window_count, window_size, window_step_size,
    split, index_map, seed=42,
):
    kept_streams = []
    for (A, B), lst in pairs_map.items():
        kept = lst[:n_pos_keep]
        for snr in snr_range:
            for baseA, baseB in kept:
                emA = _global_idx_for_mod(A, baseA, index_map)
                emB = _global_idx_for_mod(B, baseB, index_map)
                mods = [(A, emA), (B, emB)]
                stream = _build_repeating_stream_windows(
                    mods, emitters,
                    snr=snr,
                    window_count=window_count,
                    window_size=window_size,
                    window_step_size=window_step_size,
                    seed=seed, split=split
                )
                kept_streams.append(stream)
    return kept_streams


def build_k3_from_positions(
    trip_map, n_pos_keep, emitters, snr_range,
    window_count, window_size, window_step_size,
    split, index_map, seed=42,
):
    kept_streams = []
    for (A, B, C), lst in trip_map.items():
        kept = lst[:n_pos_keep]
        for snr in snr_range:
            for baseA, baseB, baseC in kept:
                emA = _global_idx_for_mod(A, baseA, index_map)
                emB = _global_idx_for_mod(B, baseB, index_map)
                emC = _global_idx_for_mod(C, baseC, index_map)
                mods = [(A, emA), (B, emB), (C, emC)]
                stream = _build_repeating_stream_windows(
                    mods, emitters,
                    snr=snr,
                    window_count=window_count,
                    window_size=window_size,
                    window_step_size=window_step_size,
                    seed=seed, split=split
                )
                kept_streams.append(stream)
    return kept_streams


def compute_targets_positions(
    n_k1_streams: int,
    ratio: dict,
    snr_count: int,
    n_pairs_per_perm_raw: int,
    n_trips_per_perm_raw: int,
    loose_round_up=True,
):
    # same math as before but in streams:
    total_streams_target = int(
        round(n_k1_streams / ratio.get(1, 1.0) * sum(ratio.values()))
    )
    k2_streams_target = total_streams_target * ratio.get(2, 0.0)
    k3_streams_target = total_streams_target * ratio.get(3, 0.0)
    if loose_round_up:
        k2_streams_target = int(math.ceil(k2_streams_target))
        k3_streams_target = int(math.ceil(k3_streams_target))
    else:
        k2_streams_target = int(math.floor(k2_streams_target))
        k3_streams_target = int(math.floor(k3_streams_target))

    # convert streams -> positions to keep (per permutation), synchronized
    # streams = positions * (#permutations) * snr_count
    n_perm2 = n_pairs_per_perm_raw["num_perms"]  # e.g., 12 for 4 classes
    n_perm3 = n_trips_per_perm_raw["num_perms"]  # e.g., 24 for 4 classes

    def to_positions(streams_target, per_perm_count_raw, num_perms):
        # desired positions per permutation = ceil(streams_target / (num_perms * snr))
        if snr_count == 0 or num_perms == 0:
            return 0
        n_pos = int(math.ceil(streams_target / (num_perms * snr_count)))
        # and cap by what we actually have
        return max(0, min(n_pos, per_perm_count_raw["len_per_perm"]))

    n_pos2 = to_positions(k2_streams_target, n_pairs_per_perm_raw, n_perm2)
    n_pos3 = to_positions(k3_streams_target, n_trips_per_perm_raw, n_perm3)

    return n_pos2, n_pos3, k2_streams_target, k3_streams_target


def cull_streams_to_ratio(
    k1_streams: Dict[Tuple[str, int, int], List[WindowKey]],
    k2_streams: List[List[WindowKey]],
    k3_streams: List[List[WindowKey]],
    modulation_list: List[str],
    ratio: Dict[int, float],  # e.g., {1:0.75, 2:0.20, 3:0.05}
    loose_round_up: bool = True,
    seed: int = 1234,
) -> Dict[int, List[List[WindowKey]]]:
    """
    Keep all K=1 (or cull if you want), and downsample K=2 and K=3 uniformly per permutation.
    """
    rng = np.random.default_rng(seed)
    # Current counts
    n1 = len(k1_streams)
    # Desired totals based on K=1 as anchor
    target_total = int(round(n1 / ratio.get(1, 1.0) * sum(ratio.values())))
    # derive desired K2 and K3 counts
    desired_k2 = (
        int(math.floor(target_total * ratio.get(2, 0.0)))
        if not loose_round_up
        else int(math.ceil(target_total * ratio.get(2, 0.0)))
    )
    desired_k3 = (
        int(math.floor(target_total * ratio.get(3, 0.0)))
        if not loose_round_up
        else int(math.ceil(target_total * ratio.get(3, 0.0)))
    )

    # Helper: split K2/K3 streams by their ordered permutation key
    def perm_key(stream: List[WindowKey]) -> Tuple[str, ...]:
        mods = tuple(m.mod for m in stream[0].mods)  # ordered
        return mods

    def downsample_uniform(
        streams: List[List[WindowKey]], target: int
    ) -> List[List[WindowKey]]:
        if target >= len(streams):
            return streams
        # bucket by permutation
        buckets: Dict[Tuple[str, ...], List[List[WindowKey]]] = {}
        for st in streams:
            buckets.setdefault(perm_key(st), []).append(st)
        # how many to keep per perm (uniform; round up)
        perms = sorted(buckets.keys())
        keep_per_perm = {}
        total_kept = 0
        remaining = target
        # first pass: equal share floor
        base = target // len(perms)
        extra = target % len(perms)
        for i, p in enumerate(perms):
            k = min(base + (1 if i < extra else 0), len(buckets[p]))
            keep_per_perm[p] = k
            total_kept += k
        # sample within each perm
        kept = []
        for p, lst in buckets.items():
            rng.shuffle(lst)
            kept.extend(lst[: keep_per_perm[p]])
        return kept

    k2_kept = downsample_uniform(k2_streams, desired_k2)
    k3_kept = downsample_uniform(k3_streams, desired_k3)

    return {1: list(k1_streams.values()), 2: k2_kept, 3: k3_kept}


def build_dataset_with_pre_cull(
    emitters: List[dict],
    modulation_list: List[str],
    snr_range: List[int],
    *,
    window_count: int,
    window_size: int,
    window_step_size: int,
    split: str,
    k_ratio: Dict[int, float],  # e.g., {1:0.75, 2:0.20, 3:0.05}
    seed: int = 42,
):
    """
    1) Build K=1 (all streams) -> n_k1_streams
    2) Partition K=2/K=3 emitter combos
    3) Cull partitions to target counts (in combos) BEFORE building
    4) Build only kept combos -> streams (|snr_range| per combo)
    5) Return k_streams dict
    """
    rng = np.random.default_rng(seed)
    num_emitters = len(emitters)

    # -- K=1: build all (mod × all emitters × SNR)
    k1_streams = build_k1_streams_all_emitters(
        emitters,
        modulation_list,
        snr_range,
        window_count,
        window_size,
        window_step_size,
        split,
        seed=seed,
    )
    n_k1_streams = len(k1_streams)

    # -- compute targets based on ratio
    k2_streams_target, k3_streams_target, k2_pairs_target, k3_triplets_target = (
        _compute_k_targets(
            n_k1_streams, k_ratio, snr_count=len(snr_range), loose_round_up=True
        )
    )

    # -- K=2: partition → cull → build
    class_pairs_ordered = list(itertools.permutations(modulation_list, 2))
    pairs_map_full = partition_emitters_k2_pairs(
        num_emitters, class_pairs_ordered, seed=seed
    )
    pairs_map_kept = _downsample_partition_uniform(pairs_map_full, k2_pairs_target, rng)
    k2_streams = build_streams_k2_from_pairs(
        pairs_map_kept,
        emitters,
        snr_range,
        window_count,
        window_size,
        window_step_size,
        split,
        seed=seed,
    )

    # -- K=3: partition → cull → build
    class_triplets_ordered = list(itertools.permutations(modulation_list, 3))
    trip_map_full = partition_emitters_k3_triplets(
        num_emitters, class_triplets_ordered, seed=seed
    )
    trip_map_kept = _downsample_partition_uniform(
        trip_map_full, k3_triplets_target, rng
    )
    k3_streams = build_streams_k3_from_triplets(
        trip_map_kept,
        emitters,
        snr_range,
        window_count,
        window_size,
        window_step_size,
        split,
        seed=seed,
    )

    # pack
    k_streams = {1: list(k1_streams.values()), 2: k2_streams, 3: k3_streams}
    return k_streams


def flatten_k_streams(k_streams: Dict[int, List[List[WindowKey]]]) -> List[str]:
    keys = []
    for K, streams in k_streams.items():
        for stream in streams:
            keys.extend(w.to_key() for w in stream)
    return keys


def sanity_counts_text(k_streams: Dict[int, List[List[WindowKey]]], window_count: int):
    print("=" * 70)
    print("DATASET SUMMARY")
    print("=" * 70)
    total_streams = sum(len(v) for v in k_streams.values())
    total_windows = sum(len(st) for v in k_streams.values() for st in v)
    print(f"Total streams: {total_streams:,}")
    print(f"Total windows: {total_windows:,}")
    for K in sorted(k_streams.keys()):
        S = len(k_streams[K])
        W = S * window_count
        print(f"  K={K}: streams={S:,}  windows={W:,}")


def combination_sanity_report(
    k_streams: Dict[int, List[List[WindowKey]]],
    *,
    snr_range: List[int],
    window_count: int,
    show_examples_per_comb: int = 0,  # set >0 to print a few example emitter tuples
):
    """
    Prints:
      • totals per K (streams & windows)
      • for each K:
          - number of class combinations observed
          - per-combination stream counts (and divisibility by |SNR|)
          - per-combination UNIQUE emitter-combination count (ignoring SNR)
          - optional sample emitter tuples

    Assumptions:
      • Each 'stream' is a list[WindowKey].
      • All windows in a stream share the same mods/SNR (only start/end slide).
      • Order of mods in a stream encodes the class combination (K=1: (A,), K=2: (A,B), K=3: (A,B,C)).
    """
    S = len(snr_range)

    print("=" * 78)
    print("COMBINATION / EMITTER COMBINATION REPORT")
    print("=" * 78)

    total_streams_all = sum(len(v) for v in k_streams.values())
    total_windows_all = sum(
        len(st) * window_count for v in k_streams.values() for st in v
    )
    print(f"Total streams: {total_streams_all:,}")
    print(f"Total windows: {total_windows_all:,}")
    print(f"SNR values: {snr_range} (|SNR|={S})")
    print("-" * 78)

    for K in sorted(k_streams.keys()):
        streams = k_streams[K]
        print(f"\nK={K}")
        print("-" * 78)
        if not streams:
            print("  (no streams)")
            continue

        # per-combination buckets
        comb_to_streams = defaultdict(list)  # comb(tuple[str]) -> list of streams
        comb_to_emitter_tuples = defaultdict(
            set
        )  # comb -> set of emitter tuples (ignore SNR)
        comb_to_snr_counts = defaultdict(Counter)  # comb -> Counter{snr: #streams}

        for st in streams:
            first = st[0]
            # ordered class combination for this stream
            comb = tuple(m.mod for m in first.mods)
            # ordered emitter tuple
            emit_tuple = tuple(m.emitter for m in first.mods)
            snr = int(first.snr)

            comb_to_streams[comb].append(st)
            comb_to_emitter_tuples[comb].add(emit_tuple)
            comb_to_snr_counts[comb][snr] += 1

        # summary
        combs_sorted = sorted(comb_to_streams.keys())
        print(f"  Observed class combinations: {len(combs_sorted)}")
        total_streams_k = len(streams)
        total_windows_k = total_streams_k * window_count
        print(f"  Streams: {total_streams_k:,}  Windows: {total_windows_k:,}")

        # header
        print("  " + "-" * 74)
        print(
            "  {:<28} | {:>9} | {:>11} | {:>10} | {}".format(
                "combination (ordered)",
                "#streams",
                "#emitters",
                "streams%SNR",
                "per-SNR counts",
            )
        )
        print("  " + "-" * 74)

        # rows
        bad_divisible = 0
        for comb in combs_sorted:
            n_streams = len(comb_to_streams[comb])
            n_emitters_unique = len(comb_to_emitter_tuples[comb])
            divisible = n_streams % S == 0
            if not divisible:
                bad_divisible += 1
            per_snr = comb_to_snr_counts[comb]
            per_snr_str = ", ".join(
                f"{snr}:{cnt}" for snr, cnt in sorted(per_snr.items())
            )
            print(
                "  {:<28} | {:>9} | {:>11} | {:>10} | {}".format(
                    " × ".join(comb),
                    f"{n_streams:,}",
                    f"{n_emitters_unique:,}",
                    "OK" if divisible else "NO",
                    per_snr_str,
                )
            )

            # optional: show a few emitter tuples for the comb
            if show_examples_per_comb > 0:
                ex = list(comb_to_emitter_tuples[comb])[:show_examples_per_comb]
                print("    examples:", ", ".join(str(t) for t in ex))

        print("  " + "-" * 74)
        if bad_divisible == 0:
            print(
                f"  All combinations have stream counts divisible by |SNR|={S} (good)."
            )
        else:
            print(
                f"  WARNING: {bad_divisible} combination(s) not divisible by |SNR|={S}."
            )

        # totals per K double-checks
        sum_emitters = sum(len(comb_to_emitter_tuples[p]) for p in combs_sorted)
        sum_streams = sum(len(comb_to_streams[p]) for p in combs_sorted)
        print(
            f"  Totals check → unique emitter-combinations across combs: {sum_emitters:,}"
        )
        print(
            f"                 streams across combs: {sum_streams:,}  (should equal above Streams)"
        )
        print("-" * 78)


def _mix_window_from_key(
    w: WindowKey,
    emitters: List[dict],
) -> np.ndarray:
    """
    Sum pulses for all mods in the key inside [0, win_len).
    Returns complex64 array of shape (win_len,).
    Assumes a single stored base pulse per emitter, placed at ToA (no repetition inside window).
    """
    W = w.win_len
    out = np.zeros(W, dtype=np.complex64)
    for m in w.mods:
        em = emitters[m.emitter]
        pulse = (em["I"].astype(np.float32) + 1j * em["Q"].astype(np.float32)).astype(
            np.complex64, copy=False
        )
        pw = len(pulse)
        s = int(m.toa)
        e = min(W, s + pw)
        if e > s:
            out[s:e] += pulse[: (e - s)]
    return out


def _add_awgn_to_snr(
    xc: np.ndarray, snr_db: float, rng: np.random.Generator
) -> np.ndarray:
    """
    xc: complex64 signal (length W)
    Target SNR in dB relative to signal power over the window.
    """
    if np.all(xc == 0):
        # pure noise window: just unit-variance noise
        noise = (
            rng.normal(size=xc.shape).astype(np.float32)
            + 1j * rng.normal(size=xc.shape).astype(np.float32)
        ) / np.sqrt(2.0)
        return noise.astype(np.complex64)
    sig_pow = float(np.mean(np.abs(xc) ** 2))
    snr_lin = 10.0 ** (snr_db / 10.0)
    noise_var = sig_pow / snr_lin
    noise_std = math.sqrt(noise_var)
    noise = (
        rng.normal(scale=noise_std, size=xc.shape).astype(np.float32)
        + 1j * rng.normal(scale=noise_std, size=xc.shape).astype(np.float32)
    ) / np.sqrt(2.0)
    return xc + noise.astype(np.complex64)


class MultiHotStreamSegmentDataset(Dataset):
    """
    Returns fixed-length sequences (segments) from streams to drive a temporal model.
    Each item is a segment that ENDS at (stream_id, end_idx) and includes up to T-1
    previous windows from the same stream (left-padded if not enough history).

    Output:
      x_seq  : (T, 2, W)  float32   [I,Q]
      y_seq  : (T, C)     float32   multi-hot per timestep (for optional aux loss)
      y_last : (C,)       float32   label for the last window only (main target)
      mask   : (T,)       bool      1 for real frames, 0 for padded frames
      meta   : dict       { 'stream_id', 'end_idx', 'K', 'snr', 'perm': tuple[str], ... }
    """

    def __init__(
        self,
        k_streams: Dict[int, List[List[WindowKey]]],
        emitters: List[dict],
        *,
        segment_len: int,  # T
        end_stride: int = 1,  # step between valid end positions inside each stream
        threshold_time_frac: float = 0.10,
        seed: int = 1234,
        noise_fn: callable = None,  # <--- NEW
        noise_power_db: float = 0.0,  # <--- NEW, passed to noise_fn
        add_noise_class: bool = True,  # <--- NEW
        noise_label: str = "__noise__",  # <--- NEW
        noise_power_db_range: tuple[float,float] | None = None,
        balance_mode: str | None = None,  # None | "global" | "per_stream"
        noise_pos_ratio: float = 1.0,  # noise : positive target ratio (e.g., 1.0 = 50/50)
        resample_seed: int | None = None,  # optional extra seed for balancing draw
        mode: str = "sequence",                 # "sequence" | "resnet"
        drop_ambiguous: bool = False,           # skip non-1hot end labels for CE
        snr_levels: tuple[int, ...] | None = None,  # e.g., (5, 0, -5)
        balance_snr: bool = False,              # uniform across snr_levels
    ):
        assert segment_len >= 1
        self.emitters = emitters
        self.threshold_time_frac = float(threshold_time_frac)
        self.seed = int(seed)
        self.segment_len = int(segment_len)
        self.end_stride = int(end_stride)

        self.balance_mode = balance_mode
        self.noise_pos_ratio = float(noise_pos_ratio)
        self.resample_seed = resample_seed

        self.noise_power_db_range = noise_power_db_range   # e.g., (-20.0, +5.0)

        self.mode = str(mode)
        assert self.mode in ("sequence", "resnet")  # NEW
        self.drop_ambiguous = bool(drop_ambiguous)  # NEW
        self.req_snr_levels = tuple(snr_levels) if snr_levels is not None else None  # NEW
        self.balance_snr = bool(balance_snr)       # NEW

        # Flatten streams and precompute (stream_id, end_idx) positions
        self.streams: List[List[WindowKey]] = []
        self.stream_k: List[int] = []  # K value per stream (len(mods) in first window)
        self.stream_perm: List[Tuple[str, ...]] = []
        self.stream_snr: List[int] = []
        self.index: List[Tuple[int, int]] = []  # (stream_id, end_idx)
        raw_indices: list[tuple[int, int]] = []  # (sid, end_idx)

        self.noise_fn = noise_fn
        self.noise_power_db = float(noise_power_db)

        self.add_noise_class = bool(add_noise_class)
        self.noise_label = str(noise_label)

        # build class map with optional noise class
        self.class_to_id, self.id_to_class, self.noise_class_id = (
            class_map_from_emitters(
                emitters,
                add_noise_class=self.add_noise_class,
                noise_label=self.noise_label,
            )
        )
        self.C = len(self.class_to_id)

        # Build stream lists and indexing
        for K, streams in k_streams.items():
            for st in streams:
                if not st:
                    continue
                sid = len(self.streams)
                self.streams.append(st)
                self.stream_k.append(len(st[0].mods))
                self.stream_perm.append(tuple(m.mod for m in st[0].mods))
                self.stream_snr.append(int(st[0].snr))
                L = len(st)
                for end_idx in range(0, L, self.end_stride):
                    raw_indices.append((sid, end_idx))

        # Cache window length
        self.W = self.streams[0][0].win_len if self.streams else 0

        # Build balanced index
        self.index = self._build_balanced_index(raw_indices)

        self.class_counts_last = None  # torch.Tensor(C,)
        self.total_last = 0

        # --- NEW: per-stream noise floor reference ---
        # We want a stable noise environment per stream. We approximate it by:
        #   - taking the first window in the stream that actually has a pulse
        #   - using that window's declared SNR as the "stream SNR"
        # If a stream is pure noise (no mods anywhere), fall back to 0 dB.
        self.stream_noise_db: dict[int, float] = {}
        for sid, st in enumerate(self.streams):
            ref_snr = None
            for w in st:
                if len(w.mods) > 0:
                    ref_snr = float(w.snr)
                    break
            if ref_snr is None:
                ref_snr = 0.0  # pure-noise stream fallback
            self.stream_noise_db[sid] = ref_snr


    def __len__(self) -> int:
        return len(self.index)

    def _label_last(self, sid: int, end_idx: int) -> np.ndarray:
        return _labels_from_key_time_threshold(
            self.streams[sid][end_idx],
            self.emitters,
            self.class_to_id,
            self.threshold_time_frac,
            add_noise_class=getattr(self, "add_noise_class", False),
            noise_class_id=getattr(self, "noise_class_id", None),
        )


    def _mix_one(
        self,
        w: WindowKey,
        rng: np.random.Generator,
        override_snr_db: float | None = None,
        force_stream_snr_db: float | None = None,
    ):
        """
        Build (I,Q) for a single window w, then add noise.

        override_snr_db:
            lets caller say "pretend this window's snr is X" (used in resnet_mode balancing)
        force_stream_snr_db:
            NEW: locks the noise environment for this stream. We treat this as the SNR
            context for BOTH pulse frames and noise-only frames so the background floor
            is consistent across the stream.
        """
        xc = _mix_window_from_key(w, self.emitters)  # complex (W,)

        # What SNR are we targeting?
        if force_stream_snr_db is not None:
            snr_db = float(force_stream_snr_db)
        elif override_snr_db is not None:
            snr_db = float(override_snr_db)
        else:
            snr_db = float(w.snr)

        # Decide how to add noise.
        # We IGNORE the old special-case branch that used noise_power_db_range to make
        # super-quiet "blank" frames. Instead, EVERY frame gets noise consistent with
        # snr_db above. That keeps the floor visually continuous across the stream.
        #
        # We still respect self.noise_fn if provided.
        if self.noise_fn is None:
            # Fall back to AWGN with target snr_db.
            noisy_complex = _add_awgn_to_snr(xc, snr_db, rng)
            I_noisy = noisy_complex.real.astype(np.float32, copy=False)
            Q_noisy = noisy_complex.imag.astype(np.float32, copy=False)

        else:
            # We'll just push the same snr_db through noise_fn.
            # We *do not* drop into the "len(w.mods)==0" branch with random noise power,
            # because that caused unrealistically quiet gaps.
            I_noisy, Q_noisy = self.noise_fn(
                rng,
                xc.real.astype(np.float32, copy=False),
                xc.imag.astype(np.float32, copy=False),
                snr_db=snr_db,
                noise_power_db=self.noise_power_db,
            )

            I_noisy = np.asarray(I_noisy, dtype=np.float32)
            Q_noisy = np.asarray(Q_noisy, dtype=np.float32)

        x = np.stack([I_noisy, Q_noisy], axis=0)  # (2, W)

        # Labels stay the same logic as before.
        y = _labels_from_key_time_threshold(
            w,
            self.emitters,
            self.class_to_id,
            self.threshold_time_frac,
            add_noise_class=getattr(self, "add_noise_class", False),
            noise_class_id=getattr(self, "noise_class_id", None),
        )
        return x, y


    def _is_noise_at_end(self, sid: int, end_idx: int) -> bool:
        """Return True if the last (target) window for (sid,end_idx) is 'noise'."""
        w = self.streams[sid][end_idx]
        y = _labels_from_key_time_threshold(
            w,
            self.emitters,
            self.class_to_id,
            self.threshold_time_frac,
            add_noise_class=getattr(self, "add_noise_class", False),
            noise_class_id=getattr(self, "noise_class_id", None),
        )
        if getattr(self, "add_noise_class", False) and (
            self.noise_class_id is not None
        ):
            return bool(y[self.noise_class_id] == 1.0)
        else:
            # mods-only setup: noise == "no classes present"
            return bool(y.sum() == 0.0)

    def _build_snr_balanced_index(self, raw_indices: list[tuple[int,int]]) -> list[tuple[int,int]]:  # NEW
        if not self.req_snr_levels:
            return raw_indices
        buckets = {int(s): [] for s in self.req_snr_levels}
        for sid, end in raw_indices:
            s = int(self.stream_snr[sid])
            if s in buckets:
                buckets[s].append((sid, end))
        # cap each bucket to the smallest size for uniformity
        min_len = min((len(v) for v in buckets.values() if len(v) > 0), default=0)
        rng = np.random.default_rng(self.resample_seed or self.seed)
        out = []
        for s, arr in buckets.items():
            if len(arr) == 0: 
                continue
            if len(arr) > min_len:
                arr = rng.choice(arr, size=min_len, replace=False).tolist()
            out.extend(arr)
        rng.shuffle(out)
        return out

    def _build_balanced_index(
        self, raw_indices: list[tuple[int, int]]
    ) -> list[tuple[int, int]]:
        """
        Builds the sampling index with optional pos:noise balancing (global or per_stream)
        and optional SNR balancing across self.req_snr_levels.

        - If self.balance_mode is None: shuffle all, then (optionally) SNR-balance.
        - If "global": cap noise to noise_pos_ratio * #positives, then (optionally) SNR-balance.
        - If "per_stream": cap noise within each stream, then (optionally) SNR-balance.
        - If self.balance_snr and self.req_snr_levels is not None: downsample each requested
        SNR bucket to the size of the smallest non-empty bucket, then shuffle.
        """
        rng = np.random.default_rng(self.resample_seed or self.seed)

        # ----- helper: apply SNR balancing to a prepared index list -----
        def _apply_snr_balance(indices: list[tuple[int, int]]) -> list[tuple[int, int]]:
            # Skip if not requested or attributes not present.
            if not getattr(self, "balance_snr", False):
                return indices
            req = getattr(self, "req_snr_levels", None)
            if not req:
                return indices

            # Bucket by requested SNR levels using stream-level SNR metadata.
            buckets: dict[int, list[tuple[int, int]]] = {int(s): [] for s in req}
            for sid, end in indices:
                s = int(self.stream_snr[sid])
                if s in buckets:
                    buckets[s].append((sid, end))

            # Keep only non-empty buckets; if 0 or 1 buckets have data, nothing to balance.
            non_empty = [b for b in buckets.values() if len(b) > 0]
            if len(non_empty) <= 1:
                return indices

            # Downsample each non-empty bucket to the smallest size for uniform SNR.
            min_len = min(len(b) for b in non_empty)
            out: list[tuple[int, int]] = []
            for lvl, arr in buckets.items():
                if len(arr) == 0:
                    continue
                if len(arr) > min_len:
                    arr = rng.choice(arr, size=min_len, replace=False).tolist()
                out.extend(arr)

            rng.shuffle(out)
            return out

        # ------------------------------ no balancing ------------------------------
        if self.balance_mode is None:
            idx = raw_indices[:]  # copy
            rng.shuffle(idx)
            return _apply_snr_balance(idx)

        # ------------------------------ global balance ----------------------------
        if self.balance_mode == "global":
            pos: list[tuple[int, int]] = []
            neg: list[tuple[int, int]] = []
            for sid, end in raw_indices:
                (neg if self._is_noise_at_end(sid, end) else pos).append((sid, end))

            # Cap noise to the requested ratio relative to positives.
            max_noise = int(self.noise_pos_ratio * max(1, len(pos)))
            if len(neg) > max_noise:
                neg = rng.choice(neg, size=max_noise, replace=False).tolist()

            out = pos + neg
            rng.shuffle(out)
            return _apply_snr_balance(out)

        # ---------------------------- per-stream balance --------------------------
        if self.balance_mode == "per_stream":
            by_sid: dict[int, list[int]] = {}
            for sid, end in raw_indices:
                by_sid.setdefault(sid, []).append(end)

            out: list[tuple[int, int]] = []
            for sid, ends in by_sid.items():
                pos_ends: list[int] = []
                neg_ends: list[int] = []
                for end in ends:
                    (neg_ends if self._is_noise_at_end(sid, end) else pos_ends).append(end)

                if not pos_ends:
                    # No positives in this stream: keep a tiny sample of noise so the stream isn't lost entirely.
                    keep = min(len(neg_ends), 8)
                    if keep > 0:
                        out.extend((sid, e) for e in rng.choice(neg_ends, size=keep, replace=False).tolist())
                    continue

                max_noise = int(self.noise_pos_ratio * len(pos_ends))
                if len(neg_ends) > max_noise:
                    neg_ends = rng.choice(neg_ends, size=max_noise, replace=False).tolist()

                out.extend((sid, e) for e in pos_ends)
                out.extend((sid, e) for e in neg_ends)

            rng.shuffle(out)
            return _apply_snr_balance(out)

        raise ValueError(f"Unknown balance_mode={self.balance_mode!r}")

    def __getitem__(self, idx: int):
        sid, end_idx = self.index[idx]
        st = self.streams[sid]
        L = len(st)
        T = self.segment_len
        W = self.W
        C = self.C

        if self.mode == "resnet":
            w = st[end_idx]
            rng = np.random.default_rng([self.seed, sid, end_idx, 0])

            # We still allow req_snr_levels to override, BUT we keep a stable noise
            # floor per stream. Priority:
            #   1. req_snr_levels if provided (because you explicitly asked for SNR balancing)
            #   2. otherwise the stream's own baseline
            stream_snr_db = self.stream_noise_db[sid]

            if self.req_snr_levels:
                override_snr = float(
                    self.req_snr_levels[rng.integers(0, len(self.req_snr_levels))]
                )
                force_snr = override_snr
            else:
                force_snr = stream_snr_db

            x_i, y_i = self._mix_one(
                w,
                rng,
                override_snr_db=None,
                force_stream_snr_db=force_snr,
            )

            k = int(y_i.sum())
            if self.drop_ambiguous:
                is_zero_hot_as_noise = (k == 0) and self.add_noise_class and (self.noise_class_id is not None)
                if (k > 1) or ((k == 0) and not is_zero_hot_as_noise):
                    j = (idx + 1) % len(self.index)
                    return self.__getitem__(j)

            if k == 0 and self.add_noise_class and (self.noise_class_id is not None):
                y_int = int(self.noise_class_id)
            elif k >= 1:
                y_int = int(y_i.argmax())
            else:
                y_int = int(self.noise_class_id) if (self.add_noise_class and self.noise_class_id is not None) else int(y_i.argmax())

            return torch.from_numpy(x_i), torch.tensor(y_int, dtype=torch.long)

        # Determine start_idx, pad if not enough history
        start_idx = max(0, end_idx - (T - 1))
        real_len = end_idx - start_idx + 1
        pad = T - real_len

        # Prepare arrays
        x_seq = np.zeros((T, 2, W), dtype=np.float32)
        y_seq = np.zeros((T, C), dtype=np.float32)
        mask = np.zeros((T,), dtype=bool)

        # Fill real frames (align to the RIGHT; left-pad = 0..pad-1)
        rng = np.random.default_rng([self.seed, sid, end_idx])
        t_ptr = pad

        # NEW: grab this stream's fixed noise/SNR context
        stream_snr_db = self.stream_noise_db[sid]

        for i in range(start_idx, end_idx + 1):
            w = st[i]
            x_i, y_i = self._mix_one(
                w,
                rng,
                override_snr_db=None,
                force_stream_snr_db=stream_snr_db,
            )
            x_seq[t_ptr] = x_i
            y_seq[t_ptr, :] = _labels_from_key_time_threshold(
                w,
                self.emitters,
                self.class_to_id,
                self.threshold_time_frac,
                add_noise_class=self.add_noise_class,
                noise_class_id=self.noise_class_id,
            )
            mask[t_ptr] = True
            t_ptr += 1

        # Main target is the last timestep
        y_last = y_seq[-1].copy()

        meta = {
            "stream_id": sid,
            "end_idx": end_idx,
            "K": self.stream_k[sid],
            "perm": self.stream_perm[sid],
            "snr": self.stream_snr[sid],
            "real_len": int(real_len),
            "pad": int(pad),
        }

        return (
            torch.from_numpy(x_seq),  # (T, 2, W)
            torch.from_numpy(y_seq),  # (T, C)
            torch.from_numpy(y_last),  # (C,)
            torch.from_numpy(mask),  # (T,)
            meta,
        )

    def _compute_last_counts(self, use_balanced: bool = True):
        idx = (
            self.index
            if use_balanced
            else [
                (sid, end)
                for sid, ends in {i: [] for i in range(len(self.streams))}.items()
            ]
        )  # (if you kept raw_indices somewhere, use that instead)
        C = len(self.class_to_id)
        counts = np.zeros((C,), dtype=np.int64)
        total = 0
        for sid, end_idx in idx:
            y = self._label_last(sid, end_idx)
            counts += y.astype(np.int64)
            total += 1
        self.class_counts_last = torch.from_numpy(counts)
        self.total_last = total

    def get_pos_weight(self, epsilon: float = 1.0) -> torch.Tensor:
        """
        Returns pos_weight[c] = (#negatives) / (#positives) for last-step labels,
        with a small epsilon to avoid div-by-zero for ultra-rare classes.
        """
        if self.class_counts_last is None:
            self._compute_last_counts(use_balanced=True)
        pos = self.class_counts_last.to(torch.float32)
        total = float(self.total_last)
        neg = total - pos
        return neg / (pos + epsilon)


def assert_combo_multiples(k_streams, modulation_list, snr_range):
    C = len(modulation_list)
    S = len(snr_range)
    n_pairs = math.comb(C, 2)
    n_trips = math.comb(C, 3)
    n2 = len(k_streams.get(2, []))
    n3 = len(k_streams.get(3, []))
    if n2 and (n2 % (S * n_pairs) != 0):
        raise AssertionError(f"K=2={n2} not multiple of S*n_pairs={S*n_pairs}")
    if n3 and (n3 % (S * n_trips) != 0):
        raise AssertionError(f"K=3={n3} not multiple of S*n_trips={S*n_trips}")


def collate_stream_segments(batch):
    """
    batch: list of (x_seq, y_seq, y_last, mask, meta)
    Returns:
      xs:   (B, T, 2, W)
      ys:   (B, T, C)
      yend: (B, C)
      msk:  (B, T)
      meta: list[dict]
    """
    xs, ys, yend, msk, metas = zip(*batch)
    xs = torch.stack(xs, dim=0)
    ys = torch.stack(ys, dim=0)
    yend = torch.stack(yend, dim=0)
    msk = torch.stack(msk, dim=0)
    return xs, ys, yend, msk, list(metas)


def collate_stream_segments_to_images(batch):
    """
    batch: list of (x_seq, y_seq, y_last, mask, meta)
      x_seq: (T, 2, W) torch.Tensor  or sometimes (T, W, 2)
    Returns:
      xpow: (B*T, 3, 224, 224) float32 torch.Tensor
      xpha: (B*T, 3, 224, 224) float32 torch.Tensor
      ys  : (B, T, C)
      yend: (B, C)
      msk : (B, T)
      meta: list[dict]
    """
    xs, ys, yend, msk, metas = zip(*batch)  # xs: list of (T, 2, W) tensors

    B = len(xs)
    # Infer T from first element; normalise shape to (T, 2, W) torch.float32
    def _norm_x(x):
        # ensure tensor on CPU
        if not torch.is_tensor(x):
            x = torch.as_tensor(x)
        x = x.detach()  # just in case
        if x.ndim != 3:
            raise ValueError(f"expected 3D tensor for x_seq, got shape {tuple(x.shape)}")
        # Accept (T, 2, W) or (T, W, 2)
        if x.shape[1] == 2:
            # (T, 2, W)
            pass
        elif x.shape[-1] == 2:
            # (T, W, 2) -> (T, 2, W)
            x = x.movedim(-1, 1)
        else:
            raise ValueError(f"x_seq must have a 2-channel IQ dim; got shape {tuple(x.shape)}")
        return x.contiguous().to(torch.float32)

    xs = tuple(_norm_x(x) for x in xs)
    T = xs[0].shape[0]
    W = xs[0].shape[-1]

    power_imgs = []
    phase_imgs = []

    # Loop over B, T and produce (3,224,224) images as torch.float32
    for b in range(B):
        x_seq = xs[b]  # (T, 2, W) torch.float32
        for t in range(T):
            iq = x_seq[t]  # (2, W) torch.float32  (torch.Tensor! not numpy)

            # ---- POWER ----
            p_spec = iq_to_power_spectrogram(
                iq,
                n_fft=64, win_len=64, hop_len=32,
                normalise_noise=True, to_db=True,
                noise_floor_mode="fixed", noise_floor_db=1.0,
                db_min=-100.0, db_max=20.0,
                return_mode="snr01",
            )
            p_img = spec_to_resnet_input(p_spec, size=224, channels="tile3")  # -> (3,224,224)
            if not torch.is_tensor(p_img):
                p_img = torch.from_numpy(p_img)
            power_imgs.append(p_img.to(torch.float32, copy=False))

            # ---- PHASE-EDGE ----
            q_spec = iq_to_phase_spectrogram(iq, return_mode="snr01")
            q_img = spec_to_resnet_input(q_spec, size=224, channels="tile3")  # -> (3,224,224)
            if not torch.is_tensor(q_img):
                q_img = torch.from_numpy(q_img)
            phase_imgs.append(q_img.to(torch.float32, copy=False))

    xpow = torch.stack(power_imgs, dim=0).contiguous()  # (B*T, 3, 224, 224)
    xpha = torch.stack(phase_imgs, dim=0).contiguous()  # (B*T, 3, 224, 224)

    ys   = torch.stack(ys,   dim=0)
    yend = torch.stack(yend, dim=0)
    msk  = torch.stack(msk,  dim=0)
    return xpow, xpha, ys, yend, msk, list(metas)



class MockModel(nn.Module):
    def __init__(self, C):
        super().__init__()
        # toy projection from (2,W) -> d, then a Transformer-like block would go here
        self.pool = nn.AdaptiveAvgPool1d(64)  # (B,T,2,W)->(B,T,2,64)
        self.head = nn.Linear(2 * 64, C)  # simple head per step

    def forward(self, x, key_padding_mask=None):
        # x: (B,T,2,W)
        B, T, Cin, W = x.shape
        x = x.reshape(B * T, Cin, W)
        x = self.pool(x)  # (B*T,2,64)
        x = x.flatten(1)  # (B*T, 128)
        logits = self.head(x).reshape(B, T, -1)  # (B,T,C)
        return logits


def add_noise_as_k1_class(
    k_streams: dict[int, list[list[WindowKey]]],
    *,
    num_classes_without_noise: int,   # e.g., len(modulation_list)
    snr_range: list[int],
    window_count: int,
    window_size: int,
    window_step_size: int,
    split: str = "train",
) -> dict[int, list[list[WindowKey]]]:
    """
    Make 'noise' have the same K=1 representation as any single modulation class.
    We infer how many K=1 streams there are per modulation by averaging, then
    create that many pure-noise (K=0) streams and attach them to K=1 bucket
    with an explicit 'noise' marker in metadata (so downstream code can see it).

    Note: We *do not* remove or alter existing streams.
    """

    k_streams = dict(k_streams)     # shallow copy
    k1_streams = k_streams.get(1, [])
    if not k1_streams or num_classes_without_noise <= 0:
        return k_streams

    # Approximate “per-class” stream count by average
    total_k1 = len(k1_streams)
    target_per_class = max(1, int(round(total_k1 / num_classes_without_noise)))

    # Build that many pure-noise streams
    noise_streams = []
    for snr in snr_range:
        for _ in range(target_per_class):
            ws, we = 0, window_size
            st = []
            for _w in range(window_count):
                st.append(WindowKey(
                    split=split,
                    start=ws, end=we, win_len=window_size,
                    mods=[],              # <- EMPTY mods => pure noise
                    snr=int(snr)
                ))
                ws += window_step_size
                we += window_step_size
            noise_streams.append(st)

    # We’ll store these in the K=1 list to keep the loader balance simple
    # (they are still distinguishable because len(mods)==0 if you want to check).
    k_streams.setdefault(1, []).extend(noise_streams)
    return k_streams

def _print_repetition_raster(stream, emitters, max_windows: int = 64, mode: str = "presence"):
    """
    mode: "presence" -> 0/1; "counts" -> number of pulses per window (clamped 0..9)
    """
    Wmax = min(len(stream), max_windows)
    header = "win  " + " ".join([f"{i:02d}" for i in range(Wmax)])
    print("\n=== PRI repetition raster ({} per window) ===".format("presence" if mode=="presence" else "counts"))
    print(header)
    print("-" * len(header))

    if mode == "presence":
        pres = _stream_presence_by_emitter(stream, max_windows=Wmax)
        # print first up to 8 rows to keep compact
        for (mname, em_id), row in list(pres.items())[:8]:
            print(f"{mname[:8]:>8}  " + " ".join(str(int(v)) for v in row))
    else:
        # counts mode
        # Build unique (mod,emitter) order from first windows
        uniq = []
        seen = set()
        for w in stream:
            for m in w.mods:
                key = (m.mod, m.emitter)
                if key not in seen:
                    seen.add(key); uniq.append(key)
            if len(uniq) >= 8:
                break
        for (mname, em_id) in uniq:
            row = []
            for w in stream[:Wmax]:
                c = sum(1 for mm in w.mods if (mm.mod == mname and mm.emitter == em_id))
                row.append(c)
            print(f"{mname[:8]:>8}  " + " ".join(str(min(9, c)) for c in row))

def _stream_has_repetition(stream) -> bool:
    """
    Heuristic: counts how many pulse *instances* of (mod, emitter) appear across windows.
    If any (mod,emitter) appears in >= 2 windows, we consider it a 'repeated pulse' stream.
    """
    counts = Counter()
    for w in stream:
        # Count at most once per window per (mod,emitter) to avoid double-counting
        # multiple pulses of the same emitter within the same window.
        seen_in_win = set()
        for m in w.mods:
            key = (m.mod, m.emitter)
            seen_in_win.add(key)
        counts.update(seen_in_win)
    return any(v >= 2 for v in counts.values())


def _stream_presence_by_emitter(stream, max_windows: int | None = None):
    """
    Returns: dict[(mod, emitter)] -> list[int] presence per window (0/1).
    Presence is 1 if that (mod,emitter) has >= 1 pulse intersecting the window.
    """
    Wmax = len(stream) if max_windows is None else min(max_windows, len(stream))
    pres = defaultdict(lambda: [0] * Wmax)
    for i in range(Wmax):
        w = stream[i]
        seen = set()
        for m in w.mods:
            key = (m.mod, m.emitter)
            seen.add(key)
        for key in seen:
            pres[key][i] = 1
    return pres


def _emitter_pri_samples(dataset, key):
    """
    key = (mod_name, emitter_global_id). Returns PRI (samples) if available, else None.
    """
    try:
        _, em_id = key
        em = dataset.emitters[em_id]
        # assume 'pri' already stored in samples; adapt if your field differs
        return int(em.get("pri", 0))
    except Exception:
        return None


def _best_repetition_candidate_stream(stream):
    """
    Score a stream by the maximum presence count among its (mod,emitter) rows.
    Returns (best_score, best_key, presence_row).
    """
    pres = _stream_presence_by_emitter(stream)
    best_score, best_key, best_row = 0, None, None
    for key, row in pres.items():
        score = sum(row)  # number of 1's across windows
        if score > best_score:
            best_score, best_key, best_row = score, key, row
    return best_score, best_key, best_row


def print_repetition_raster_for_dataset(
    dataset,
    max_windows: int = 64,
    min_hits: int = 2,
    prefer_short_pri: bool = True,
    fallback_ok: bool = True,
    mode: str = "presence",
):
    """
    Picks a stream that visibly repeats (any (mod,emitter) has >= min_hits '1's).
    - prefer_short_pri: if true and PRI info exists, try streams where the 'best' repeating emitter
      has shorter PRI first.
    - fallback_ok: if nothing repeats, prints the 'best available' stream (max hits).
    """
    if not getattr(dataset, "streams", None):
        print("No streams available in dataset.")
        return

    candidates = []
    best_overall = (0, None, None, None)  # (hits, stream, key, row)

    # Score each stream and collect repeating ones
    for s in dataset.streams:
        hits, key, row = _best_repetition_candidate_stream(s)
        if hits is None:
            continue
        # Track best overall anyway
        if hits > best_overall[0]:
            best_overall = (hits, s, key, row)
        if hits >= min_hits:
            pri = _emitter_pri_samples(dataset, key) if prefer_short_pri else None
            candidates.append((hits, pri if pri is not None else 1_000_000_000, s, key, row))

    chosen = None
    if candidates:
        # Sort: most hits desc, then shortest PRI asc
        candidates.sort(key=lambda t: (-t[0], t[1]))
        _, _, s, key, row = candidates[0]
        chosen = s
    elif fallback_ok and best_overall[1] is not None:
        # Use best available; warn
        hits, s, key, row = best_overall
        print(f"(No stream reached min_hits={min_hits}; showing best available with hits={hits})")
        chosen = s

    if chosen is None:
        print("No suitable stream found to display.")
        return

    _print_repetition_raster(chosen, dataset.emitters, max_windows=max_windows, mode=mode)


def prepMultiHotStreamLoader(
    modulation_list: list[str],
    snr_range: list[int],
    filepath: str,
    batch_size: int,
    history_length: int,
    window_count: int,
    window_size: int,
    window_step_size: int,
    k_ratio: dict[int, float],
    seed: int,
    rng,
    split: str = "train",
    stride: int = 1,
    threshold_time_frac: float = 0.02,
    noise_fn=add_noise,
    add_noise_class=True,
    noise_label="noise",
    balance_mode="per_stream",
    noise_pos_ratio=1.0,
    resample_seed=None,
    shuffle=True,
    num_workers=2,
    persistent_workers=True,
    prefetch_factor=2,
    collate_fn=collate_stream_segments,
    image_mode: bool = True,
    return_keys: bool = False,
    pin_memory: bool = False,
    # ---------------- NEW ----------------
    resnet_mode: bool = False,
    drop_ambiguous: bool = False,
    snr_levels: tuple[int, ...] | None = None,
    balance_snr: bool = False,
    # ---------------- NEW NEW ----------------
    emitter_keep_by_mod: dict[str, list[int]] | None = None,  # <--- ADD THIS
    # Prints
    progress_steps: bool = True,
    summary_prints: bool = True,
):
    # For ResNet mode we DO NOT use image collation; encoder builds its own spectrograms.
    if (not resnet_mode) and image_mode:
        collate_fn = collate_stream_segments_to_images
    # else:
    #     collate_fn = None  # default PyTorch collate

    if progress_steps: 
        print("Emitters")
    emitters, index_map = load_emitters_from_mod_list(
        filepath, modulation_list, full_data=False
    )

    # -------- NEW: filter emitters by modulation (used for 80/20 split) --------
    if emitter_keep_by_mod is not None:
        new_emitters = []
        new_index_map = {}
        cursor = 0

        for mod in modulation_list:
            s, e = index_map[mod]
            block = emitters[s:e]
            keep = emitter_keep_by_mod.get(mod, list(range(len(block))))
            # keep should be stable order
            keep = sorted(keep)
            block_kept = [block[i] for i in keep]

            new_index_map[mod] = (cursor, cursor + len(block_kept))
            new_emitters.extend(block_kept)
            cursor += len(block_kept)

        emitters = new_emitters
        index_map = new_index_map


    if progress_steps: 
        print("K = 1")
    k1_streams = build_k1_streams_all_emitters(
        emitters,
        modulation_list,
        snr_range,
        window_count,
        window_size,
        window_step_size,
        split,
        index_map,
        seed=seed,
    )
    n_k1 = len(k1_streams)

    pairs_ordered = list(itertools.combinations(modulation_list, 2))
    trips_ordered = list(itertools.combinations(modulation_list, 3))

    any_mod = pairs_ordered[0][0] if pairs_ordered else modulation_list[0]
    s, e = index_map[any_mod]
    E_base = e - s

    if progress_steps: 
        print("K = 2")
    pairs_map = global_partition_k2(E_base, pairs_ordered, seed=seed)
    n_pairs_per_perm_raw = {
        "len_per_perm": len(next(iter(pairs_map.values()))),
        "num_perms": len(pairs_ordered),
    }

    if progress_steps: 
        print("K = 3")
    trips_map = global_partition_k3(E_base, trips_ordered, seed=seed)
    n_trips_per_perm_raw = {
        "len_per_perm": len(next(iter(trips_map.values()))),
        "num_perms": len(trips_ordered),
    }

    if progress_steps: 
        print("Target positions")
    n_pos2, n_pos3, k2_streams_target, k3_streams_target = compute_targets_positions(
        n_k1,
        k_ratio,
        snr_count=len(snr_range),
        n_pairs_per_perm_raw=n_pairs_per_perm_raw,
        n_trips_per_perm_raw=n_trips_per_perm_raw,
        loose_round_up=True,
    )

    k2_streams = build_k2_from_positions(
        pairs_map,
        n_pos2,
        emitters,
        snr_range,
        window_count,
        window_size,
        window_step_size,
        split,
        index_map=index_map,
        seed=seed,
    )

    k3_streams = build_k3_from_positions(
        trips_map,
        n_pos3,
        emitters,
        snr_range,
        window_count,
        window_size,
        window_step_size,
        split,
        index_map=index_map,
        seed=seed,
    )

    k_streams = {1: list(k1_streams.values()), 2: k2_streams, 3: k3_streams}

    # NEW: only add noise class when requested
    if add_noise_class:
        k_streams = add_noise_as_k1_class(
            k_streams,
            num_classes_without_noise=len(modulation_list),
            snr_range=snr_range,
            window_count=window_count,
            window_size=window_size,
            window_step_size=window_step_size,
            split=split,
        )

    assert_combo_multiples(k_streams, modulation_list, snr_range)
    keys_json = flatten_k_streams(k_streams)
    if summary_prints: 
        sanity_counts_text(k_streams, window_count=window_count)

    # -------- dataset / loader --------
    if resnet_mode:
        ds = MultiHotStreamSegmentDataset(
            k_streams,
            emitters,
            segment_len=1,
            end_stride=stride,
            threshold_time_frac=threshold_time_frac,
            seed=seed,
            noise_fn=noise_fn,
            add_noise_class=add_noise_class,
            noise_label=noise_label,
            balance_mode=balance_mode,
            noise_pos_ratio=noise_pos_ratio,
            resample_seed=resample_seed,
            # NEW:
            mode="resnet",
            drop_ambiguous=drop_ambiguous,
            snr_levels=snr_levels,
            balance_snr=balance_snr,
        )
    else:
        ds = MultiHotStreamSegmentDataset(
            k_streams,
            emitters,
            segment_len=history_length,
            end_stride=stride,
            threshold_time_frac=threshold_time_frac,
            seed=seed,
            noise_fn=noise_fn,
            add_noise_class=add_noise_class,
            noise_label=noise_label,
            balance_mode=balance_mode,
            noise_pos_ratio=noise_pos_ratio,
            resample_seed=resample_seed,
            # default mode="sequence"
        )

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
    )

    if return_keys:
        return (ds, loader, keys_json)
    return ds, loader


##### MAIN #####
if __name__ == "__main__":

    print("Running")
    start_time = time()

    seed = 42
    rng = np.random.default_rng(seed)

    # split = "train"
    split = "test"
    filepath = f"C:\\Apps\\Code\\AIMC_Spec_4\\{split}\\"

    modulation_list = [
        # FM
        "unmodulated",
        "lfm_up",
        "lfm_down",
        "dlfm_up_down",
        # "NLFM_up",
        # "NLFM_down",
        # "mlfm",
        # "dlfm_down_up",
        # PM
        # "bpsk",
        # "qpsk",
        # "barker_11",
        # "barker_13",
        # "p1",
        # "p2",
        # "p3",
        # "p4",
    ]

    snr_range = [
        0,
    ]

    sample_rate_hz = 50_000_000

    window_size = 1000  # 20 us
    window_step_size = 1000
    window_count = 100  # or derive from seconds_per_emitter, etc.

    # Distribution over K (after warm-up)
    # k_ratio = {1: 0.75, 2: 0.2, 3: 0.05}
    k_ratio = {1: 1.0}

    threshold = 0.02
    # theshold = 0.1

    batch_size = 1
    history_length = 64
    # history_length = 16

    dataset, loader = prepMultiHotStreamLoader(
        modulation_list=modulation_list,
        snr_range=snr_range,
        filepath=filepath,
        batch_size=batch_size,
        history_length=history_length,
        window_count=window_count,
        window_size=window_size,
        window_step_size=window_step_size,
        k_ratio=k_ratio,
        seed=seed + 1,
        rng=np.random.default_rng(seed + 1),
        split="val",
        stride=1,
        threshold_time_frac=threshold,
        add_noise_class=True,
        noise_label="noise",
        balance_mode=None,
        noise_pos_ratio=1.0,
        resample_seed=None,
        shuffle=False,
        num_workers=4,
        persistent_workers=False,
        prefetch_factor=4,
        collate_fn=None,
        image_mode=False,
        return_keys=False,
        pin_memory=True,
        resnet_mode=True,
        drop_ambiguous=True,
        snr_levels=tuple(snr_range),
        balance_snr=False,
        progress_steps=False,
        summary_prints=False,
    )

    print_repetition_raster_for_dataset(
        dataset,
        max_windows=64,
        min_hits=2,            # <-- ensures at least two visible '1's
        prefer_short_pri=True, # <-- favors streams whose repeating emitter has shorter PRI
        fallback_ok=True,
        mode="presence"        # "presence" shows 0/1 like your example
    )

    print(f"Time taken to prep data set & loader: {time() - start_time}")