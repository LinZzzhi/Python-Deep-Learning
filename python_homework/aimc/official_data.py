"""Small demonstrations using downloaded official AIMC-Spec v2 pulses."""

import ctypes
import ctypes.util
from functools import lru_cache
import hashlib
from pathlib import Path

import h5py
import numpy as np

from .engine import write_json
from .resources import ResourceConfig, ResourceGovernor

try:
    import hdf5plugin  # Registers optional filters when installed.
except ImportError:
    hdf5plugin = None

OFFICIAL_REF = "sebastiancocks/aimc-spec-v2"
OFFICIAL_SNR = [-6, -3, 0, 3, 6]


class InvalidPulseError(ValueError):
    pass


@lru_cache(maxsize=1)
def _blosc_codec():
    library = ctypes.util.find_library("blosc")
    if library is None:
        raise RuntimeError("Install hdf5plugin to read Blosc-compressed HDF5.")
    codec = ctypes.CDLL(library)
    codec.blosc_cbuffer_validate.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.POINTER(ctypes.c_size_t)]
    codec.blosc_cbuffer_validate.restype = ctypes.c_int
    codec.blosc_decompress_ctx.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
    codec.blosc_decompress_ctx.restype = ctypes.c_int
    return codec


def read_array(dataset):
    if h5py.h5z.filter_avail(32001) or "32001" not in dataset._filters:
        return dataset[:]
    # HDF5 exposes raw chunks; the system Blosc library decodes the actual data.
    codec = _blosc_codec()
    creation = dataset.id.get_create_plist()
    filters = [creation.get_filter(i)[0] for i in range(creation.get_nfilters())]
    if dataset.ndim != 1 or dataset.chunks is None or any(f not in {2, 32001} for f in filters):
        raise ValueError("Unsupported filter chain; install hdf5plugin for this file.")
    result = np.empty(dataset.shape, dtype=dataset.dtype)
    for offset in range(0, len(result), dataset.chunks[0]):
        mask, encoded = dataset.id.read_direct_chunk((offset,))
        expected_bytes = dataset.chunks[0] * dataset.dtype.itemsize
        for index in reversed(range(len(filters))):
            if mask & (1 << index):
                continue
            if filters[index] == 32001:
                source = ctypes.create_string_buffer(encoded)
                decoded_size = ctypes.c_size_t()
                if codec.blosc_cbuffer_validate(source, len(encoded), ctypes.byref(decoded_size)) != 0:
                    raise ValueError("Invalid Blosc chunk.")
                if decoded_size.value != expected_bytes or expected_bytes > 1024 * 1024:
                    raise ValueError("Unexpected decompressed chunk size.")
                decoded = ctypes.create_string_buffer(expected_bytes)
                if codec.blosc_decompress_ctx(source, decoded, expected_bytes, 1) != expected_bytes:
                    raise ValueError("Blosc decompression failed.")
                encoded = decoded.raw
            else:
                encoded = np.frombuffer(encoded, dtype=np.uint8).reshape(
                    dataset.dtype.itemsize, -1).T.copy().tobytes()
        if len(encoded) != expected_bytes:
            raise ValueError("Unexpected raw chunk size.")
        values = np.frombuffer(encoded, dtype=dataset.dtype)
        count = min(len(values), len(result) - offset)
        result[offset:offset + count] = values[:count]
    return result


def clean_pulse(group):
    real, imag = read_array(group["I"]), read_array(group["Q"])
    if real.shape != imag.shape or real.ndim != 1 or not 1 <= len(real) <= 1000:
        raise InvalidPulseError(f"Unexpected IQ shape in {group.name}.")
    iq = np.stack([real, imag]).astype(np.float32)
    if not np.isfinite(iq).all() or not np.any(iq):
        raise InvalidPulseError(f"Invalid clean pulse in {group.name}.")
    return iq


def noisy_window(clean, snr_db, seed, window_size=1000):
    power = clean[0].astype(np.float64) ** 2 + clean[1].astype(np.float64) ** 2
    power = np.mean(power[power > 0])
    window = np.zeros((2, window_size), dtype=np.float32)
    window[:, :clean.shape[1]] = clean * np.sqrt(10 ** (snr_db / 10) / power)
    rng = np.random.default_rng(seed)
    window += rng.standard_normal(window.shape).astype(np.float32) / np.sqrt(2)
    return window


def prepare_small_official(raw_dir, out_dir, train_per_class=100, val_per_class=25,
                           test_per_class=50, seed=42, resources=None):
    raw_dir, out_dir = Path(raw_dir), Path(out_dir)
    classes = sorted(path.stem for path in raw_dir.glob("*.h5"))
    if len(classes) != 30:
        raise ValueError(f"Expected 30 official class files, found {len(classes)}.")
    counts = {"train": train_per_class, "val": val_per_class, "test": test_per_class}
    if min(counts.values()) < 1:
        raise ValueError("Every split must contain pulses from every class.")
    governor = ResourceGovernor(resources or ResourceConfig(threads=1), out_dir / "preparation")
    buffers = {split: {"iq": [], "labels": [], "snr_db": [], "source_emitter": []} for split in counts}
    source_counts, selected_ids = {}, {}
    excluded_invalid, excluded_duplicate = {}, {}
    seen = set()
    for class_id, name in enumerate(classes):
        with h5py.File(raw_dir / f"{name}.h5", "r") as handle:
            keys = sorted(handle.keys(), key=lambda key: int(key.rsplit("_", 1)[1]))
            source_counts[name] = len(keys)
            order = np.random.default_rng([seed, class_id]).permutation(len(keys))
            selected = []
            excluded_invalid[name], excluded_duplicate[name] = [], []
            for start in range(0, len(order), 32):
                with governor.burst():
                    for index in order[start:start + 32]:
                        try:
                            clean = clean_pulse(handle[keys[index]])
                        except InvalidPulseError:
                            excluded_invalid[name].append(keys[index])
                            continue
                        digest = hashlib.sha256(clean.tobytes()).digest()
                        if digest in seen:
                            excluded_duplicate[name].append(keys[index])
                            continue
                        seen.add(digest)
                        selected.append((keys[index], clean))
                        if len(selected) == sum(counts.values()):
                            break
                if len(selected) == sum(counts.values()):
                    break
            if len(selected) != sum(counts.values()):
                raise ValueError(f"Not enough unique official pulses in {name}.")
            selected_ids[name], offset = {}, 0
            for split_id, (split, count) in enumerate(counts.items()):
                chosen = selected[offset:offset + count]
                selected_ids[name][split] = [key for key, _ in chosen]
                offset += count
                with governor.burst():
                    for key, clean in chosen:
                        emitter = int(key.rsplit("_", 1)[1])
                        for snr_index, snr in enumerate(OFFICIAL_SNR):
                            buffers[split]["iq"].append(noisy_window(
                                clean, snr, [seed, class_id, split_id, emitter, snr_index]))
                            buffers[split]["labels"].append(class_id)
                            buffers[split]["snr_db"].append(snr)
                            buffers[split]["source_emitter"].append(emitter)
        print(f"Official class {class_id + 1}/30: {name} prepared", flush=True)
    for split, values in buffers.items():
        with governor.burst():
            order = np.random.default_rng([seed, 100, list(counts).index(split)]).permutation(len(values["labels"]))
            np.savez(out_dir / f"{split}.npz", iq=np.stack(values["iq"])[order],
                     labels=np.asarray(values["labels"], dtype=np.int64)[order],
                     snr_db=np.asarray(values["snr_db"], dtype=np.float32)[order],
                     source_emitter=np.asarray(values["source_emitter"], dtype=np.int32)[order],
                     classes=np.asarray(classes), sample_rate_hz=50_000_000)
    manifest = {"source": "official_AIMC_Spec_v2_test_repartitioned_small_demo",
                "dataset": OFFICIAL_REF, "dataset_version": 1, "classes": classes,
                "raw_dir": str(raw_dir.resolve()), "raw_counts": source_counts,
                "seed": seed, "clean_pulses_per_class": counts, "snr_levels_db": OFFICIAL_SNR,
                "n_samples": 1000, "sample_rate_hz": 50_000_000,
                "split_sizes": {split: count * len(classes) * len(OFFICIAL_SNR) for split, count in counts.items()},
                "protocol": "Official test files repartitioned by clean emitter; all SNRs stay in one split. "
                            "Small demonstration, not the official benchmark protocol.",
                "excluded_invalid_emitters": excluded_invalid,
                "excluded_duplicate_waveforms": excluded_duplicate,
                "selected_emitters": selected_ids}
    write_json(out_dir / "manifest.json", manifest)
    governor.monitor(force=True)
    return manifest
