import ctypes
import ctypes.util
import time

import h5py
import numpy as np
import pytest
import torch

from aimc.official_data import InvalidPulseError, clean_pulse, noisy_window, read_array
from aimc.resources import ResourceConfig, ResourceGovernor


def test_raw_blosc_and_hdf5_shuffle_decode(tmp_path, monkeypatch):
    library = ctypes.util.find_library("blosc")
    if library is None:
        pytest.skip("System Blosc codec is absent.")
    codec = ctypes.CDLL(library)
    codec.blosc_compress_ctx.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_size_t,
                                       ctypes.c_size_t, ctypes.c_void_p, ctypes.c_void_p,
                                       ctypes.c_size_t, ctypes.c_char_p, ctypes.c_size_t, ctypes.c_int]
    codec.blosc_compress_ctx.restype = ctypes.c_int
    expected = np.random.default_rng(7).standard_normal(239).astype(np.float32)
    with h5py.File(tmp_path / "compressed.h5", "w") as handle:
        creation = h5py.h5p.create(h5py.h5p.DATASET_CREATE)
        creation.set_chunk((128,))
        creation.set_shuffle()
        creation.set_filter(32001, h5py.h5z.FLAG_OPTIONAL, (2, 2, 4, 512, 9, 1, 1))
        dataset = h5py.Dataset(h5py.h5d.create(
            handle.id, b"iq", h5py.h5t.py_create(expected.dtype),
            h5py.h5s.create_simple(expected.shape), dcpl=creation))
        for offset in (0, 128):
            chunk = np.zeros(128, dtype=np.float32)
            values = expected[offset:offset + 128]
            chunk[:len(values)] = values
            shuffled = chunk.view(np.uint8).reshape(-1, 4).T.copy().tobytes()
            source = ctypes.create_string_buffer(shuffled)
            destination = ctypes.create_string_buffer(len(shuffled) + 16)
            length = codec.blosc_compress_ctx(9, 0, 4, len(shuffled), source, destination,
                                               len(destination), b"lz4", 0, 1)
            assert length > 0
            dataset.id.write_direct_chunk((offset,), destination.raw[:length], filter_mask=0)
        monkeypatch.setattr(h5py.h5z, "filter_avail", lambda filter_id: False)
        np.testing.assert_array_equal(read_array(dataset), expected)


def test_pulse_only_snr_and_seed():
    clean = np.stack([np.ones(600), np.zeros(600)]).astype(np.float32)
    first = noisy_window(clean, 3, seed=42)
    np.testing.assert_array_equal(first, noisy_window(clean, 3, seed=42))
    signal = np.zeros_like(first)
    signal[:, :600] = clean * np.sqrt(10 ** 0.3)
    measured = 10 * np.log10(np.mean((signal[:, :600] ** 2).sum(0)) /
                             np.mean(((first - signal) ** 2).sum(0)))
    assert abs(measured - 3) < 0.3
    assert np.any(first[:, 600:])


def test_zero_official_pulse_is_rejected(tmp_path):
    with h5py.File(tmp_path / "invalid.h5", "w") as handle:
        group = handle.create_group("emitter_822")
        group.create_dataset("I", data=np.zeros(33, dtype=np.float32))
        group.create_dataset("Q", data=np.zeros(33, dtype=np.float32))
        with pytest.raises(InvalidPulseError, match="Invalid clean pulse"):
            clean_pulse(group)


def test_governor_adds_required_idle(monkeypatch):
    governor = object.__new__(ResourceGovernor)
    governor.config = ResourceConfig(threads=1, duty_cycle=0.25)
    governor.device = torch.device("cpu")
    governor.active_seconds = governor.idle_seconds = 0.0
    monkeypatch.setattr(governor, "monitor", lambda: None)
    times = iter([10.0, 10.04])
    sleeps = []
    monkeypatch.setattr(time, "monotonic", lambda: next(times))
    monkeypatch.setattr(time, "sleep", sleeps.append)
    with governor.burst():
        pass
    assert sleeps[0] == pytest.approx(0.12)
    assert governor.active_seconds / (governor.active_seconds + governor.idle_seconds) == pytest.approx(0.25)
